# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Train a policy to play Missile Command. Run it with ``poe train``.

Everything you would normally want to change is in `TrainConfig` below, and every
field says what it does and why it defaults where it does. The loop itself is
deliberately linear — collect a rollout, update, log, occasionally evaluate and
drop a watchable episode — so it can be read top to bottom.

    poe train                          # sensible defaults
    poe train -- --updates 2000        # train longer
    poe train -- --envs 2048 --record-every 25

Three things are wired in for you, because they are what make a run
interpretable rather than just a number going up:

* **The yardstick.** Every `eval_every` updates the policy is scored on the M4
  protocol — the same 32 seeds, aggregated by the same function as the scripted
  baseline — and printed next to the baseline's 18,036. That is the number that
  decides whether M6 succeeded, so it is also appended to ``runs/evals.csv``
  rather than only scrolling past: it is the one measurement in a run that is in
  the baseline's own units, and the training console draws the baseline across it.
* **Watchable episodes.** Every `record_every` updates one episode is written to
  ``runs/`` as ``update-<n>.mdr``. Open it from the app's REPLAYS menu and watch
  what the policy is actually doing; a reward curve will not tell you that it has
  learned to ignore MIRVs.
* **A run you can stop without losing it.** ``touch runs/STOP`` and the loop
  finishes the update it is on, writes a final checkpoint, flushes the metrics
  and exits; ``touch runs/PAUSE`` blocks it between updates until the file goes
  away. Killing the process instead throws away everything since the last
  checkpoint. See :mod:`md.control` — the training console's buttons write
  exactly these files, and nothing else.
Where ``runs/`` is depends on where you are: the directory beside you in a
checkout, and the per-user data directory once this is installed from a package.
``--out-dir`` and ``$MD_RUNS_DIR`` override, and :mod:`md.paths` has the order.

* **Checkpoints.** Written to ``runs/checkpoints`` — every ``checkpoint_every``
  updates plus a ``policy-final.pt`` at the end, so a short run still leaves the
  policy it trained. Weights only, not optimizer state: these are for scoring or
  watching a past policy, **not** for resuming a run. There is no resume yet.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import modelcard, paths
from .control import Control
from .env import Actions, Flags, Observations, Shaping, VecEnv
from .eval import evaluate, format_summary
from .ppo import ObsLayout, PPOConfig, Rollout, build_policy, update

#: Chooses an action index per environment, given the batch and its action mask.
Policy_fn = Callable[[Observations, Flags], Actions]

#: The scripted baseline (docs/ROADMAP.md, M4). What a learned policy has to beat.
BASELINE_MEAN_SCORE = 18_036.0


@dataclass
class TrainConfig:
    """Everything worth changing, in one place."""

    #: Environments stepped in parallel. Throughput keeps climbing to a few
    #: thousand (see bindings/README.md); the ceiling is usually GPU memory for
    #: the forward pass, not the simulation.
    envs: int = 1024
    #: Agent steps collected per environment before each update. steps * envs is
    #: the batch size — 128 * 1024 = 131k samples per update.
    steps: int = 128
    #: How many updates to run.
    updates: int = 1000
    #: Ticks per agent step. 4 is ~15 decisions a second, near a human's rate.
    frame_skip: int = 4
    #: Episode length cap, in ticks. 120k is ~33 minutes of play.
    max_ticks: int = 120_000
    #: Score the policy on the canonical seeds this often (0 disables).
    eval_every: int = 50
    #: Drop a watchable episode into runs/ this often (0 disables).
    record_every: int = 25
    #: Save a checkpoint this often (0 disables).
    checkpoint_every: int = 100
    seed: int = 0
    #: "cuda", "cpu", or None to pick automatically.
    device: str | None = None
    #: Where the run writes. None means decide at run time — ``./runs`` in a
    #: checkout, the per-user data directory when installed (see `md.paths`).
    out_dir: Path | None = None
    #: Resume from this checkpoint (weights, optimizer and iteration).
    resume: Path | None = None


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config: TrainConfig, ppo: PPOConfig | None = None) -> nn.Module:
    """Run the training loop and return the trained policy."""
    ppo = ppo or PPOConfig()
    device = _device(config.device)
    torch.manual_seed(config.seed)

    env = VecEnv(
        num_envs=config.envs,
        frame_skip=config.frame_skip,
        max_ticks=config.max_ticks,
        shaping=Shaping(),
        seed=config.seed,
    )
    layout = _layout(env)
    policy = build_policy(ppo.architecture, layout, env.action_count, ppo.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate, eps=1e-5)
    rollout = Rollout(config.steps, config.envs, env.obs_size, env.action_count, device)

    first = 1
    if config.resume is not None:
        payload: dict[str, Any] = torch.load(config.resume, map_location=device, weights_only=True)
        policy.load_state_dict(payload["policy"])
        optimizer.load_state_dict(payload["optimizer"])
        first = int(payload["iteration"]) + 1
        print(f"resumed from {config.resume} at update {first}")

    # Environment 0 carries the recordings — one watchable episode at a time.
    if config.record_every > 0:
        env.record(0)
    # ./runs in a checkout, the per-user data directory when installed. Resolved
    # once, here, so every artifact below lands in the same place and the printed
    # paths are the real ones (md/paths.py).
    out_dir = paths.runs_dir(config.out_dir)
    checkpoints = out_dir / "checkpoints"
    shape = (env.obs_size, env.action_count)
    # A CSV alongside the printed line: the terminal is for watching a run, this
    # is for plotting it afterwards. Appended, so a resumed run keeps its history.
    metrics_path = out_dir / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not metrics_path.exists()
    metrics = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(metrics)
    if new_file:
        writer.writerow(
            [
                "update",
                "samples",
                "return",
                "entropy",
                "policy_loss",
                "value_loss",
                "clip_fraction",
                "steps_per_second",
            ]
        )

    # Checked once per update. A run starts running: a STOP left behind by the
    # last one must not kill this one before its first update.
    control = Control(out_dir)
    control.clear()
    _write_config(out_dir / "config.json", config, ppo, out_dir)
    # Beside it, what the run is *training* — the console reads this rather than
    # a checkpoint, because opening one needs torch and it must never import it
    # (docs/ROADMAP.md, M8, risk 3). Written once: within a run the shapes never
    # change, and the iteration is already in each checkpoint's name.
    modelcard.write(
        out_dir,
        modelcard.describe(
            {name: tuple(tensor.shape) for name, tensor in policy.state_dict().items()},
            architecture=ppo.architecture,
            obs_size=shape[0],
            action_count=shape[1],
            hidden=ppo.hidden,
        ),
    )

    print(
        f"training on {device} | {config.envs} envs x {config.steps} steps "
        f"= {config.envs * config.steps:,} samples/update | baseline {BASELINE_MEAN_SCORE:,.0f}"
    )
    print(
        f"  pause with `touch {control.pause_file}`, "
        f"stop gracefully with `touch {control.stop_file}`"
    )

    # Tracks the return of episodes as they finish, for a readable progress line.
    episode_returns: list[float] = []
    running = np.zeros(config.envs, dtype=np.float64)
    started = time.perf_counter()

    for iteration in range(first, first + config.updates):
        for step in range(config.steps):
            obs = torch.from_numpy(env.observations).to(device)
            mask = torch.from_numpy(env.action_masks()).to(device)
            action, log_prob, value = policy.act(obs, mask)

            actions: Actions = action.cpu().numpy().astype(np.int32)
            _, reward, terminated, truncated, info = env.step(actions)
            done: Flags = terminated | truncated

            step_reward = torch.from_numpy(reward).to(device)
            # Truncation is not death. An episode cut off by the tick cap still had
            # a future worth something, so fold gamma * V(final_obs) into its last
            # reward; only true termination is worth zero from here on. Without
            # this, surviving to the cap scores the same as losing there, and the
            # long runs are exactly what we are trying to train toward.
            if truncated.any():
                cut = np.flatnonzero(truncated)
                final = torch.from_numpy(info["final_observation"][cut]).to(device)
                step_reward[cut] += ppo.gamma * policy.value(final)

            rollout.obs[step] = obs
            rollout.masks[step] = mask
            rollout.actions[step] = action
            rollout.log_probs[step] = log_prob
            rollout.values[step] = value
            rollout.rewards[step] = step_reward
            # Zero at any end, truncation included: the next slot is a new episode
            # and the GAE trace must not run across it.
            rollout.continues[step] = torch.from_numpy(~done).to(device).float()

            running += reward
            if done.any():
                episode_returns.extend(running[done].tolist())
                running[done] = 0.0

        with torch.no_grad():
            last_obs: Observations = env.observations
            _, last_value = policy(
                torch.from_numpy(last_obs).to(device),
                torch.from_numpy(env.action_masks()).to(device),
            )
        advantages, returns = rollout.advantages(last_value, ppo.gamma, ppo.gae_lambda)
        stats = update(policy, optimizer, rollout, advantages, returns, ppo)

        recent = episode_returns[-200:]
        elapsed = time.perf_counter() - started
        steps_done = iteration * config.steps * config.envs
        # Early on no episode has finished yet, so there is nothing to average.
        # Say so rather than printing nan, which reads as a bug.
        mean_return = float(np.mean(recent)) if recent else float("nan")
        rate = steps_done / elapsed
        writer.writerow(
            [
                iteration,
                steps_done,
                f"{mean_return:.4f}",
                f"{stats['entropy']:.4f}",
                f"{stats['policy_loss']:.6f}",
                f"{stats['value_loss']:.6f}",
                f"{stats['clip_fraction']:.4f}",
                f"{rate:.1f}",
            ]
        )
        metrics.flush()  # so a plot can follow a run that is still going
        ret = f"{mean_return:>8.2f}" if recent else "       -"
        print(
            f"update {iteration:>5} | return {ret} "
            f"| entropy {stats['entropy']:.3f} | value {stats['value_loss']:.3f} "
            f"| {steps_done / elapsed / 1e3:.0f}k steps/s"
        )

        if config.record_every > 0 and iteration % config.record_every == 0:
            path = out_dir / f"update-{iteration:05d}.mdr"
            if env.save_recording(0, path, update=iteration, label=f"UPDATE {iteration}"):
                print(f"  recorded {path}")

        if config.eval_every > 0 and iteration % config.eval_every == 0:
            summary = _score(policy, device)
            _log_eval(out_dir / "evals.csv", iteration, summary)
            print(format_summary(summary))
            delta = summary.mean_score - BASELINE_MEAN_SCORE
            verdict = "ahead of" if delta > 0 else "behind"
            print(f"  {abs(delta):,.0f} {verdict} the scripted baseline")

        if config.checkpoint_every > 0 and iteration % config.checkpoint_every == 0:
            _save(
                policy,
                optimizer,
                iteration,
                shape,
                ppo,
                layout,
                checkpoints / f"policy-{iteration:05d}.pt",
            )

        # Between updates, never inside one: the rollout and the update are a
        # unit, and stopping in the middle of one leaves a batch half-collected.
        if control.paused():
            print(f"  paused after update {iteration} — remove {control.pause_file} to continue")
            if control.wait_while_paused():
                print("  resumed into a stop request")
            else:
                print("  resumed")
        if control.stopping():
            print(f"  stop requested — finishing after update {iteration}")
            break

    # Always checkpoint the finished policy. Without this a run whose length is
    # not a multiple of checkpoint_every throws away the thing it just trained —
    # and it is what makes a graceful stop cost nothing at all.
    _save(policy, optimizer, iteration, shape, ppo, layout, checkpoints / "policy-final.pt")
    metrics.close()
    control.clear()  # so the next run in this directory is not born stopped
    print(f"  final policy -> {checkpoints / 'policy-final.pt'}")
    print(f"  metrics      -> {metrics_path}")
    return policy


def _write_config(path: Path, config: TrainConfig, ppo: PPOConfig, out_dir: Path) -> None:
    """Record what produced this run, beside what it produced.

    Six months later the checkpoints are still there and the shell history is
    not. Written on every run, not only the ones a console starts, because the
    question "what were the settings" is asked of whichever run turned out to be
    interesting. The *resolved* output directory is recorded rather than the
    ``None`` that asked for it, so the file says where the run actually went.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = dataclasses.asdict(config) | {"out_dir": str(out_dir)}
    payload = {"train": settings, "ppo": dataclasses.asdict(ppo)}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


#: Columns of ``evals.csv`` — the fields of the shared C++ ``Summary``, in order.
EVAL_COLUMNS = (
    "update",
    "mean_score",
    "min_score",
    "max_score",
    "mean_wave",
    "mean_cities_left",
    "mean_accuracy",
    "survived",
    "episodes",
)


def _log_eval(path: Path, iteration: int, summary: Any) -> None:
    """Append one scored evaluation, in the scripted baseline's own units.

    A separate file from ``metrics.csv`` on purpose. That one carries the training
    return, which is shaped, scaled and summed undiscounted — a fine diagnostic,
    but *not* a score, so drawing 18,036 across it would be comparing units that
    have no relationship. These rows are the ones that do compare: same 32 seeds,
    same C++ ``summarize``, greedy play, exactly what ``poe eval`` reports for the
    scripted agent. They are also sparse (one per ``eval_every``), which is
    another reason not to bolt them onto the per-update file as mostly-empty
    columns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(EVAL_COLUMNS)
        writer.writerow(
            [
                iteration,
                f"{summary.mean_score:.2f}",
                summary.min_score,
                summary.max_score,
                f"{summary.mean_wave:.3f}",
                f"{summary.mean_cities_left:.3f}",
                f"{summary.mean_accuracy:.4f}",
                summary.survived,
                summary.episodes,
            ]
        )


def _save(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    shape: tuple[int, int],
    ppo: PPOConfig,
    layout: ObsLayout,
    path: Path,
) -> None:
    """Write a checkpoint that is enough to *resume* from, not just to score.

    That means the optimizer too: Adam carries momentum estimates, and restarting
    without them makes the first few updates after a resume behave nothing like
    the ones before it. The observation/action sizes ride along so a checkpoint
    can be loaded without first constructing an environment to ask.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "obs_size": shape[0],
            "action_count": shape[1],
            "hidden": ppo.hidden,
            # Which network these weights belong to. Absent in checkpoints written
            # before there was a choice, and those are all "mlp" — see load_policy.
            "architecture": ppo.architecture,
            "layout": dataclasses.asdict(layout),
        },
        path,
    )


def _layout(env: VecEnv) -> ObsLayout:
    """The observation's entity layout, for a policy that slices it."""
    spec = env.spec
    return ObsLayout(
        threats=spec.threats,
        interceptors=spec.interceptors,
        blasts=spec.blasts,
        obs_size=env.obs_size,
    )


def load_policy(path: Path, device: torch.device | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a policy from a checkpoint. Returns it alongside the raw payload."""
    device = device or torch.device("cpu")
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    # Checkpoints from before the entity policy existed carry neither key, and
    # every one of them is a flat MLP — so defaulting keeps them loadable.
    architecture = payload.get("architecture", "mlp")
    stored = payload.get("layout")
    layout = (
        ObsLayout(**stored)
        if stored is not None
        else ObsLayout(threats=0, interceptors=0, blasts=0, obs_size=int(payload["obs_size"]))
    )
    policy = build_policy(architecture, layout, payload["action_count"], payload["hidden"]).to(
        device
    )
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy, payload


def greedy_policy(policy: nn.Module, device: torch.device) -> Policy_fn:
    """Wrap a network as the callable `md.eval.evaluate` expects.

    Greedy, because the scripted baseline is deterministic — comparing a sampled
    policy against it would be measuring two different things.
    """

    def act(obs: Observations, mask: Flags) -> Actions:
        with torch.no_grad():
            logits, _ = policy(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))
            return logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    return act


def _score(policy: nn.Module, device: torch.device) -> object:
    """Score the current policy on the canonical seeds, greedily.

    Greedy rather than sampled: the baseline is deterministic, so comparing a
    sampled policy against it would be measuring two different things.
    """

    def act(obs: Observations, mask: Flags) -> Actions:
        with torch.no_grad():
            logits, _ = policy(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))
            return logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    return evaluate(act)


def score_checkpoint(path: Path, device_name: str | None = None, record: Path | None = None) -> int:
    """Score a saved policy on the canonical seeds, without training anything.

    This is how you compare checkpoints: same seeds, same aggregation as the
    scripted baseline, so the numbers sit next to each other honestly. Pass
    `record` to also drop a watchable episode of that policy playing.
    """
    device = _device(device_name)
    policy, payload = load_policy(path, device)
    print(f"loaded {path} (update {payload['iteration']})")

    if record is not None:
        # One env, one seed, recorded start to finish.
        env = VecEnv(num_envs=1, threads=1, shaping=Shaping(), seed=0)
        env.record(0)
        act = greedy_policy(policy, device)
        for _ in range(200_000):
            _, _, terminated, truncated, _ = env.step(act(env.observations, env.action_masks()))
            if terminated[0] or truncated[0]:
                break
        if env.save_recording(0, record, update=int(payload["iteration"]), label=path.stem.upper()):
            print(f"recorded {record}")

    summary = evaluate(greedy_policy(policy, device))
    print(format_summary(summary))
    delta = summary.mean_score - BASELINE_MEAN_SCORE
    print(f"  {abs(delta):,.0f} {'ahead of' if delta > 0 else 'behind'} the scripted baseline")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Missile Command policy with PPO.")
    defaults = TrainConfig()
    parser.add_argument("--envs", type=int, default=defaults.envs)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--frame-skip", type=int, default=defaults.frame_skip)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=defaults.max_ticks,
        help="Episode length cap in ticks; lower it to see episodes finish sooner.",
    )
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--record-every", type=int, default=defaults.record_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where the run writes (default: ./runs in a checkout, else the user data dir).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Continue training from a checkpoint."
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Score a saved policy on the canonical seeds and exit (no training).",
    )
    parser.add_argument(
        "--record-to", type=Path, default=None, help="With --load, also write a watchable episode."
    )
    # Every PPO hyperparameter as a flag, generated from the dataclass so the two
    # cannot drift apart. They are `None` unless given, so an unspecified one
    # keeps the reasoned default in PPOConfig rather than being restated here.
    for field in dataclasses.fields(PPOConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=type(field.default),
            default=None,
            help=f"PPOConfig.{field.name} (default {field.default})",
        )
    args = parser.parse_args(argv)

    if args.load is not None:
        return score_checkpoint(args.load, args.device, args.record_to)

    given = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(PPOConfig)
        if getattr(args, field.name) is not None
    }
    train(
        TrainConfig(
            envs=args.envs,
            steps=args.steps,
            updates=args.updates,
            frame_skip=args.frame_skip,
            max_ticks=args.max_ticks,
            eval_every=args.eval_every,
            record_every=args.record_every,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            device=args.device,
            out_dir=args.out_dir,
            resume=args.resume,
        ),
        PPOConfig(**given),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
