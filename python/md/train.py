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
  decides whether M6 succeeded.
* **Watchable episodes.** Every `record_every` updates one episode is written to
  ``runs/`` as ``update-<n>.mdr``. Open it from the app's REPLAYS menu and watch
  what the policy is actually doing; a reward curve will not tell you that it has
  learned to ignore MIRVs.
* **Checkpoints.** Written to ``runs/checkpoints`` — every ``checkpoint_every``
  updates plus a ``policy-final.pt`` at the end, so a short run still leaves the
  policy it trained. Weights only, not optimizer state: these are for scoring or
  watching a past policy, **not** for resuming a run. There is no resume yet.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .env import Actions, Flags, Observations, Shaping, VecEnv
from .eval import evaluate, format_summary
from .ppo import Policy, PPOConfig, Rollout, update

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
    out_dir: Path = Path("runs")


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config: TrainConfig, ppo: PPOConfig | None = None) -> Policy:
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
    policy = Policy(env.obs_size, env.action_count, ppo.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate, eps=1e-5)
    rollout = Rollout(config.steps, config.envs, env.obs_size, env.action_count, device)

    # Environment 0 carries the recordings — one watchable episode at a time.
    if config.record_every > 0:
        env.record(0)
    checkpoints = config.out_dir / "checkpoints"

    print(
        f"training on {device} | {config.envs} envs x {config.steps} steps "
        f"= {config.envs * config.steps:,} samples/update | baseline {BASELINE_MEAN_SCORE:,.0f}"
    )

    # Tracks the return of episodes as they finish, for a readable progress line.
    episode_returns: list[float] = []
    running = np.zeros(config.envs, dtype=np.float64)
    started = time.perf_counter()

    for iteration in range(1, config.updates + 1):
        for step in range(config.steps):
            obs = torch.from_numpy(env.observations).to(device)
            mask = torch.from_numpy(env.action_masks()).to(device)
            action, log_prob, value = policy.act(obs, mask)

            actions: Actions = action.cpu().numpy().astype(np.int32)
            _, reward, terminated, truncated, _ = env.step(actions)
            done: Flags = terminated | truncated

            rollout.obs[step] = obs
            rollout.masks[step] = mask
            rollout.actions[step] = action
            rollout.log_probs[step] = log_prob
            rollout.values[step] = value
            rollout.rewards[step] = torch.from_numpy(reward).to(device)
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
        ret = f"{np.mean(recent):>8.2f}" if recent else "       -"
        print(
            f"update {iteration:>5} | return {ret} "
            f"| entropy {stats['entropy']:.3f} | value {stats['value_loss']:.3f} "
            f"| {steps_done / elapsed / 1e3:.0f}k steps/s"
        )

        if config.record_every > 0 and iteration % config.record_every == 0:
            path = config.out_dir / f"update-{iteration:05d}.mdr"
            if env.save_recording(0, path, update=iteration, label=f"UPDATE {iteration}"):
                print(f"  recorded {path}")

        if config.eval_every > 0 and iteration % config.eval_every == 0:
            summary = _score(policy, device)
            print(format_summary(summary))
            delta = summary.mean_score - BASELINE_MEAN_SCORE
            verdict = "ahead of" if delta > 0 else "behind"
            print(f"  {abs(delta):,.0f} {verdict} the scripted baseline")

        if config.checkpoint_every > 0 and iteration % config.checkpoint_every == 0:
            _save(policy, checkpoints / f"policy-{iteration:05d}.pt")

    # Always checkpoint the finished policy. Without this a run whose length is
    # not a multiple of checkpoint_every throws away the thing it just trained.
    _save(policy, checkpoints / "policy-final.pt")
    print(f"  final policy -> {checkpoints / 'policy-final.pt'}")
    return policy


def _save(policy: Policy, path: Path) -> None:
    """Write the weights. Only the policy — not the optimizer — so these are for
    evaluating or watching a past policy, not for resuming a run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)


def _score(policy: Policy, device: torch.device) -> object:
    """Score the current policy on the canonical seeds, greedily.

    Greedy rather than sampled: the baseline is deterministic, so comparing a
    sampled policy against it would be measuring two different things.
    """

    def act(obs: Observations, mask: Flags) -> Actions:
        with torch.no_grad():
            logits, _ = policy(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))
            return logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    return evaluate(act)


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
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    args = parser.parse_args(argv)

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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
