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

* **Validation without benchmark leakage.** Every `eval_every` updates the
  policy is scored on a fixed 32-seed validation split and appended to
  ``runs/evals.csv``. That split chooses ``policy-best.pt``; the disjoint
  canonical seeds are held back for an explicit ``--load`` score, so repeated
  training decisions cannot tune against the final 98,542 scripted yardstick.
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
  policy it trained. They carry weights, optimizer state, iteration, and the
  observation/action schema plus network architecture: ``--load`` scores one,
  while ``--resume`` continues training only when those still match the live
  environment and requested PPO configuration.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import modelcard, paths, runlog
from .benchmark import (
    CANONICAL_BASELINE_MEAN_SCORE,
    CANONICAL_FRAME_SKIP,
    CANONICAL_INFERENCE_DEVICE,
    CANONICAL_MAX_TICKS,
    CANONICAL_SEED_OFFSET,
    CANONICAL_SPLIT,
    SEEDS_PER_SPLIT,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
    canonical_baseline_comparable,
)
from .control import Control
from .env import Actions, Flags, Observations, Shaping, VecEnv
from .eval import default_seeds, evaluate, format_summary, validation_seeds
from .ppo import ObsLayout, PPOConfig, Rollout, build_policy, update

#: Chooses an action index per environment, given the batch and its action mask.
Policy_fn = Callable[[Observations, Flags], Actions]

#: Compatibility alias; the protocol and its value live together in benchmark.py.
BASELINE_MEAN_SCORE = CANONICAL_BASELINE_MEAN_SCORE


class CheckpointCompatibilityError(ValueError):
    """A checkpoint belongs to a different observation or action schema."""


@dataclass
class TrainConfig:
    """Everything worth changing, in one place."""

    #: Environments stepped in parallel. Throughput keeps climbing to a few
    #: thousand (see bindings/README.md); the ceiling is usually GPU memory for
    #: the forward pass, not the simulation.
    envs: int = 1024
    #: Agent steps collected per environment before each update. Longer rollouts
    #: carry credit across more of a wave; 256 * 1024 = 262k samples per update.
    steps: int = 256
    #: How many updates to run. On resume this is the number of additional
    #: updates; the checkpoint retains the original annealing horizon.
    updates: int = 1000
    #: Ticks per agent step. 4 is ~15 decisions a second, near a human's rate.
    frame_skip: int = 4
    #: Episode length cap, in ticks. 120k is ~33 minutes of play.
    max_ticks: int = 120_000
    #: Score the policy on the fixed validation seeds this often (0 disables).
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
    #: Adam learning rate at the end of the linear schedule. PPOConfig's
    #: learning_rate is its starting value.
    learning_rate_final: float = 1.0e-5
    #: Entropy-bonus weight at the end of the linear schedule. PPOConfig's
    #: entropy_coef is its starting value.
    entropy_coef_final: float = 2.0e-3
    #: Number of updates over which to anneal. None means the fresh run's
    #: --updates; on resume it reuses the checkpoint's original horizon.
    schedule_updates: int | None = None


@dataclass(frozen=True)
class LinearSchedule:
    """The two PPO coefficients that anneal together over update numbers."""

    start_update: int
    end_update: int
    learning_rate_start: float
    learning_rate_final: float
    entropy_coef_start: float
    entropy_coef_final: float

    @property
    def updates(self) -> int:
        return self.end_update - self.start_update + 1

    def at(self, update_number: int) -> tuple[float, float]:
        """Return (learning rate, entropy coefficient), clamped at both ends."""
        span = self.end_update - self.start_update
        progress = (
            0.0 if span <= 0 else min(1.0, max(0.0, (update_number - self.start_update) / span))
        )
        learning_rate = self.learning_rate_start + progress * (
            self.learning_rate_final - self.learning_rate_start
        )
        entropy_coef = self.entropy_coef_start + progress * (
            self.entropy_coef_final - self.entropy_coef_start
        )
        return learning_rate, entropy_coef


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#: What the reward-weight flags are called, so `Shaping.gamma` and
#: `PPOConfig.gamma` — two different discounts — do not collide.
REWARD_PREFIX = "reward-"


def _flag_type(default: object) -> Callable[[str], object]:
    """How to parse a config field given on the command line.

    `type=bool` would be a trap: `bool("False")` is `True`, so
    `--reward-enabled False` would *enable* shaping. argparse has no built-in
    for this and every project that forgets writes the same bug.
    """
    if isinstance(default, bool):
        return _boolean
    return type(default)  # type: ignore[return-value]


def _boolean(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def train(
    config: TrainConfig, ppo: PPOConfig | None = None, shaping: Shaping | None = None
) -> nn.Module:
    """Run the training loop and return the trained policy."""
    ppo = ppo or PPOConfig()
    # The reward weights are a *third* group of knobs, and the one with the
    # sharpest teeth: `waste_penalty` and `multikill_bonus` are not
    # potential-based, so unlike everything in `phi` they genuinely change what
    # the policy converges to (see `Shaping`). Passing them in rather than
    # constructing them here is what lets the console offer them at all.
    shaping = shaping or Shaping()
    device = _device(config.device)
    torch.manual_seed(config.seed)

    env = VecEnv(
        num_envs=config.envs,
        frame_skip=config.frame_skip,
        max_ticks=config.max_ticks,
        shaping=shaping,
        seed=config.seed,
    )
    layout = _layout(env)
    shape = (env.obs_size, env.action_count)

    # Validate before constructing a potentially large network or rollout. A
    # state-dict size error says only that two tensors differ; this explains that
    # the checkpoint was trained against a different game observation schema.
    resume_payload: dict[str, Any] | None = None
    if config.resume is not None:
        resume_payload = torch.load(config.resume, map_location=device, weights_only=True)
        _validate_checkpoint_compatibility(
            resume_payload,
            expected_layout=layout,
            expected_action_count=shape[1],
            path=config.resume,
        )
        _validate_resume_policy_config(resume_payload, expected=ppo, path=config.resume)

    first = 1
    if resume_payload is not None:
        first = int(resume_payload["iteration"]) + 1

    schedule = _resolve_schedule(config, ppo, resume_payload, first=first)
    policy = build_policy(ppo.architecture, layout, env.action_count, ppo.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate, eps=1e-5)
    rollout = Rollout(config.steps, config.envs, env.obs_size, env.action_count, device)

    if resume_payload is not None:
        policy.load_state_dict(resume_payload["policy"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        print(f"resumed from {config.resume} at update {first}")

    # Environment 0 carries the recordings — one watchable episode at a time.
    if config.record_every > 0:
        env.record(0)
    # ./runs in a checkout, the per-user data directory when installed. Resolved
    # once, here, so every artifact below lands in the same place and the printed
    # paths are the real ones (md/paths.py).
    out_dir = paths.runs_dir(config.out_dir)
    checkpoints = out_dir / "checkpoints"
    # Restore checkpoint-selection state and the checkpoint it names as one
    # indivisible operation. Metadata alone is insufficient when a continuation
    # writes to a different output directory: it would suppress every worse
    # candidate while leaving no policy-best.pt to ship.
    validation_protocol = _validation_protocol(config, _inference_device(device))
    best_score, best_iteration = _restore_best_validation(
        resume_payload, expected_protocol=validation_protocol
    )
    if best_iteration > 0:
        assert config.resume is not None
        if _materialize_restored_best(
            resume_path=config.resume,
            resume_payload=resume_payload,
            destination=checkpoints / "policy-best.pt",
            score=best_score,
            iteration=best_iteration,
            expected_protocol=validation_protocol,
            expected_layout=layout,
            expected_action_count=shape[1],
            expected_ppo=ppo,
        ):
            print(f"  restored validation best {best_score:,.0f} at update {best_iteration}")
        else:
            print(
                "  warning: the resumed checkpoint names an earlier validation best, "
                "but its policy-best.pt could not be verified; selecting a new best "
                "for this continuation"
            )
            best_score, best_iteration = float("-inf"), 0
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
                "auxiliary_loss",
            ]
        )

    # Checked once per update. A run starts running: a STOP left behind by the
    # last one must not kill this one before its first update.
    control = Control(out_dir)
    control.clear()
    _write_config(out_dir / "config.json", config, ppo, schedule, out_dir, shaping)
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
        f"= {config.envs * config.steps:,} samples/update | "
        f"validation {SEEDS_PER_SPLIT} seeds"
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
        learning_rate, entropy_coef = schedule.at(iteration)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        update_config = dataclasses.replace(ppo, entropy_coef=entropy_coef)

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
        stats = update(policy, optimizer, rollout, advantages, returns, update_config)

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
                f"{stats['auxiliary_loss']:.6f}",
            ]
        )
        metrics.flush()  # so a plot can follow a run that is still going
        ret = f"{mean_return:>8.2f}" if recent else "       -"
        # "shaped ret", not "return": this is the shaped, scaled, undiscounted sum
        # (md.env.Shaping) and has no fixed relationship to the game score. The
        # honest scoreboard is the eval block below and runs/evals.csv.
        print(
            f"update {iteration:>5} | shaped ret {ret} "
            f"| entropy {stats['entropy']:.3f} | value {stats['value_loss']:.3f} "
            f"| auxiliary {stats['auxiliary_loss']:.3f} "
            f"| lr {learning_rate:.2e} | ent coef {entropy_coef:.4f} "
            f"| {steps_done / elapsed / 1e3:.0f}k steps/s"
        )

        if config.record_every > 0 and iteration % config.record_every == 0:
            path = out_dir / f"update-{iteration:05d}.mdr"
            if env.save_recording(0, path, update=iteration, label=f"UPDATE {iteration}"):
                print(f"  recorded {path}")

        if config.eval_every > 0 and iteration % config.eval_every == 0:
            inference_device = _inference_device(device)
            summary = _score(
                policy,
                device,
                frame_skip=config.frame_skip,
                max_ticks=config.max_ticks,
            )
            _log_eval(
                out_dir / "evals.csv",
                iteration,
                summary,
                seed_split=VALIDATION_SPLIT,
                seed_offset=VALIDATION_SEED_OFFSET,
                seed_count=SEEDS_PER_SPLIT,
                frame_skip=config.frame_skip,
                max_ticks=config.max_ticks,
                inference_device=inference_device,
            )
            print(format_summary(summary))
            print(
                f"  validation score {summary.mean_score:,.0f} "
                f"({SEEDS_PER_SPLIT} seeds, frame skip {config.frame_skip}, "
                f"{inference_device})"
            )
            # Select on validation, not on the held-out benchmark. A peak at
            # update 800 must still survive a regression by update 1000.
            if summary.mean_score > best_score:
                best_score = summary.mean_score
                best_iteration = iteration
                best_path = checkpoints / "policy-best.pt"
                _save(
                    policy,
                    optimizer,
                    iteration,
                    shape,
                    ppo,
                    layout,
                    schedule,
                    best_path,
                    best_validation=_best_validation_record(
                        best_score, best_iteration, validation_protocol
                    ),
                )
                print(f"  new best — saved {best_path}")
            else:
                print(f"  best so far {best_score:,.0f} at update {best_iteration}")

        if config.checkpoint_every > 0 and iteration % config.checkpoint_every == 0:
            _save(
                policy,
                optimizer,
                iteration,
                shape,
                ppo,
                layout,
                schedule,
                checkpoints / f"policy-{iteration:05d}.pt",
                best_validation=_best_validation_record(
                    best_score, best_iteration, validation_protocol
                ),
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
    _save(
        policy,
        optimizer,
        iteration,
        shape,
        ppo,
        layout,
        schedule,
        checkpoints / "policy-final.pt",
        best_validation=_best_validation_record(best_score, best_iteration, validation_protocol),
    )
    metrics.close()
    control.clear()  # so the next run in this directory is not born stopped
    print(f"  final policy -> {checkpoints / 'policy-final.pt'}")
    if best_iteration > 0:
        # The one to score or ship: the final policy is often not the best one.
        print(
            f"  best policy  -> {checkpoints / 'policy-best.pt'} "
            f"({best_score:,.0f} at update {best_iteration})"
        )
    print(f"  metrics      -> {metrics_path}")
    return policy


def _write_config(
    path: Path,
    config: TrainConfig,
    ppo: PPOConfig,
    schedule: LinearSchedule,
    out_dir: Path,
    shaping: Shaping | None = None,
) -> None:
    """Record what produced this run, beside what it produced.

    Six months later the checkpoints are still there and the shell history is
    not. Written on every run, not only the ones a console starts, because the
    question "what were the settings" is asked of whichever run turned out to be
    interesting. The *resolved* output directory is recorded rather than the
    ``None`` that asked for it, so the file says where the run actually went.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = dataclasses.asdict(config) | {"out_dir": str(out_dir)}
    payload = {
        "train": settings,
        "ppo": dataclasses.asdict(ppo),
        # What the agent was *paid for*. Now that these are settable, a run whose
        # config.json omitted them would be one nobody could reproduce — and the
        # two non-potential terms genuinely change what the policy converges to,
        # so they are the last thing to leave out of the record.
        "shaping": dataclasses.asdict(shaping or Shaping()),
        "schedule": dataclasses.asdict(schedule),
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


#: Columns of ``evals.csv`` — the fields of the shared C++ ``Summary``. The
#: original nine come first and in their original order, so anything that read
#: this file before the statistics were widened still finds them; the complete
#: set (survival, damage, spend, kills-per-shot distribution) and the exact
#: evaluation protocol are appended after.
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
    "mean_ticks",
    "mean_waves_cleared",
    "mean_cities_lost",
    "mean_bases_left",
    "mean_bases_lost",
    "mean_ammo_left",
    "mean_bonus_cities",
    "mean_mirv_splits",
    "mean_shots",
    "mean_kills",
    "mean_hits",
    "mean_hit_rate",
    "shots_0kill",
    "shots_1kill",
    "shots_2kill",
    "shots_3kill",
    "shots_4plus",
    "seed_split",
    "seed_offset",
    "seed_count",
    "frame_skip",
    "max_ticks",
    "inference_device",
)


def _log_eval(
    path: Path,
    iteration: int,
    summary: Any,
    *,
    seed_split: str,
    seed_offset: int,
    seed_count: int,
    frame_skip: int,
    max_ticks: int,
    inference_device: str,
) -> None:
    """Append one scored evaluation and the protocol that produced it.

    A separate file from ``metrics.csv`` on purpose. That one carries the training
    return, which is shaped, scaled and summed undiscounted — a fine diagnostic,
    but *not* a score. These rows use the shared C++ ``summarize`` and greedy
    play, but routine rows are validation measurements rather than claims against
    the held-out scripted baseline. The appended metadata makes that distinction
    machine-checkable, including cadence and inference backend.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_eval_schema(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(EVAL_COLUMNS)
        hist = list(summary.kills_per_shot)
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
                f"{summary.mean_ticks:.1f}",
                f"{summary.mean_waves_cleared:.3f}",
                f"{summary.mean_cities_lost:.3f}",
                f"{summary.mean_bases_left:.3f}",
                f"{summary.mean_bases_lost:.3f}",
                f"{summary.mean_ammo_left:.2f}",
                f"{summary.mean_bonus_cities:.3f}",
                f"{summary.mean_mirv_splits:.3f}",
                f"{summary.mean_shots:.2f}",
                f"{summary.mean_kills:.2f}",
                f"{summary.mean_hits:.2f}",
                f"{summary.mean_hit_rate:.4f}",
                *hist,
                seed_split,
                seed_offset,
                seed_count,
                frame_skip,
                max_ticks,
                inference_device,
            ]
        )


def _migrate_eval_schema(path: Path) -> None:
    """Atomically widen an old eval CSV, preserving its rows with blank metadata."""

    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if rows and tuple(rows[0]) == EVAL_COLUMNS:
        return

    fields: list[str] = []
    preserved: list[list[str]] = []
    for cells in rows:
        if not cells:
            continue
        try:
            float(cells[0])
        except ValueError:
            fields = cells
            continue
        if fields:
            values = dict(zip(fields, cells, strict=False))
            preserved.append([values.get(column, "") for column in EVAL_COLUMNS])

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(EVAL_COLUMNS)
            writer.writerows(preserved)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validation_protocol(config: TrainConfig, inference_device: str) -> dict[str, Any]:
    """The exact development protocol that is allowed to select policy-best."""

    return {
        "seed_split": VALIDATION_SPLIT,
        "seed_offset": VALIDATION_SEED_OFFSET,
        "seed_count": SEEDS_PER_SPLIT,
        "frame_skip": config.frame_skip,
        "max_ticks": config.max_ticks,
        "inference_device": inference_device,
    }


def _best_validation_record(
    score: float,
    iteration: int,
    protocol: dict[str, Any],
) -> dict[str, Any] | None:
    """Serializable selection evidence, or none before the first validation."""

    if iteration <= 0 or not np.isfinite(score):
        return None
    return {"score": float(score), "iteration": int(iteration), **protocol}


def _restore_best_validation(
    payload: dict[str, Any] | None,
    *,
    expected_protocol: dict[str, Any],
) -> tuple[float, int]:
    """Restore checkpoint-selection state only when its protocol still matches."""

    if payload is None:
        return float("-inf"), 0
    stored = payload.get("best_validation")
    if not isinstance(stored, dict):
        return float("-inf"), 0
    if any(stored.get(key) != value for key, value in expected_protocol.items()):
        return float("-inf"), 0
    try:
        score = float(stored["score"])
        iteration = int(stored["iteration"])
        checkpoint_iteration = int(payload["iteration"])
    except (KeyError, TypeError, ValueError):
        return float("-inf"), 0
    if not np.isfinite(score) or iteration <= 0 or iteration > checkpoint_iteration:
        return float("-inf"), 0
    return score, iteration


def _materialize_restored_best(
    *,
    resume_path: Path,
    resume_payload: dict[str, Any] | None,
    destination: Path,
    score: float,
    iteration: int,
    expected_protocol: dict[str, Any],
    expected_layout: ObsLayout,
    expected_action_count: int,
    expected_ppo: PPOConfig,
) -> bool:
    """Put the verified pre-resume best checkpoint in the continuation run.

    The usual candidate is ``policy-best.pt`` beside the resumed checkpoint.
    Resuming the best update itself is also sufficient, even if that sibling was
    moved or deleted. A temporary file plus ``os.replace`` keeps dashboards and
    evaluators from observing a half-copied checkpoint.
    """

    if resume_payload is None or iteration <= 0 or not np.isfinite(score):
        return False

    candidates: list[tuple[Path, dict[str, Any] | None]] = [
        (resume_path.parent / "policy-best.pt", None),
    ]
    try:
        resumed_iteration = int(resume_payload["iteration"])
    except (KeyError, TypeError, ValueError):
        return False
    if resumed_iteration == iteration:
        candidates.append((resume_path, resume_payload))

    seen: set[Path] = set()
    for candidate, loaded in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        try:
            payload = (
                loaded
                if loaded is not None
                else torch.load(candidate, map_location="cpu", weights_only=True)
            )
            if not isinstance(payload, dict) or int(payload["iteration"]) != iteration:
                continue
            restored_score, restored_iteration = _restore_best_validation(
                payload, expected_protocol=expected_protocol
            )
            if restored_score != score or restored_iteration != iteration:
                continue
            _validate_checkpoint_compatibility(
                payload,
                expected_layout=expected_layout,
                expected_action_count=expected_action_count,
                path=candidate,
            )
            _validate_resume_policy_config(payload, expected=expected_ppo, path=candidate)
        except (
            CheckpointCompatibilityError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue

        if resolved != destination.resolve():
            _atomic_copy(candidate, destination)
        return True
    return False


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy a checkpoint without ever exposing a partial destination file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _save(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    shape: tuple[int, int],
    ppo: PPOConfig,
    layout: ObsLayout,
    schedule: LinearSchedule,
    path: Path,
    *,
    best_validation: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint that is enough to *resume* from, not just to score.

    That means the optimizer too: Adam carries momentum estimates, and restarting
    without them makes the first few updates after a resume behave nothing like
    the ones before it. Observation/action layout and network dimensions ride
    along so a loader can verify the live environment and reconstruct the exact
    policy before accepting its weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
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
        # The update number alone cannot reconstruct an annealing schedule:
        # its original endpoints are needed so resume reaches the exact next
        # coefficient rather than restarting from the initial value.
        "schedule": dataclasses.asdict(schedule),
    }
    if best_validation is not None:
        payload["best_validation"] = best_validation
    torch.save(payload, path)


def _layout(env: VecEnv) -> ObsLayout:
    """The observation's entity layout, for a policy that slices it."""
    spec = env.spec
    return ObsLayout(
        threats=spec.threats,
        interceptors=spec.interceptors,
        blasts=spec.blasts,
        obs_size=env.obs_size,
    )


def _validate_checkpoint_compatibility(
    payload: dict[str, Any],
    *,
    expected_layout: ObsLayout,
    expected_action_count: int,
    path: Path,
) -> None:
    """Reject weights trained against a different environment schema.

    Both a flat MLP and an entity policy can load successfully before failing
    much later: an MLP at its first matrix multiply, an entity policy by silently
    slicing the wrong fields when block widths happen to leave the same total
    size. Check the semantic metadata before either network is constructed.
    """

    differences: list[str] = []
    for key, label, expected in (
        ("obs_size", "observation size", expected_layout.obs_size),
        ("action_count", "action count", expected_action_count),
    ):
        try:
            stored_value = int(payload[key])
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(
                f"checkpoint {path} has no valid {label} metadata; "
                "its environment compatibility cannot be established"
            ) from error
        if stored_value != expected:
            differences.append(f"{label} is {stored_value}, current environment is {expected}")

    stored_layout = payload.get("layout")
    architecture = str(payload.get("architecture", "mlp"))
    if stored_layout is None:
        # Checkpoints predating the entity policy have no layout. Their flat MLP
        # is still safe when its complete observation and action shapes match.
        if architecture == "entity":
            differences.append("entity layout metadata is missing")
    elif not isinstance(stored_layout, dict):
        differences.append("entity layout metadata is not a mapping")
    else:
        current_layout = dataclasses.asdict(expected_layout)
        for name, expected in current_layout.items():
            if name not in stored_layout:
                differences.append(f"layout.{name} is missing")
                continue
            stored_value = stored_layout[name]
            if stored_value != expected:
                differences.append(
                    f"layout.{name} is {stored_value}, current environment is {expected}"
                )
        unexpected = sorted(set(stored_layout) - set(current_layout))
        if unexpected:
            differences.append(f"unrecognised layout fields: {', '.join(unexpected)}")

    if differences:
        detail = "; ".join(differences)
        raise CheckpointCompatibilityError(
            f"checkpoint {path} is incompatible with the current environment schema: {detail}. "
            "It cannot be scored or resumed safely; use a checkpoint trained with "
            "the current schema"
        )


def _current_environment_schema() -> tuple[ObsLayout, int]:
    """Read the default schema from the live native environment."""
    env = VecEnv(num_envs=0, threads=1)
    return _layout(env), env.action_count


def _validate_resume_policy_config(
    payload: dict[str, Any], *, expected: PPOConfig, path: Path
) -> None:
    """Ensure resume reconstructs the same network the checkpoint contains."""
    stored_architecture = str(payload.get("architecture", "mlp"))
    try:
        stored_hidden = int(payload["hidden"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"checkpoint {path} has no valid hidden-size metadata; "
            "its policy architecture cannot be resumed safely"
        ) from error

    differences: list[str] = []
    if stored_architecture != expected.architecture:
        differences.append(
            f"architecture is {stored_architecture!r}, "
            f"requested PPOConfig is {expected.architecture!r}"
        )
    if stored_hidden != expected.hidden:
        differences.append(
            f"hidden size is {stored_hidden}, requested PPOConfig is {expected.hidden}"
        )

    if differences:
        raise CheckpointCompatibilityError(
            f"checkpoint {path} cannot be resumed with the requested PPOConfig: "
            f"{'; '.join(differences)}. Resume with matching --architecture and --hidden, "
            "or start a new run"
        )


def _resolve_schedule(
    config: TrainConfig,
    ppo: PPOConfig,
    payload: dict[str, Any] | None,
    *,
    first: int,
) -> LinearSchedule:
    """Create a fresh schedule or restore the exact one saved in a checkpoint."""

    if payload is not None and "schedule" in payload:
        stored = payload["schedule"]
        if not isinstance(stored, dict):
            raise CheckpointCompatibilityError("checkpoint schedule metadata is not a mapping")
        try:
            schedule = LinearSchedule(
                start_update=int(stored["start_update"]),
                end_update=int(stored["end_update"]),
                learning_rate_start=float(stored["learning_rate_start"]),
                learning_rate_final=float(stored["learning_rate_final"]),
                entropy_coef_start=float(stored["entropy_coef_start"]),
                entropy_coef_final=float(stored["entropy_coef_final"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointCompatibilityError(
                "checkpoint has invalid annealing-schedule metadata"
            ) from error
        checkpoint_iteration = int(payload["iteration"])
        if (
            schedule.start_update > checkpoint_iteration
            or schedule.end_update < schedule.start_update
            or schedule.learning_rate_start <= 0.0
            or not 0.0 < schedule.learning_rate_final <= schedule.learning_rate_start
            or schedule.entropy_coef_start < 0.0
            or not 0.0 <= schedule.entropy_coef_final <= schedule.entropy_coef_start
        ):
            raise CheckpointCompatibilityError(
                "checkpoint has invalid annealing-schedule endpoints"
            )

        differences: list[str] = []
        requested_updates = config.schedule_updates
        for label, stored_value, requested_value in (
            ("starting learning rate", schedule.learning_rate_start, ppo.learning_rate),
            ("final learning rate", schedule.learning_rate_final, config.learning_rate_final),
            ("starting entropy coefficient", schedule.entropy_coef_start, ppo.entropy_coef),
            ("final entropy coefficient", schedule.entropy_coef_final, config.entropy_coef_final),
        ):
            if stored_value != requested_value:
                differences.append(
                    f"{label} is {stored_value:g}, requested configuration is {requested_value:g}"
                )
        if requested_updates is not None and requested_updates != schedule.updates:
            differences.append(
                f"schedule length is {schedule.updates}, "
                f"requested configuration is {requested_updates}"
            )
        if differences:
            raise CheckpointCompatibilityError(
                "checkpoint cannot be resumed with the requested annealing schedule: "
                f"{'; '.join(differences)}. Resume with matching schedule settings, "
                "or start a new run"
            )
        return schedule

    # A checkpoint from before schedules existed has no endpoints to continue.
    # Start a new, explicit continuation from its restored Adam learning rate.
    learning_rate_start = ppo.learning_rate
    if payload is not None:
        groups = payload.get("optimizer", {}).get("param_groups", [])
        if groups:
            learning_rate_start = float(groups[0].get("lr", learning_rate_start))

    schedule_updates = config.schedule_updates
    if schedule_updates is None:
        schedule_updates = max(1, config.updates)
    if schedule_updates < 1:
        raise ValueError("schedule_updates must be at least 1")
    if not 0.0 < config.learning_rate_final <= learning_rate_start:
        raise ValueError(
            "learning_rate_final must be positive and no greater than the starting rate"
        )
    if not 0.0 <= config.entropy_coef_final <= ppo.entropy_coef:
        raise ValueError(
            "entropy_coef_final must be non-negative and no greater than the starting value"
        )
    return LinearSchedule(
        start_update=first,
        end_update=first + schedule_updates - 1,
        learning_rate_start=learning_rate_start,
        learning_rate_final=config.learning_rate_final,
        entropy_coef_start=ppo.entropy_coef,
        entropy_coef_final=config.entropy_coef_final,
    )


def load_policy(path: Path, device: torch.device | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a policy from its stored architecture. Return it and the raw payload."""
    device = device or torch.device("cpu")
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    expected_layout, expected_action_count = _current_environment_schema()
    _validate_checkpoint_compatibility(
        payload,
        expected_layout=expected_layout,
        expected_action_count=expected_action_count,
        path=path,
    )
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


def _inference_device(device: torch.device) -> str:
    """Stable CSV label for the backend used by the policy forward pass."""

    return str(device)


def _score(
    policy: nn.Module,
    device: torch.device,
    *,
    frame_skip: int,
    max_ticks: int,
) -> object:
    """Score the current policy on the fixed validation split, greedily.

    The training cadence is forwarded exactly. Otherwise a run deliberately
    trained at a non-default frame skip would silently select checkpoints under
    a different control budget.
    """

    def act(obs: Observations, mask: Flags) -> Actions:
        with torch.no_grad():
            logits, _ = policy(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))
            return logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    return evaluate(
        act,
        seeds=validation_seeds(),
        frame_skip=frame_skip,
        max_ticks=max_ticks,
    )


def score_checkpoint(path: Path, device_name: str | None = None, record: Path | None = None) -> int:
    """Run the final held-out benchmark for a validation-selected checkpoint.

    Use the validation split to compare and select checkpoints. This command is
    the one-time final score after that choice; repeatedly choosing based on its
    output would turn the held-out split into another validation set. Pass
    `record` to also drop a watchable episode of that policy playing.
    """
    # Final scoring defaults to CPU even on CUDA hosts. Tiny backend-dependent
    # logit differences can alter an argmax and therefore an entire episode.
    device = (
        torch.device(CANONICAL_INFERENCE_DEVICE)
        if device_name is None
        else torch.device(device_name)
    )
    inference_device = _inference_device(device)
    policy, payload = load_policy(path, device)
    print(f"loaded {path} (update {payload['iteration']})")
    print(
        f"held-out {CANONICAL_SPLIT} benchmark | {SEEDS_PER_SPLIT} seeds | "
        f"seed offset {CANONICAL_SEED_OFFSET} | "
        f"frame skip {CANONICAL_FRAME_SKIP} | max ticks {CANONICAL_MAX_TICKS:,} | "
        f"inference {inference_device}"
    )

    if record is not None:
        # One env, one seed, recorded start to finish.
        seeds = default_seeds()
        env = VecEnv(
            num_envs=1,
            threads=1,
            frame_skip=CANONICAL_FRAME_SKIP,
            max_ticks=CANONICAL_MAX_TICKS,
            shaping=None,
            seed=seeds[0],
        )
        env.record(0)
        act = greedy_policy(policy, device)
        for _ in range(200_000):
            _, _, terminated, truncated, _ = env.step(act(env.observations, env.action_masks()))
            if terminated[0] or truncated[0]:
                break
        if env.save_recording(0, record, update=int(payload["iteration"]), label=path.stem.upper()):
            print(f"recorded {record}")

    summary = evaluate(
        greedy_policy(policy, device),
        seeds=default_seeds(),
        frame_skip=CANONICAL_FRAME_SKIP,
        max_ticks=CANONICAL_MAX_TICKS,
    )
    print(format_summary(summary))
    delta = summary.mean_score - BASELINE_MEAN_SCORE
    if canonical_baseline_comparable(
        seed_split=CANONICAL_SPLIT,
        seed_offset=CANONICAL_SEED_OFFSET,
        seed_count=SEEDS_PER_SPLIT,
        frame_skip=CANONICAL_FRAME_SKIP,
        max_ticks=CANONICAL_MAX_TICKS,
        inference_device=inference_device,
    ):
        print(f"  {abs(delta):,.0f} {'ahead of' if delta > 0 else 'behind'} the scripted baseline")
    else:
        print(
            "  scripted-baseline comparison disabled: "
            f"published inference backend is {CANONICAL_INFERENCE_DEVICE}"
        )
    if path.parent.name == "checkpoints":
        eval_path = path.parent.parent / "evals.csv"
        _log_eval(
            eval_path,
            int(payload["iteration"]),
            summary,
            seed_split=CANONICAL_SPLIT,
            seed_offset=CANONICAL_SEED_OFFSET,
            seed_count=SEEDS_PER_SPLIT,
            frame_skip=CANONICAL_FRAME_SKIP,
            max_ticks=CANONICAL_MAX_TICKS,
            inference_device=inference_device,
        )
        print(f"  benchmark row -> {eval_path}")
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
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        help=(
            "Training device (default: auto); with --load, the held-out benchmark "
            f"defaults to {CANONICAL_INFERENCE_DEVICE}."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where the run writes (default: ./runs in a checkout, else the user data dir).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Continue training from a checkpoint."
    )
    parser.add_argument("--learning-rate-final", type=float, default=defaults.learning_rate_final)
    parser.add_argument("--entropy-coef-final", type=float, default=defaults.entropy_coef_final)
    parser.add_argument("--schedule-updates", type=int, default=defaults.schedule_updates)
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Run the final held-out benchmark for a selected policy and exit.",
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
            type=_flag_type(field.default),
            default=None,
            help=f"PPOConfig.{field.name} (default {field.default})",
        )
    # And every reward weight, the same way and for the same reason. These were
    # the one group with no flag at all: changing what the agent is *paid for*
    # meant editing `Shaping` and rebuilding, which put the most consequential
    # knobs in the project out of reach of anyone not editing the source.
    #
    # Prefixed, because `Shaping.gamma` and `PPOConfig.gamma` are two different
    # discounts and a flat namespace makes them collide — and because `--reward-`
    # groups them for a reader, which a flat `--city-weight` beside `--envs`
    # would not.
    for field in dataclasses.fields(Shaping):
        parser.add_argument(
            f"--{REWARD_PREFIX}{field.name.replace('_', '-')}",
            dest=f"reward_{field.name}",
            type=_flag_type(field.default),
            default=None,
            help=f"Shaping.{field.name} (default {field.default})",
        )
    args = parser.parse_args(argv)

    if args.load is not None:
        return score_checkpoint(args.load, args.device, args.record_to)

    given = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(PPOConfig)
        if getattr(args, field.name) is not None
    }
    weights = {
        field.name: getattr(args, f"reward_{field.name}")
        for field in dataclasses.fields(Shaping)
        if getattr(args, f"reward_{field.name}") is not None
    }
    config = TrainConfig(
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
        learning_rate_final=args.learning_rate_final,
        entropy_coef_final=args.entropy_coef_final,
        schedule_updates=args.schedule_updates,
    )
    # A copy of everything below goes to runs/train.log as well as the terminal.
    # That is what lets the console show a log pane for a run it did not start
    # — the case the whole out-of-process design exists for (md.runlog).
    with runlog.teed(paths.runs_dir(config.out_dir)):
        train(config, PPOConfig(**given), Shaping(**weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
