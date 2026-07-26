# Session handover — run statistics, PPO measures, player-model calibration

Written 2026-07-26 at a **green, uncommitted** checkpoint (109/109 C++ tests +
Python `test_env`/`test_ppo` pass). Read alongside
`2026-07-26-continue-here.md`. **Two agents share this branch** — stage explicit
paths, never a directory, and coordinate before reverting anything.

## State of the tree (all uncommitted, all green)

Three separable bodies of work, plus one deliberate mechanics change:

1. **Task 11 — full run statistics.** `StepResult` gains a kills-per-shot
   histogram (`config.hpp: kills_per_shot_bins = 5`); `EpisodeResult`/`Summary`
   gain waves_cleared, ticks, cities/bases lost, `ammo_left`, mirv_splits,
   shots/kills/hits, `wasted()`/`hits()`/`hit_rate()`, and the histogram.
   Counting is shared (`agent::tally_events` + `bin_active_blasts`) between the
   scripted `run_episode` and the Python `VecEnv`. Surfaced through
   `bindings/module.cpp` + `_md_native.pyi`, into `eval.format_summary`,
   `poe eval`, and `train.py`'s widened `evals.csv` (new columns appended, so
   name-keyed readers are safe). **Files:** core/{config.hpp,sim.hpp,sim.cpp},
   agent/{eval.hpp,eval.cpp,eval_main.cpp}, bindings/{module.cpp,vec_env.cpp},
   python/md/{_md_native.pyi,eval.py,train.py}, tests. **Still owes:** docs
   (`docs/TRAINING.md`, `docs/API.md` mention none of it) and its e2e claim.

2. **PPO stability measures** (unrelated reason to change → its own commit):
   `ppo.py` — clipped value loss (`value_clip=0.2`) + `entropy_coef` 0.01→0.02;
   `train.py` — keep the best-by-eval checkpoint (`policy-best.pt`) and honest
   output (`shaped ret`, eval-score-vs-baseline line). Diagnosed from a run that
   peaked ~80k then regressed while shaped return kept climbing.

3. **`docs/NVIDIA.md`** (untracked) — the Debian/CUDA GPU-training guide. Its own
   commit.

4. **Player-model calibration** (the mechanics change, its own commit):
   - `fire_interval` 0.15 → **0.33** (~3 shots/s), per the user.
   - **Decision-rate limit in the core:** `Config::decision_interval` (ticks per
     decision); `Sim::step` latches the action once per that many ticks and
     `move_crosshair`/`try_fire` use the latched action (crosshair still steers
     every tick, only the *decision* is paced). `reset()` clears it.
   - `agent::run_episode`/`evaluate` no longer take a `frame_skip` param; the
     cadence is the sim's `Config::decision_interval`. `md_agent_eval --frame-skip
     N` sets `config.decision_interval` (1 = 60 Hz, 4 = the policy's ~15 Hz).
   - **Golden checksum regenerated** to `0x643f44055f179ba` (test_smoke_e2e.cpp).

## The key finding this session (verified in code)

The scripted baseline was **unfairly** measured at 60 Hz while the policy runs at
15 Hz. Verified the heuristic is otherwise fair (action-only through `Sim::step`;
reads only `threat.pos`/`velocity`, never `target_index`). The edge is **reaction
latency, not firepower** (both fire ~1.2/s, far under the cap). Numbers at the
*old* mechanics (fire 0.15): scripted **113,834** @ 60 Hz vs **91,094** @ 15 Hz
(−20%). Best learned policy **80,298** (@ 15 Hz) → ~12% behind the fair 15 Hz
baseline, not 29%. See memory `baseline-decision-rate-fairness`.

## Open decisions / TODO (in priority order)

1. **`decision_interval` default = 1 (per-tick) right now.** The user chose "15 Hz
   uniform in the core, incl. human." I set the *default* back to 1 to reach green
   fast: default 4 broke 12 mechanic unit tests (they step per-tick — fix by
   adding `decision_interval = 1` to `test_sim.cpp`'s `unpaced()` helper and to
   the direct-`Config` mechanic tests in test_sim/test_intercept, the same way
   they set `fire_interval = 0`), and needs `VecEnv` to set
   `config_.decision_interval = frame_skip_` (I reverted that wiring because it
   broke vec_sim tests #103/#107 — the hand-stepped comparison sim must match) and
   the app to set it for the human. **Decide:** default 4 + do all that, or keep
   default 1 and have drivers opt in.
2. **Redo baselines** (the user asked): `poe eval` at the new mechanics —
   `md_agent_eval` (60 Hz) and `--frame-skip 4` (15 Hz) — and update
   `docs/ROADMAP.md` (M4 = 113,834 is now stale: fire is 3/s). The learned 80k
   model is also invalid under the new mechanics; retrain when ready.
3. **Commit** as separate commits (handoff rule): (a) statistics, (b) PPO change,
   (c) NVIDIA guide, (d) player-model calibration + golden. `git add -p` is
   unavailable here, and `train.py`/`config.hpp` mix reasons, so perfect
   per-reason splitting needs care.
4. **Docs:** DESIGN §5 still says "agent acts every tick (K=1)"; update it to the
   core `decision_interval`. Task 11 docs (TRAINING.md/API.md).
5. **Config-file feature** (fire interval, decision frequency, tick rate loadable
   from an optional file, defaults when absent, none written by default) —
   **deferred to a maybe-later list** by the user.

## Environment (this box, RTX 5090)

`.venv` has torch 2.13.0+cu130, numpy, nanobind, PySide6, psutil — GPU training
works (`poe train` auto-selects cuda; ~43× CPU; sweet spot 4096–8192 envs; TF32
+10%). TF32 wrapper lived in `/tmp` scratchpad (wiped on reboot; recreate: set
`torch.backends.cuda.matmul.allow_tf32=True` + `set_float32_matmul_precision`
before `md.train.main`). **The box hard-locked once** on 15-min idle screen-off
(NVIDIA 610 + Wayland + KDE DDC/CI) — see memory `system-hardlock-idle-dpms`;
disable idle screen-off before unattended runs. Human plays ~8k–10k (memory
`human-play-baseline`).

## Continuation — fair baseline and fresh training run (2026-07-26 19:00)

The user explicitly asked to stop here until another agent has made more fairness
fixes. Do not resume or start another run without that signal.

- Uniform 15 Hz was completed: `Config::decision_interval = 4` by default and
  `VecEnv` copies its `frame_skip` into the core decision interval. Mechanic tests
  that intentionally change actions every tick opt into interval 1.
- Debug and Release were both **109/109 green** immediately after that wiring.
  Their shared trajectory golden is `0x01fe4f2c28b1e200`.
- New canonical scripted baseline, 32 fixed seeds, 120,000 tick cap:
  **98,170.15625** mean, range **81,930–115,065**, mean wave 15.81,
  1.08 kills/shot. This was the interim offset-0 threshold replacing 113,834.
- That number was the offset-0 development block and is now **superseded** by
  the fairness agent's split fix. The authoritative held-out canonical block is
  offset 32: **98,542.34375**, range **83,525–108,920**. Routine checkpoint
  selection uses offset 0 as validation; only the final selected policy sees the
  canonical block. `python/md/benchmark.py` is the source of truth.
- A genuinely from-zero MLP PPO run was started (no `--resume`) with 4,096 envs,
  128 steps/update, CUDA+TF32, default PPO hyperparameters, evaluations every 25,
  intended for 1,500 updates:
  `runs/player-model-v2-from-zero/`.
- The user stopped it gracefully after update **1,437**. The final checkpoint and
  metrics were flushed. The stop marker was cleared by the trainer.
- Best checkpoint: update **1,300**, historical offset-0 validation mean
  **73,402.8**, range 16,180–88,045, mean wave 14.00, 0.79 kills/shot. It is at
  `runs/player-model-v2-from-zero/checkpoints/policy-best.pt`.
  `policy-final.pt` is the stopped update-1,437 policy and is not the model to
  prefer.
- Operational success condition discussed with the user: beat the canonical
  baseline, then beat it on 256+ unseen paired seeds with the 95% confidence
  interval of the score difference above zero, and stop after 10 evaluations /
  250 updates without a meaningful unseen-seed improvement. Global mathematical
  optimality cannot be established.

The working tree changed substantially during the run because the other agent was
active (GPU telemetry, observation/model-card/benchmark/docs work). Preserve those
changes; do not stage directories or revert them.

## Plateau-breaking training work (2026-07-26, after the stopped run)

The user assigned the remaining improvements while a separate agent continues
fairness fixes. The implementation is ready, but **no new training run was
started**; wait until the fairness work and canonical baseline are frozen.

- `--architecture entity` is now a relational actor: every threat uses shared
  weights and cross-attends separately to live interceptors and blasts. Its
  critic is a disjoint flat network, so value-loss gradients cannot overwrite
  actor features. The legacy MLP remains available for old checkpoints and
  ablations.
- The relational actor has a training-only three-output auxiliary head. Targets
  in `md.auxiliary` derive time-to-impact, current coverage, and local threat
  density from the raw observation only. PPO applies the masked loss with
  `auxiliary_coef=0.1`; inference receives no extra or privileged features.
- Rollouts default to 256 steps. Learning rate and entropy anneal linearly from
  the PPO starting values to `1e-5` and `0.002`, with the resolved schedule
  stored in checkpoints and continued by absolute update number on resume.
- `md-multiseed` starts genuinely fresh seed runs in separate directories,
  rejects resume/load overrides, and selects `policy-best.pt` from matching
  validation protocols only. It deliberately leaves the canonical split unseen
  until one final checkpoint is selected. See `docs/MULTI_SEED.md`.

Recommended fresh experiment after the fairness signal:

```bash
md-multiseed \
  --out-dir runs/relational-3seed \
  --num-seeds 3 \
  --seed-start 1000 \
  -- \
  --architecture entity \
  --updates 1500 \
  --envs 4096 \
  --eval-every 25
```

Use one isolated smoke run first to establish safe GPU memory and throughput for
the attention policy; do not assume the old MLP's 4,096-environment batch has the
same memory footprint.

Verification at handoff: all non-e2e Python tests pass, Ruff lint/format and
Pyright (using the torch-enabled `.venv` interpreter) pass, and Release is
115/115 green. The Debug build's Catch2 discovery cannot run under this session's
ptrace because LeakSanitizer rejects ptrace; this is an environment limitation,
not a discovered test failure.
