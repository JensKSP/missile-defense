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
