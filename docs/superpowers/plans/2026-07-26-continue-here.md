# Continue here — implementation handoff

A prompt for the next session. Everything below is state a fresh agent cannot
derive from the code, plus the traps this project has already sprung.

---

## The prompt

> You are continuing an implementation program on the **missile-defense** project
> (C++23 / Qt 6 / Vulkan game + PyTorch PPO trainer + PySide6 training console).
>
> **Read these two first, in this order:**
> 1. `docs/superpowers/plans/2026-07-26-continue-here.md` — this file.
> 2. `docs/superpowers/plans/2026-07-26-ai-training-user-journey.md` — the plan,
>    and the source of truth. In particular the **End-to-end tests** section,
>    whose rules apply to every task, and the **E2E:** claim under each task.
>
> **Where things stand.** Branch `task-5-managed-training-runtime`. Tasks **5**
> and **11** are complete; **Task 4 is half done** and is the thing to finish
> first. The application e2e layer exists (`python/tests/e2e/`, 11 training/console
> tests plus the game ones that skip without xvfb).
>
> **Then the plan order**, which is deliberate and not numeric: finish **Task 4**,
> then **1 → 2 → 3** (the policy chain), then **6 → 7 → 8 → 9**, then **12**, and
> **10** last.
>
> **How to work.** Test-first, always: write the failing test, watch it fail,
> implement, watch it pass. Each task ends with green focused tests, its e2e claim
> satisfied, `poe check` clean, documentation updated, and one commit. Tick the
> plan's checkboxes as you go.
>
> **Ask before assuming** on anything that changes what gets built. Do not
> self-declare a milestone done — a milestone passes only when the human confirms
> it by using it.

---

## Read this before you touch anything: you are not alone in this tree

**Other agents commit to this branch at the same time you do.** This is the single
most expensive fact in this document and it cost real damage on 2026-07-26.

* **Never `git add <directory>`.** Stage explicit file paths, and read
  `git diff --cached --name-only` before every commit. A `git add core/ agent/
  bindings/` swept another agent's in-flight player-model change into an unrelated
  statistics commit, under a message that never mentioned it, and broke the golden
  determinism test.
* **Never restore files by `cp` from a backup.** It silently reverts whatever they
  wrote in the meantime. If you must park work to verify a commit in isolation,
  park it and put it back within one tool call, or leave it alone entirely.
* **`git status` and `git log` go stale inside a single turn.** Re-run them
  immediately before committing.
* **Do not revert their changes.** Standing instruction from the human. If their
  work lands in your commit, say so and let its owner finish it.

## The one thing that is red right now

`ctest -R "Trajectory checksum"` **fails**, and it is not yours to fix.

Another agent is mid-way through moving the agent reaction-rate throttle out of
the drivers and into the simulation: `Config::decision_interval` (declared, and as
of this writing being wired into `Sim::step`), plus `fire_interval` 0.15 → 0.33.
That legitimately changes every trajectory, so the golden checksum in
`core/tests/e2e/test_smoke_e2e.cpp` has to be regenerated — and the **113,834
scripted baseline quoted all over the docs will move with it**. Their uncommitted
work was in `agent/`, `core/`, `bindings/` and `core/tests/e2e/` when this session
ended; they had started on the golden.

It supersedes the `--frame-skip` flag added in `27f7e5c` — that commit's
per-driver `frame_skip` parameter is being replaced by the Config field, which is
the better design (the player model belongs in `Sim::step`, where the crosshair
and trigger limits already live). Do not re-add it.

**Check whether this is still red before assuming anything else is broken.**

## What landed this session

| Commit | What |
|---|---|
| `27f7e5c` | `--frame-skip` on the evaluator — **being superseded**, see above |
| `4509898` | **Task 11** — the full per-run statistics + kills-per-shot histogram |
| `61f9a8c` | `policy-best.pt`: PPO peaks then regresses, so keep the best separately |
| `c1a452e` | PPO clipped value loss, `entropy_coef` 0.01 → 0.02 (hyperparameters, unproven) |
| `299c237` | `docs/NVIDIA.md` — the Debian GPU recipe, ~43× measured |
| `6a6619c` | **Task 4, first half** — the Debian three-way split + console discovery |

## Task 4 — finish this first

Done: `debian/control` (three binaries), `debian/*.install`, `debian/rules`,
`MD_INSTALL_PYTHON_PACKAGE` in `CMakeLists.txt`, `packaging/launcher.in`, the
console `.desktop`, `runner.console_executable()` / `console_command()`, and
contract tests in `test_packaging.py` + `test_ui_runner.py`. The staged tree was
verified by hand: `cmake --install` produces `usr/bin/md-console`, the package
under `dist-packages/md`, and the launcher starts and exits 0 offscreen.

Still open:

- [ ] **The TRAIN AI menu entry** (plan Step 4). `app/game_window.cpp` must do the
      same lookup `console_executable()` does, in the same order, and add the item
      only when it resolves. **Note the layout constraint**: `menu_count()` returns
      8 on the main menu and there is a comment at `game_window.cpp:158` about a
      previous growth to seven pushing START out of place — a ninth item needs the
      layout checked, not just appended. `app/game_window.cpp` is a file the other
      agent has been editing; check before you start.
- [ ] **Windows/macOS console payloads** (Step 3) — installer components so a
      game-only selection omits them; on macOS the console as its own `.app`.
- [ ] **CI package matrices** (Step 5) — `.github/workflows/ci.yml`, plus
      `lintian` on the three Debian binaries.
- [ ] **The e2e claim** — *the two packages are two different products.* Neither
      half is written yet: game-only staging tree with Python off `PATH` has no
      TRAIN AI; full tree resolves the launcher and starts `md-console` offscreen.
- [ ] Debian packaging has **never actually been built** here — `dpkg-buildpackage`
      needs build-deps this box does not have. The `.install` paths and the
      `py3versions` invocation in `debian/rules` are unverified against a real
      build. Do that in CI before believing them.

## Two things the human reported that no test covers

Both are recorded in the plan; neither is fixed.

1. **A sound repeats endlessly after game over.** Reported from real play. Not
   reproduced — reading the code ruled out the obvious cause. Two candidates
   remain (voice-slot thrashing in `audio.cpp`'s `play()`, which always steals
   `voices[0]`; and audio-callback starvation, which fits "over and over" better).
   **Reproduce with sound first.** Fixing the voice stealing and declaring victory
   would very likely leave the real bug in.
2. **A watched agent must say on screen which one it is** — `SCRIPTED` or the
   model's display name — and that makes **naming runs and models** a requirement,
   not a nicety. Written into the plan as Task 3 Step 4b and Task 6 Step 1. The
   name belongs in `model.json`, not in a filename.

## Constraints that are not negotiable

| Rule | Why, and what enforces it |
|---|---|
| **The game never depends on Python or PySide6** | The game-only package is the promise. `test_packaging.py` now asserts it against `debian/control` directly. |
| **`md.ui` must never import torch** | So the console starts anywhere and a UI crash cannot touch a run. `test_ui_boundary.py` checks it twice. |
| **Zero warnings, `-Werror`, clang-tidy clean** | Production C++ only; tests are exempt from tidy. |
| **Catch2 test names must not contain `[` or `]`** | Reserved for tags — a name with them silently registers as a tag. |
| **SPDX + copyright + `Assisted-by` header on every file** | `Jens Köhler`, MIT. |
| **Comments explain *why*** | Heavy reasoning prose, not restatement. Match it or the diff looks foreign. |
| **e2e tests are invisible and silent** | Never put a window on the user's screen or sound through their speakers. |

## Environment facts, learned the hard way

* **`sudo` needs a password here.** You cannot install system packages. Say which
  one is needed and give the command.
* **xvfb is not installed**, so the game e2e tests skip. `MD_E2E_VISIBLE=1` works
  but puts windows on the user's screen — they have asked that it not be used.
* **`python -m md.train` needs `PYTHONPATH=python`** when run from the checkout.
* **A recording is only written when an episode *finishes*.** Any short run that
  needs an `.mdr` must pass `--max-ticks` (the e2e harness uses 400). This is the
  single most likely thing to waste an hour.
* **`poe pytest` excludes e2e**; `poe test-app` runs them; `poe check-all` runs both.
* **`poe check` aborts at `typecheck`** before reaching the tests — see below. Run
  `tidy`, `test`, `test-release`, `pytest`, `coverage` yourself afterwards.

## Pre-existing failures — not yours, do not "fix" them

Confirmed against untouched `master` in a worktree this session.

* **`mypy` fails wholesale** on numpy's stubs (`Type statement is only supported in
  Python 3.12 and greater`). The venv has numpy ≥ 2.5 while `python_version =
  "3.11"`; CI pins `numpy<2.5`. It aborts `poe check` at the `typecheck` step, so
  everything after it is skipped unless you run it by hand.
* **`pyright`: exactly 62 errors** — 55 in `tools/make_icon.py` (PIL), 7 in
  `python/md/env.py` (numpy). If you see 62, you added none.
* **Two Vulkan validation errors**, baselined in
  `python/tests/e2e/harness.py::KNOWN_VALIDATION_ERRORS`. Real renderer bugs that
  predate all of this. **Do not silently extend that baseline** — a new VUID
  failing the suite is the suite working.

## Hooks already built that the next tasks should use

* **`md.runtime`** — `Runtime` (status/install/repair/remove), `recommend()`,
  `RuntimePlan`, `venv_python()`. Transactional install, checksum-signed manifest.
* **`md.ui.runner`** — `find_interpreter()` for starting a trainer;
  **`console_executable()` / `console_command()`** for finding the console. Task 4's
  C++ menu lookup must match the latter's search order exactly.
* **`md.paths`** — `runs_dir()` and `runtime_dir()`. Anything new that persists
  belongs behind a function here, not a literal path.
* **`python/tests/e2e/harness.py`** — `run_app()`, `train()`, `agent_eval()`,
  `wait_until()`, `build_wheel()`, and the `needs_*` skip guards.
* **`md_app` flags** — `--frames N`, `--until-done`, `--silent`, `--report`.
  Task 8's `--match` should follow the same shape and extend the report.
* **`md.control` / `md.modelcard`** — the pattern for "a file both sides agree on
  and neither imports the other for". Tasks 6, 7 and 9 need more of these.
