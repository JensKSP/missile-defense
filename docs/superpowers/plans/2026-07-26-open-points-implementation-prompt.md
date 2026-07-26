# Open points — implementation prompt (everything except model training)

Written 2026-07-26. The inventory below was **verified against the tree**, not
copied from the plan: where the plan and the code disagree, the code won.
`2026-07-26-continue-here.md` remains correct about the traps; this file replaces
its task inventory.

---

## The prompt

> You are continuing an implementation program on the **missile-defense** project
> (C++23 / Qt 6 / Vulkan game + PyTorch PPO trainer + PySide6 training console)
> on branch `task-5-managed-training-runtime`.
>
> **Read first, in this order:**
> 1. This file — the scope, the order, and the decisions to raise before building.
> 2. `docs/superpowers/plans/2026-07-26-ai-training-user-journey.md` — the plan and
>    the source of truth. Its **End-to-end tests** section states rules that apply
>    to every task, and every task carries an **E2E:** claim it must make true.
> 3. `docs/superpowers/plans/2026-07-26-continue-here.md` — the traps, the
>    constraints, and the pre-existing failures that are not yours to fix.
>
> **Your scope is every open point in the plan except model training.** Tasks 5
> and 11 are complete. Do not train a policy, change PPO hyperparameters or the
> network architecture, start or resume a run, or re-measure the scripted
> baseline — another agent owns all of that, and some of it is explicitly waiting
> on a human signal.
>
> **Order** — deliberate, not numeric: finish **Task 4**, then the policy chain
> **1 → 2 → 3**, then **6 → 7 → 8 → 9**, then **12**, and **10** last, because 10
> documents what the others build. **Task 12 has no dependency on any of them** —
> take it earlier if you want a self-contained win, or if the C++ work is blocked.
>
> **How to work.** Test-first, always: write the failing test, watch it fail,
> implement, watch it pass. A task ends with green focused tests, its e2e claim
> satisfied, `poe check` clean, documentation updated, the plan's checkboxes
> ticked, and **one commit** staging **explicit file paths**.
>
> **Ask before assuming** on anything that changes what gets built — the
> "Decisions to raise first" section below lists the ones already known. Do not
> self-declare a milestone done; a milestone passes only when the human confirms
> it by using it.

---

## Out of scope — model training

Owned by another agent, and in part waiting on a human signal. Do not touch:

* PPO hyperparameters, the relational/entity architecture, the auxiliary head,
  annealing schedules, `md.auxiliary`, `md-multiseed` policy.
* Starting, resuming or stopping training runs, including the recommended
  `runs/relational-3seed` experiment.
* Re-measuring or restating the scripted baseline. **98,542.34375** on the
  held-out canonical block is authoritative; `python/md/benchmark.py` is its
  source of truth. The older **113,834** is stale wherever it still appears.
* Deciding *which* learned policy is good enough to ship (see Decision 1).

Touching `python/md/train.py`, `ppo.py`, `control.py` or `ui/app.py` for a
*mechanical* reason (a rename, a new artifact the console must read) is fine and
sometimes unavoidable — but coordinate, because they are being edited right now.

---

## Decisions to raise first

Each of these changes what gets built. Ask before you build past them.

1. **There is no model worth bundling yet (blocks Task 3 Step 3).** The best
   learned policy is well behind the scripted baseline, and the checkpoint the
   older handover pointed at
   (`runs/player-model-v2-from-zero/checkpoints/policy-best.pt`) **no longer
   exists** — only its recordings remain. What survives is under
   `~/.local/share/MissileDefense/runs/{entity-1024,entity-4096,mlp-measures}/checkpoints/`.
   Build all of Task 3's *machinery* against an exported fixture policy, and ask
   the human which model — if any — ships, and what happens in the menu until one
   does. Shipping a bundled agent that loses to the scripted one is a product
   decision, not yours.
2. **Task 4's CI step is partly done, and duplicating it would be worse than
   leaving it.** `.github/workflows/build.yml` already runs `dpkg-buildpackage`
   and `lintian`. The plan asks for package matrices in `.github/workflows/ci.yml`.
   Confirm whether the intent is a second job in `ci.yml` or extending the
   existing `build.yml` one — then close the real gap, which is that **nothing
   verifies the three binaries are three different products**.
3. **What a "match" is on the command line (Task 8).** The plan says `--match`
   should follow the shape of `--frames` / `--until-done` / `--silent` /
   `--report`. Two policies, two seeds, one window — confirm the surface before
   writing the parser.

---

## Open points, verified against the tree

### Task 4 — package and discover the console (finish first)

Done and confirmed: `debian/control` builds three binaries (`missile-defense`,
`python3-md`, `missile-defense-training`), `debian/*.install`, `debian/rules`,
`MD_INSTALL_PYTHON_PACKAGE`, `packaging/launcher.in`, the console `.desktop`,
`runner.console_executable()` / `console_command()`, and the contract tests.

- [ ] **The TRAIN AI menu entry** (Step 4). Absent from `app/` — verified. It must
      do the same lookup as `console_executable()`, **in the same order**, and add
      the item only when it resolves. The layout constraint in the plan is
      **milder than it reads**: `menu_item_top_y()` already shrinks the step for
      longer lists via `active_count()` (`app/game_window.cpp:155-170`), so a
      ninth item will not push START into the byline the way seven once did.
      Check it visually anyway. `menu_count()` returns 8 on the main menu.
- [ ] **Windows/macOS console payloads** (Step 3). Genuinely untouched:
      `packaging/` holds only icons, `launcher.in` and the two `.desktop` files,
      and `build.yml` never mentions the console. Installer components so a
      game-only selection omits them; on macOS the console as its own `.app`.
- [ ] **CI package matrices** (Step 5) — see Decision 2.
- [ ] **The e2e claim: the packages are two different products.** Neither half
      exists. A game-only staging tree with Python off `PATH` shows no TRAIN AI;
      a full tree resolves the launcher and starts `md-console` offscreen.
- [ ] **Debian packaging has never been built on this box** — `dpkg-buildpackage`
      needs build-deps that are not installed and `sudo` needs a password here.
      The `.install` paths and `py3versions` in `debian/rules` are unverified
      against a real build. Trust CI, not this machine.

### Tasks 1 → 2 → 3 — the policy chain

Nothing exists: `grep` for `.mdp` across the tree returns **nothing**, there is no
native inference path, and `app/main.cpp` has no policy flag. This is the chain
that makes the README's "there is not one to run yet" section obsolete.

- [ ] **Task 1 — the portable `.mdp` format.** Round-trip and rejection tests
      first, then the format, then the compatibility promise in the docs.
- [ ] **Task 2 — native C++ loading and inference**, with a Python exporter
      fixture and **cross-process parity** proven against the Python policy.
- [ ] **Task 3 — bundle and watch it**: installed resource on every platform,
      game watch selection, game-only behaviour verified. Blocked on Decision 1
      for *which* model; the machinery is not blocked.
- [ ] **Task 3 Step 4b — say on screen who is playing.** Asked for directly by
      the human. Today `app/renderer.cpp:644` draws a bare `"AI PLAYING"`; it must
      say `SCRIPTED` or the model's **display name**, which makes naming runs and
      models a requirement rather than a nicety. The name belongs in `model.json`,
      not in a filename. Pairs with Task 6 Step 1.

### Tasks 6 → 7 → 8 → 9 — the contest

None of it exists. Each task carries its own e2e claim in the plan.

- [ ] **Task 6 — run library and explicit model promotion**: discovery and
      storage, the run library screen, **atomic** promotion, the league store,
      promotion UI.
- [ ] **Task 7 — league evaluation and paired matches**: canonical-completeness
      tests, native evaluator extension, tournament orchestration, Model League
      actions. Only complete, compatible canonical evaluations may rank.
- [ ] **Task 8 — synchronized split-screen spectator**: `MatchPlayer`, the match
      window, launch paths, controls and rendering. See Decision 3.
- [ ] **Task 9 — cleanup, archive, restore, export, import**: write the **archive
      security** tests first (unsafe extraction is the risk), then services, then
      the UI. Cleanup must protect promoted and pinned artifacts and report exact
      reclaimed bytes; archives verify before anything is deleted.

### Task 12 — statistical analysis in the console (independent)

Task 11 shipped the statistics and they are documented in `docs/API.md` and
`docs/TRAINING.md`; nothing consumes them in the UI. `python/md/ui/` has no
`stats.py` — verified.

- [ ] Kills-per-shot distribution as a bar chart.
- [ ] Survival and damage over the run — mean ticks survived, cities/bases lost.
- [ ] Per-run summary card: the full latest-eval stat block in one panel.
- [ ] Compare across runs, reusing the existing vs-run picker.
- [ ] One screen, live, dark, empty states handled (the M8 design intent).
- [ ] TDD `stats.py` under pytest with no display, then the view.

### Task 10 — make the UI journey the default (last)

- [ ] Installed-user acceptance scripts, README training entry rewritten,
      platform and packaging docs updated, full local gate, packaging CI.
- [ ] **Specifically**: `README.md`'s "Run the pre-trained, packed model" section
      currently exists to say there is not one, and "Run your own model in the
      game" routes through checkpoint → recording → replay. Both are honest today
      and both become wrong the moment Tasks 1–3 land. Rewrite them then, not
      before. The human has confirmed the current wording stands until it does.

### Not attached to any task

- [ ] **A sound repeats endlessly after game over.** Reported from real play,
      never reproduced. **Reproduce with sound first.** Two candidates:
      voice-slot thrashing in `audio.cpp`'s `play()` (always steals `voices[0]`,
      which should stop within a second and so does *not* match "over and over"),
      and audio-callback starvation (fits the report better, does not stop on its
      own). The voice-stealing defect is worth fixing either way, but fixing it
      and declaring victory without a repro would very likely leave the real bug in.
- [ ] **`VUID-vkAcquireNextImageKHR-semaphore-01779`** — one swapchain acquire
      semaphore where there must be one per frame in flight. Every frame, not an
      edge case. Undefined behaviour that happens to work on this driver.
- [ ] **`VUID-VkShaderModuleCreateInfo-pCode-08740`** — shaders compiled against a
      newer target than the instance requires.
      Both are baselined in `python/tests/e2e/harness.py::KNOWN_VALIDATION_ERRORS`.
      They are renderer bugs with their own tests, not part of this program —
      but **do not silently extend that baseline**: a new VUID failing the suite
      is the suite working.
- [ ] **The one missing e2e retro-fit: runtime setup installs, health-checks and
      is then used**, hermetically, against a local `--find-links` wheel directory
      rather than the network. `harness.build_wheel()` exists for exactly this;
      no test drives a real install yet.
- [ ] **The plan's other retro-fit checkboxes are stale.** `python/tests/e2e/`
      now has 38 tests across `test_game.py` (6), `test_training.py` (12),
      `test_console.py` (14) and `test_journey.py` (6), which cover the game, the
      recording round-trip, the training artifacts and the console attach.
      Verify and tick them rather than rewriting them.

---

## Constraints that are not negotiable

| Rule | Why, and what enforces it |
|---|---|
| **Other agents commit to this branch right now** | Stage **explicit paths**, never a directory; read `git diff --cached --name-only` before every commit. A `git add core/ agent/ bindings/` once swept another agent's work into an unrelated commit and broke the golden determinism test. `git status`/`git log` go stale inside a turn — re-run them immediately before committing. **Never revert their changes** (standing human instruction); if their work lands in your commit, say so. |
| **The game never depends on Python or PySide6** | The game-only package is the promise. `test_packaging.py` asserts it against `debian/control`. |
| **`md.ui` must never import torch** | So the console starts anywhere and a UI crash cannot touch a run. `test_ui_boundary.py` checks it twice. |
| **Zero warnings, `-Werror`, clang-tidy clean** | Production C++ only; tests are exempt from tidy. |
| **Catch2 test names must not contain `[` or `]`** | Reserved for tags — a name containing them silently registers as a tag. |
| **SPDX + copyright + `Assisted-by` header on every file** | `Jens Köhler`, MIT. |
| **Comments explain *why*** | Heavy reasoning prose, not restatement. Match it or the diff looks foreign. |
| **e2e tests are invisible and silent** | Never put a window on the user's screen or sound through their speakers. |
| **Anything that persists goes behind `md.paths`** | `runs_dir()`, `runtime_dir()` — not a literal path. |

## Environment traps

* **`sudo` needs a password here.** You cannot install system packages; name the
  package and give the command instead.
* **xvfb is not installed**, so the game e2e tests skip. `MD_E2E_VISIBLE=1` works
  but puts windows on the user's screen — the human has asked that it not be used.
* **`python -m md.train` needs `PYTHONPATH=python`** from the checkout.
* **A recording is only written when an episode *finishes*.** Any short run that
  needs an `.mdr` must pass `--max-ticks` (the e2e harness uses 400). This is the
  single most likely thing to waste an hour.
* **`poe pytest` excludes e2e**; `poe test-app` runs them; `poe check-all` runs both.
* **`poe check` aborts at `typecheck`** before reaching the tests — run `tidy`,
  `test`, `test-release`, `pytest`, `coverage` yourself afterwards.
* **Another `md_app` may be running.** Matching a window by title alone drives
  *that* one. Launch it, keep the PID, and match on `xdotool getwindowpid`.
* **`git fetch` fails with `Permission denied (publickey)`** unless you pass
  `SSH_AUTH_SOCK=/run/user/1000/gcr/ssh`.

## Pre-existing failures — not yours, do not "fix" them

* **`mypy` fails wholesale** on numpy's stubs (`Type statement is only supported
  in Python 3.12 and greater`): the venv has numpy ≥ 2.5 while `python_version =
  "3.11"`, and CI pins `numpy<2.5`. It aborts `poe check` at `typecheck`.
* **`pyright`: exactly 62 errors** — 55 in `tools/make_icon.py` (PIL), 7 in
  `python/md/env.py` (numpy). If you see 62, you added none.
* **The Debug build's Catch2 discovery** cannot run under a ptrace-restricted
  session because LeakSanitizer rejects ptrace. Environment limitation, not a
  test failure.
