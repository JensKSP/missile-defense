<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Jens Köhler -->
<!-- Assisted-by: Claude Code (Anthropic) -->

# Handover — 2026-07-27

Written at the end of a session that ran out of room. Everything below is
pushed; `master` is at `fed710a` and the working tree is clean.

## Open, in the order I would take them

### 1. Six trainer end-to-end tests fail

```
python/tests/e2e/test_console.py
  test_a_new_run_starts_the_tiles_at_nothing_notes_included
  test_a_protocol_change_starts_a_new_score_curve_and_controls_the_ladder
  test_the_run_a_console_would_start_is_a_command_you_could_type
  test_a_preset_fills_the_form_and_editing_it_stops_claiming_to_be_one
  test_the_dialog_says_what_a_run_will_cost_the_card_before_it_starts
  test_the_dialog_saves_a_preset_and_refuses_to_overwrite_a_built_in
```

**These predate this session** — the same six failed in the CI run for `e42ff4e5`.
They are not in `poe check` (the e2e suite is deselected there), so the local
gate is green while CI's `app-e2e` job is red.

Run them with:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest python/tests/e2e/test_console.py -q
```

**One cause is fixed, and it was not the only one.** `missile_defense.runs.sources.is_canonical_benchmark`
compares `aim_trail` and `reaction_delay`, which the handicap work made part of
the protocol — but `EVAL_COLUMNS` in `missile_defense/training/train.py` never gained them. Every row
read as "nonstandard protocol", so no score could be shown against the ladder and
the trainer could never say a run beat HIGH. The training loop writes both columns now
and `test_console.py`'s `EVALS_HEADER` was widened to match; the first two tests
still fail afterwards, so there is at least one more reason. Start by diffing the
expected and actual tile note in `test_a_new_run_starts_the_tiles_at_nothing_notes_included`
with `-v`.

The four dialog tests may well be a different fault again — I never got as far as
reading their failures.

### 2. Confirm CI

Four commits went out this session and I never saw a green run for any of them:
`c1776fb`, `9058020`, `33e43ce`, `3811f11`, `fed710a`.

```bash
gh run list --limit 5 --json workflowName,status,conclusion,databaseId \
  --jq '.[] | "\(.conclusion // .status)\t\(.workflowName)\t\(.databaseId)"'
```

Known-fixed since the last observed failure, unverified in CI:

* `app-e2e` — `VUID-vkDestroyInstance-instance-00629` on every scenario, caused by
  applying the Wayland workaround on xcb. Gated on the platform now, and the local
  `poe vulkan-runtime` (which runs under Xvfb/xcb exactly as CI does) passes.
* `gate` and `artifacts / macos` — both were the same single test importing PySide6
  where it is not installed. `test_library.py` now skips instead.
* A new `bootstrap` job exists and has never run.

### 3. Windows

Jens intended to reboot to Windows and test the installer. The `artifacts / windows`
job was **green** in the last observed run, so the artifact is worth testing.

## What changed this session

**The Wayland exit crash is fixed and the game runs natively on Wayland.**
`GameWindow::event` releases the Vulkan instance on `Close`. Everything about it —
the upstream defect (QTBUG-123214, untriaged since 2024, unchanged through Qt's dev
branch), the cost (the `VkSurfaceKHR` leaks, and that leak *is* the mechanism), and
why it is gated on the platform — is in [WAYLAND.md](WAYLAND.md) and in the long
note on the function. `app/tests/wayland_teardown.cpp` is a bare `QVulkanWindow`
witness run in two modes, and `python/tests/e2e/test_wayland_teardown.py` asserts
three things that can each fail alone: the cause still exists, the effect is the
workaround's, and the shipped binary still applies it. All three skip without a
compositor, so they run on Jens's desktop and nowhere else.

`poe app`, the trainer and the desktop entry no longer pin a platform. `poe shot`
is the one exception and says why in `tools/capture.py`: a Wayland client cannot
have its window grabbed.

**The reward is shown as an equation** in the config panel, with the run's own
numbers and one clause per term saying why it is priced. `missile_defense/ui/reward.py` has no
Qt in it and is tested directly.

**Three faults a clean worktree exposed**, none of which CI could see, because
every job installs its own dependency list and so never walks the path a person
walks:

* `tools/bootstrap.py` did not install nanobind, so CMake never created the
  `_md_native` target and `poe bindings` failed with `unknown target`;
* the runtime installer downloaded five gigabytes of CUDA torch, health-checked it
  by importing the binding, found none, reported **"could not import torch"** —
  naming the one thing that had worked — and deleted the directory, torch included.
  It now names the import that actually failed and checks for the binding *before*
  the first byte;
* my own Wayland fix was applied on every platform, which is what tripped the
  Vulkan gate.

The new `bootstrap` CI job runs the documented command and then asks the only
question that matters about its result.

**`Runtime.verify()`** asks the runtime to prove itself rather than trusting the
manifest, and the trainer now runs it on a background thread and turns Start back
into Set up when it says no. The two-minute stall I could not explain in an earlier
commit was a cold CUDA start after a fresh install — one-time, ~1 s warm. The
window waits for that thread on close, because Qt aborts the process when a running
`QThread` is destroyed.

## Things I got wrong, so they are not re-derived

* I claimed the Wayland workaround **cost no leak**. It leaks the `VkSurfaceKHR`,
  and that is the point of it. CI found this, not me.
* I claimed the bootstrap gap had existed **"for months"**. The repository's first
  commit is 2026-07-24. It was days.
* Earlier in the session I claimed the Wayland crash was **not NVIDIA-specific**
  from a lavapipe test that had never swapped the crashing component. The
  conclusion was right and the evidence was not; it is now properly measured
  (lavapipe crashes identically, 24/24 on both drivers).

## Standing constraints

* `poe check` is the gate and is currently green.
* A milestone passes only on Jens's confirmation — do not self-declare done.
* Other agents commit to this branch; stage explicit paths, never a directory.
* `git clean -ffdx` needs the second `-f` (nested git checkouts under
  `build/*/_deps/`), and takes `runs/` and any promoted `models/<name>/` with it
  permanently. See the README's "Starting from a clean tree".
