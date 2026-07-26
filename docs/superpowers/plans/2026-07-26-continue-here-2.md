# Continue here (2) — after the policy chain

Written 2026-07-26, replacing the task inventory in
`2026-07-26-open-points-implementation-prompt.md`. That file's *traps* and
*constraints* sections are still correct and still worth reading;
its task list is now out of date in the ways below.

Everything here was verified against the tree, not read off a plan.

---

## What landed

| Commit | Task | What |
|---|---|---|
| `ddf75a2` | **4** | TRAIN AI menu entry + `app/console.cpp` lookup, Windows `.cmd` / macOS `.app` console payloads, `game`/`python` install components, Debian CI proving the three binaries are three products |
| `1cf6c40` `40b0d06` | **12** | The console's STATISTICS tab: 14 stat tiles, the kills-per-shot distribution, four cause curves, deltas against the compared run |
| `b22c01b` | **1** | The `.mdp` format |
| `de61d1c` | **2** | `md.export_policy`, `md::agent::Policy`, `Driver`/`PolicyDriver`, cross-process parity |
| `46b27dd` | **3** | `--watch-model`, the WATCH AI submenu, the HUD naming the agent |
| `2f458de` | **6 (services)** | `md.library`, `md.league`, `paths.models_dir()` |

`poe check`'s components pass except the two pre-existing failures below.
`ctest` is 142/142. pyright is 62, unchanged.

---

## The one thing that blocks shipping an agent

**Every checkpoint in existence is unloadable.** They declare `obs_size` 1895;
the simulation encodes **1959**. `blast_features` went 4 → 5 (blast lifetime
phase) in `a100aec` — the same commit that moved the baseline to 98,542 — which
is +64 floats over 64 blast slots.

```
runs/                         obs=1895  iter=800   mean 79,724   STALE
runs/mlp-measures             obs=1895  iter=1300  mean 76,743   STALE
runs/entity-1024              obs=1895  iter=550   mean 15,864   STALE
runs/entity-4096              obs=1895  iter=200   mean  9,744   STALE
```

So there is nothing to re-evaluate and nothing to bundle. The human chose "ship
`runs/` update 800" before this was known; the choice is moot until a run on the
current simulation produces something.

**Everything around it is done and waiting.** Drop a `models/pretrained.mdp` in
and: `app/CMakeLists.txt` installs it under component `game` on all three
platforms, `GameWindow::pretrained_path()` finds it in all three layouts, WATCH
AI grows a submenu naming it, and the HUD says its name while it plays. One
export is the whole remaining step.

`md::agent::PolicyDriver` refuses an incompatible policy at construction rather
than feeding it a short observation — `test_policy.cpp` pins that, and it is the
common case now, not an exotic one.

---

## Open, in the plan's order

### Task 6 — the two screens

The services are committed and tested (`test_library.py` 23, `test_league.py` 18).
What is missing is UI:

- [ ] `python/md/ui/library.py` — the run library as the console's landing view.
      One row per run: name, state, best score, last activity, disk, promotion
      status. Opening a row shows the dashboard unchanged. `md.library.discover`
      and `md.library.describe` already produce everything a row needs.
- [ ] `python/md/ui/league.py` — the Model League table, and **Enter Model
      League** on a stopped run: default checkpoint (`best_evaluated_checkpoint`),
      alternatives, name, note, and the export validation `league.promote`
      already performs.
- [ ] Rename and note editing, which `library.rename` / `library.set_note`
      support and nothing calls.
- [ ] **E2E:** promote from the library, then assert the **native** evaluator
      loads and plays the resulting `policy.mdp`. `md_agent_eval --policy` does
      exactly that already, so this is a short test.

Naming defaults for a *new* run (`library.default_name`) are written and the
new-run dialog does not use them yet.

### Task 7 — league evaluation and paired matches

Nothing exists. `md/tournament.py`. Two things are already in place that the
plan assumed would need building:

* `md_agent_eval --policy <mdp>` runs a learned contestant through the *same*
  `run_episode`, event tallying and `summarize` as the scripted one.
* `--action-log <file>` writes one index per sampled decision, which is what a
  paired match manifest needs for both sides.

Still to do: one machine-readable row per seed (the evaluator prints a human
block today), the canonical-completeness rules, and `league.record_result` being
called with `canonical: true` only when the whole protocol matches
(`md.benchmark.canonical_baseline_comparable` is the existing check).

### Task 8 — split-screen spectator

Nothing exists. **The human chose the CLI shape**: `--match <manifest>` *plus*
ad-hoc `--match-left <a.mdr> --match-right <b.mdr>`, so same-seed/config
enforcement has to live outside the manifest as well as in it.

### Task 9 — cleanup, archive, restore, export, import

Nothing exists except the halves the library and league already carry:
`library.Storage` splits a run's bytes three ways for a cleanup plan,
`library.within()` refuses a path outside the managed root, pins are stored and
read, and `league.import_policy` validates an imported `.mdp` before writing.
Write the archive *security* tests first — unsafe extraction is the risk.

### Task 10 — docs and the default journey

Last, by design. Note that README's "there is not one to run yet" is **still
true** and should stay until a model ships.

---

## Loose ends

- [ ] **The hermetic runtime-install e2e** — the one retro-fit that never
      landed. `harness.build_wheel()` exists for it; no test drives a real
      install against a local `--find-links` directory.
- [ ] **The game-over sound loop.** Another agent has `app/voices.hpp` and
      `app/tests/unit/test_voices.cpp` in flight — the voice-bank half. Still
      **not reproduced**, and the handover's warning stands: fixing the voice
      stealing and declaring victory would likely leave the real bug in.
- [ ] The two baselined VUIDs, untouched and still baselined. The baseline was
      not extended.

## Two things another agent did to my work, recorded rather than undone

* `3aa6119` swept my in-flight `app/main.cpp` changes (`--watch-model`, the
  report's `driver` and `pretrained` fields) into a release-readiness commit
  whose message does not mention them.
* Someone is **extending Task 2 to the `entity` architecture** right now —
  `policy_format.ARCHITECTURES`, `export_policy`, and a whole relational forward
  pass in `agent/src/policy.cpp`. Good, and it means those four files were
  deliberately left alone by this session after `de61d1c`.

## Pre-existing failures — still not yours

* **mypy** fails wholesale on numpy's stubs; it aborts `poe check` at
  `typecheck`, so run `tidy`, `test`, `test-release`, `pytest`, `coverage`
  yourself afterwards.
* **pyright: exactly 62.** If you see 62, you added none.
* `clang-format-21` flags `bench/main.cpp` and
  `bindings/tests/unit/test_vec_env.cpp` on master. Not new.
