# Testing & Code Quality

This project is developed **test-first**. Everything below runs locally via `poe`
(after `. .venv/bin/activate`), and the same tasks form the quality gate.

## Test layers

| Layer | Location | Framework | CTest label | Run with |
|-------|----------|-----------|-------------|----------|
| C++ unit | `core/tests/unit/` | Catch2 v3 | `unit` | `poe test-unit` |
| C++ simulation e2e | `core/tests/e2e/` | Catch2 v3 | `e2e` | `poe test-e2e` |
| Python | `python/tests/` | pytest | — | `poe pytest` |
| **Application e2e** | `python/tests/e2e/` | pytest | — | `poe test-app` |

- **Unit** tests are fast and isolated — the TDD inner loop is just `poe test-unit`
  (it auto-builds first).
- **Simulation e2e** tests drive a whole episode (`reset` → `step`… → terminal) and
  assert end-to-end invariants (determinism, scoring, termination), in-process.
- **Application e2e** tests drive the *shipped surfaces* in their own processes —
  the game binary, a real training run, the console window. See below.
- Catch2 test names must **not contain `[` `]`** — those are reserved for tags.

## Application end-to-end tests

Everything else here tests code. These test the **program**: a real binary, real
files in a temporary directory, and assertions on what a person could observe.
If a claim can be checked by calling a function, it is a unit test and belongs
beside the code — the distinction is the process boundary, not the size of the
assertion.

They cover four subjects, three of which exist today:

| Subject | Driven as | The claim |
|---|---|---|
| The game | `md_app` with a frame budget | boots, renders, plays, watches, replays, exits 0 |
| A training run | `md-train` into a temp `--out-dir` | writes every artifact it owes |
| The console | `md.ui` offscreen on that directory | attaches, reads, controls, sets up a runtime |
| The contest | league + tournament | *(not built yet — see the journey plan)* |

The one worth knowing about is **the replay round-trip**. A recording is
`(seed, config, action indices)` and nothing else, so replaying one correctly
means the C++ simulation reproduced, tick for tick, what the Python trainer's
batched environment did — through a file, across a language boundary, in a
separate process. If anything in the simulation, the encoding or the recorder
drifts, the score comes out different. It is the strongest single assertion in
the suite.

### They are invisible and silent by design

A test suite that throws game windows onto the screen of whoever is at the
keyboard is a suite people stop running, and one that steals focus mid-run can
lose them what they were typing. So:

* the game renders into **Xvfb**, a real X server with no monitor attached;
* it is started with **`--silent`**, which also suppresses persisting that, so a
  test run never leaves someone's sound switched off;
* everything it would write — recordings, high scores, the audio and fullscreen
  preferences — is redirected into a temporary directory via `XDG_CONFIG_HOME`,
  `XDG_DATA_HOME` and `MD_RUNS_DIR`;
* the console runs under `QT_QPA_PLATFORM=offscreen`, which needs nothing
  installed at all.

Qt's `offscreen` platform is *not* an option for the game: it has no Vulkan
support (`This plugin does not support createPlatformVulkanInstance`), so the
game cannot start under it. Hence Xvfb, which is a real X server.

```bash
sudo apt install xvfb     # Debian/Ubuntu — without it the game tests skip
poe test-app              # the whole application e2e suite
poe check-all             # the full gate plus this suite

MD_E2E_VISIBLE=1 poe test-app   # watch them on your own screen instead
```

Without Xvfb the game tests **skip** with that instruction as the reason; the
console and training tests still run. That is the right behaviour on a build box
with no graphics stack.

### The flags this needed

`md_app` blocked in `QGuiApplication::exec()` until a human closed the window, so
nothing could drive it. Three flags fixed that, and none of them is only a test
affordance:

| Flag | What it does | Also useful for |
|---|---|---|
| `--frames N` | quit after N rendered frames | timing a renderer change |
| `--until-done` | quit when the game or recording ends | attract mode, a kiosk |
| `--silent` | no sound, and do not persist that | anything unattended |
| `--report` | one JSON line: mode, state, frames, ticks, score, wave, cities | benchmarking |

### Rules

1. **No `sleep`** — `harness.wait_until` polls with a deadline. A sleep long
   enough for a loaded runner wastes minutes locally; one tuned locally is next
   month's CI flake.
2. **Skip, never fail, on a missing optional half.** torch and PySide6 are
   optional by design, so a machine without them reports *skipped* with the
   package named.
3. **Every run is bounded.** A hang must surface as a failed test, not a job that
   runs until the runner gives up.
4. **Validation errors are failures.** The debug build enables
   `VK_LAYER_KHRONOS_validation`, and any `VUID` not in
   `harness.KNOWN_VALIDATION_ERRORS` fails the test — which is how "did it render
   correctly?" is checked without capturing a pixel.

> **Two known validation errors are recorded there**, both found by this suite the
> first time it ran and both pre-existing: a swapchain acquire semaphore reused
> while a wait on it is still pending, and a SPIR-V capability whose environment
> requirement is not met. They are baselined so that *new* ones fail immediately
> rather than being buried — not because they are acceptable.

## The TDD loop

```bash
. .venv/bin/activate
poe test-unit      # write a failing test, watch it fail, implement, watch it pass
```

## Quality gates (zero-warning policy)

Production code (the sim / game / AI) must compile and lint **perfectly clean**.
Tests and glue code share the same warnings but are not held to clang-tidy.

| Gate | Task | Enforcement |
|------|------|-------------|
| C++ format | `poe format-check` | clang-format-21, `--Werror` |
| C++ warnings | (build) | `-Wall -Wextra -Wpedantic …` **`-Werror`** on `md::warnings` |
| C++ static analysis | `poe tidy` | clang-tidy-21, `--warnings-as-errors=*` (production sources only) |
| Python format | `poe fmt-py-chk` | ruff format `--check` |
| Python lint | `poe lint` | ruff check |
| Python types | `poe typecheck` | mypy `--strict` |

Run **everything** with:

```bash
poe check     # format + lint + types + tidy + tests (Debug AND Release) — full local CI
```

Both configurations are built and tested: **Debug** (`-O0`, ASan/UBSan) and **Release**
(`-O2`, `NDEBUG`). This matters because `-Werror` is on in both and several warnings are
optimization-sensitive (they only fire under `-O2`), and because `NDEBUG` changes code
paths. `poe test-release` runs the Release build + tests on their own.

## Optional: git pre-commit hook

```bash
poe install-hooks   # runs format + lint + build + unit tests before each commit
```

## Toggles

- `-DMD_WERROR=OFF` — disable warnings-as-errors (e.g. trying a new compiler).
- `-DMD_SANITIZE=OFF` — disable ASan/UBSan (on by default in the `debug` preset).
