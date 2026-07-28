# Testing & Code Quality

This project is developed **test-first**. Everything below runs locally via `poe`
(after `. .venv/bin/activate`), and the same tasks form the quality gate.
Create or update that environment with `python3 -m tools.bootstrap`.

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
  the game binary, a real training run, the trainer window. See below.
- Catch2 test names must **not contain `[` `]`** — those are reserved for tags.

## Application end-to-end tests

Everything else here tests code. These test the **program**: a real binary, real
files in a temporary directory, and assertions on what a person could observe.
If a claim can be checked by calling a function, it is a unit test and belongs
beside the code — the distinction is the process boundary, not the size of the
assertion.

They cover four subjects:

| Subject | Driven as | The claim |
|---|---|---|
| The game | `md_app` with a frame budget | boots, renders, plays, watches, replays, exits 0 |
| A training run | `missile-defense-train` into a temp `--out-dir` | writes every artifact it owes |
| The trainer | `missile_defense.ui` offscreen on that directory | attaches, reads, controls, sets up a runtime |
| The contest | promotion, pairing, `--match` split screen | a promoted model is playable, a pairing plays as one screen, a broken one is refused |

The contest row was marked *not built yet* long after it was; `test_promotion.py`
and `test_match.py` cover it. What is **not** yet covered end to end is the full
league journey as a packaged user meets it — see the 1.0 completion plan.

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
* the trainer runs under `QT_QPA_PLATFORM=offscreen`, which needs nothing
  installed at all.

Qt's `offscreen` platform is *not* an option for the game: it has no Vulkan
support (`This plugin does not support createPlatformVulkanInstance`), so the
game cannot start under it. Hence Xvfb, which is a real X server.

> **Assert on `AppRun.output`, never `AppRun.stderr`.** `xvfb-run` runs its
> command as `"$@" 2>&1`, so under the wrapper — which is every run in CI — the
> two streams are merged before pytest sees them and `stderr` is always empty.
> This is not a nicety: `validation_errors()` grepped `stderr` for Vulkan `VUID`
> lines, found none, and pronounced every run's renderer clean for as long as
> this suite had run under Xvfb. A zero-validation-error gate that reads the
> wrong stream reports success forever.

```bash
sudo apt install xvfb     # Debian/Ubuntu — without it the game tests skip
poe test-app              # the whole application e2e suite
poe check-all             # the full gate plus this suite

MD_E2E_VISIBLE=1 poe test-app   # watch them on your own screen instead
poe test-wayland          # the five that can only run on your own screen
```

Without Xvfb the game tests **skip** with that instruction as the reason; the
trainer and training tests still run. That is the right behaviour on a build box
with no graphics stack.

### The Wayland exception

Five tests cannot be made invisible: they exercise Qt's *Wayland* plugin (see
[WAYLAND.md](WAYLAND.md)), and an X server — virtual or not — cannot stand in for
a compositor. A nested `kwin_wayland --virtual` was tried and cannot composite on
this NVIDIA box either. So they run on the real desktop, with real windows taking
the focus of whoever is at the keyboard, which is why **nothing runs them
automatically**: not `poe test-app`, not `poe check-all`. They skip everywhere
with `poe test-wayland` named as the way to run them, and that task is the
deliberate act of asking.

Run them after touching the renderer's start-up or teardown, or Qt/driver
packages — that is what they are evidence about. They also cost about thirty
seconds, so there is no reason to avoid them beyond the interruption.

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
   `VK_LAYER_KHRONOS_validation`, and *any* `VUID` in a run's output fails the
   test — which is how "did it render correctly?" is checked without capturing a
   pixel.

> **There is no allow-list, and adding one back needs a very good argument.** Two
> validation errors were baselined when this suite first ran, and both are now
> fixed rather than excused: `VkShaderModuleCreateInfo-pCode-08740`, which was an
> instance declaring `apiVersion = 0`, and
> `VUID-vkAcquireNextImageKHR-semaphore-01779`, which is a genuine `QVulkanWindow`
> defect worked around in `Renderer::submit`. A baseline of errors we have decided
> to live with stops being a to-do and becomes background noise, and every
> renderer change after it gets reviewed against a dirty baseline.

`app/tests/vulkan_baseline.cpp` is a bare `QVulkanWindow` with none of this
project in it. `test_vulkan_validation.py` runs it to prove that 01779 is Qt's
and not ours, and — more usefully — to fail the day Qt stops raising it, because
that is the day the workaround should be deleted.

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
| Shader validity | `poe vulkan-shaders` | `spirv-val` against `vulkan1.0`, the version the instance asks for |
| Renderer validity | `poe vulkan-runtime` | the game under `VK_LAYER_KHRONOS_validation` + synchronization validation; any message fails |

Run **everything** with:

```bash
poe check     # format + lint + types + tidy + shaders + tests (Debug AND Release)
poe check-all # the above, plus Vulkan runtime validation and the application e2e suite
```

### What the Vulkan gates need

```bash
sudo apt install spirv-tools   # provides spirv-val, which `poe check` now runs
```

`poe vulkan-shaders` is in `poe check` because it costs a second and needs no
GPU, display or build. `poe vulkan-runtime` is in `poe check-all` because it
starts the real game once per rendering scenario.

Neither skips quietly when its tools are missing — they exit with the package to
install. A check that silently does nothing reports green, which is precisely how
the validation gate here sat inert for weeks.

`poe vulkan-runtime --best-practices` additionally runs the best-practices layer.
Its findings are **printed and do not fail the gate**: what it currently reports
is that `VK_EXT_debug_utils` is a debugging extension (enabled because we are
debugging) and that a few buffers are small enough to be worth sub-allocating.
Correctness and synchronization hazards gate; performance advice is printed.

Both configurations are built and tested: **Debug** (`-O0`, ASan/UBSan) and **Release**
(`-O2`, `NDEBUG`). This matters because `-Werror` is on in both and several warnings are
optimization-sensitive (they only fire under `-O2`), and because `NDEBUG` changes code
paths. `poe test-release` runs the Release build + tests on their own.

### Why the end-to-end suite runs the game with `detect_leaks=0`

The debug build carries LeakSanitizer, and running the game under it reports
leaks — none of which this project allocates. Every block is allocated inside
libxcb's event queue, libdbus, the Vulkan loader, or the NVIDIA driver's own
initialisation (on X11, most of it behind a `QOpenGLContext` that the *KDE
platform theme* creates and destroys). The game's own allocations are Qt-parented
or RAII and are freed; its Vulkan objects are destroyed by
`releaseSwapChainResources()` and `releaseResources()`, which the validation
layer confirms by reporting nothing outstanding at `vkDestroyInstance`.

So the suite drops that one check (`harness.app_environ`) and keeps every other
ASan check, use-after-free above all. It is not a leak budget: if a leak of ours
ever appears, it will be visible as a `md::` frame at the allocation site, and
the way to look is

```bash
ASAN_OPTIONS=detect_leaks=1 build/debug/app/md_app --play --frames 90 --silent
```

There used to be one exception to "none of it is ours": the `VkSurfaceKHR` the
Wayland workaround orphaned, 72 bytes in the loader. `GameWindow::event` now
destroys it — [WAYLAND.md](WAYLAND.md) says why that is safe only at exactly one
moment.

## torch, and the seventy-six tests that need it

Seventy-six tests import torch — 20 unit and 56 e2e, thirteen of the unit ones
`test_league.py`, which covers model promotion. Without it they skip, and
`python3 -m tools.bootstrap` therefore installs it as a step of its own
(`TEST_TOOLS`, `TEST_INDEX` in `tools/bootstrap.py`). On an existing checkout it
is the same one command:

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

**The CPU wheel, deliberately.** The alternative is three gigabytes of CUDA that
would be the *second* copy on the machine — the trainer's managed runtime already
holds the one real training uses. And the venv copy cannot quietly become that
one: `runner.find_interpreter` prefers the managed runtime over "this
interpreter", so a venv torch is what the tests import and never what a training
run gets.

**Skipping is right; skipping everywhere is not.** On a machine with no torch a
skip is the correct answer. On this project's own development machines it is a
lie — torch is there, in the managed runtime, and the tests simply cannot see it.
Seventy-six tests that only ever run in CI is how a failure reaches a release,
which is why the CI gate installs it too. It did not until recently, and the cost
was concrete: two of these tests sat broken for a day after `_log_eval` gained
the handicap columns, with a green pipeline the whole time.

**Still shipped to nobody**, and this is not where that is decided: torch is
`[project.optional-dependencies].train`, a `Suggests` in `debian/control`, and an
extra no install pulls. The development venv is the only place it appears.

The bootstrap step is allowed to fail — a large download from a second index —
so a machine that cannot reach it still gets a working checkout.

## Optional: git pre-commit hook

```bash
poe install-hooks   # runs format + lint + build + unit tests before each commit
```

## Toggles

- `-DMD_WERROR=OFF` — disable warnings-as-errors (e.g. trying a new compiler).
- `-DMD_SANITIZE=OFF` — disable ASan/UBSan (on by default in the `debug` preset).
