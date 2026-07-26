# AI Training User Journey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an installed, UI-only journey for training, understanding,
promoting, comparing, spectating, archiving, and restoring learned Missile
Defense policies while preserving a Python-free game-only package.

**Architecture:** Keep the existing PySide6 console as the training application
and the C++ game as the renderer/spectator. Export promoted PyTorch checkpoints
to a versioned data-only native policy format that both the evaluator and game
load. Store runs, promoted models, matches, archives, and the managed training
runtime behind focused Python/C++ services rather than adding those
responsibilities to `python/md/ui/app.py` or `app/game_window.cpp`.

**Tech Stack:** C++23, CMake, Qt 6/Vulkan, Python 3.11+, PySide6, PyTorch,
nanobind, pytest, Catch2, scikit-build-core, Debian debhelper, CPack/NSIS, macOS
application bundles, ZIP archives with JSON manifests.

**Design:** `docs/superpowers/specs/2026-07-26-ai-training-user-journey-design.md`

---

## Scope decomposition

This is a program of independently shippable milestones. Complete them in order;
each milestone ends with green focused tests, the full applicable gate, updated
documentation, and a commit. Do not build the tournament UI before the exported
policy is the exact artifact being evaluated and rendered.

## File map

New files have one responsibility each:

| Path | Responsibility |
|---|---|
| `python/md/policy_format.py` | Read/write/validate the data-only native policy |
| `python/md/export_policy.py` | Convert and parity-check a PyTorch checkpoint |
| `agent/include/md/agent/policy.hpp` | Public native learned-policy interface |
| `agent/src/policy.cpp` | Parse policy files and run masked inference |
| `agent/tests/unit/test_policy.cpp` | Native format and inference tests |
| `python/md/runtime.py` | Detect, install, repair, and remove managed training runtime |
| `python/md/ui/runtime_dialog.py` | One-click runtime setup UI |
| `python/md/library.py` | Run/model/match discovery and storage accounting |
| `python/md/league.py` | Local model registry and immutable promotion |
| `python/md/tournament.py` | Canonical evaluation and paired-match orchestration |
| `python/md/archive.py` | Safe archive creation, verification, and restore |
| `python/md/ui/library.py` | Run library and cleanup/archive actions |
| `python/md/ui/league.py` | Model League and tournament controls |
| `replay/include/md/replay/match.hpp` | Paired replay manifest/player interface |
| `replay/src/match.cpp` | Paired manifest parsing and synchronized playback |
| `replay/tests/unit/test_match.cpp` | Match integrity and synchronization tests |
| `app/match_window.hpp` / `app/match_window.cpp` | Split-screen spectator state |
| `packaging/missile-defense-training.desktop` | Console desktop entry |
| `python/tests/e2e/harness.py` | Drive the shipped binaries as a user does |
| `python/tests/e2e/test_game.py` | The game: boot, play, watch, replay, exit |
| `python/tests/e2e/test_training.py` | A real short run, and every artifact it owes |
| `python/tests/e2e/test_console.py` | The console against a live run directory |
| `python/tests/e2e/test_journey.py` | The whole installed-user journey, in order |

Existing large UI files should compose these services; they should not absorb
their persistence, package-management, archive, or tournament logic.

---

## End-to-end tests

**Why this section exists.** Every layer of this project is tested except the one
the user actually touches. `core/tests/e2e/` drives a whole *simulation* — which
is not the same claim as "the game starts", "the console can start a run", or
"the recording that run wrote plays back". Those are the failures a person meets
first, and nothing in the tree would currently catch one.

**Definition.** An e2e test here **drives a shipped surface in its own process** —
a real binary, a real subprocess, a real window, real files in a real directory —
and asserts on what a user could observe. If it can be written by calling a
function, it is a unit test and belongs beside the code. The distinction is the
process boundary, not the size of the assertion.

### The four subjects

| Subject | Driven as | The claim it makes |
|---|---|---|
| **The game** | `md_app` with a frame budget | boots, renders, plays, watches, replays, exits 0 |
| **A training run** | `md-train` in a temp `--out-dir` | learns briefly and writes every artifact it owes |
| **The console** | `md.ui` offscreen on that directory | attaches, reads, controls, launches, sets up a runtime |
| **The contest** | league + tournament CLIs and UI | promotes, evaluates fairly, ranks, and plays a match back |

The fourth does not exist yet; Tasks 6–8 build it, and each carries the e2e that
proves it. The first three exist now and get theirs immediately, because the
point of adding tests early is to have them *before* the next task moves the
ground under them.

### What has to change first

The game cannot be tested at all today: `md_app` blocks in `QGuiApplication::exec()`
and only ever leaves it when a human closes the window, so CI can do nothing with
it but kill it and guess. Three small flags fix that, and they are the
prerequisite for every game-level e2e below:

* `--frames N` — quit after N rendered frames. A hard upper bound, so a hung
  window is a *failure* rather than a job that runs until the runner times out.
* `--until-done` — quit as soon as the game or the recording ends. This is the
  one that makes a replay assertion deterministic: a recording has a fixed
  length, so "play it to the end and say what happened" has exactly one answer.
* `--report` — write one JSON line describing the end state (mode, ticks, score,
  wave, cities, how it ended). This is the assertion surface; without it the only
  observable is an exit code, which cannot tell "played a game" from "showed a
  menu for four seconds".

These are test affordances, but not *only* test affordances: `--frames` with
`--report` is also how you benchmark a renderer change, and `--until-done` is
what a kiosk or an attract mode would want.

### Rules

Each of these exists because the obvious version of the test is flaky or lies.

1. **No `sleep`.** Poll for the condition with a deadline. A sleep long enough for
   a loaded CI runner is a sleep that wastes minutes on every developer's machine,
   and one tuned for a developer's machine is the flake CI reports next month.
2. **Real artifacts, temporary directories.** Every test gets its own `--out-dir`
   under `tmp_path`. Nothing reads or writes the developer's `runs/`, and nothing
   depends on a previous test having run.
3. **Skip, do not fail, when an optional half is absent.** torch and PySide6 are
   optional by design (the game must never depend on either), so a machine
   without them must report *skipped*, not red. The skip reason names the package.
4. **Determinism where it is available, bounds where it is not.** A replay played
   to its end has one right answer and the test asserts it exactly. A frame-budgeted
   live game does not, and the test asserts what must hold regardless — it advanced,
   it did not crash, it exited cleanly.
5. **Validation errors are failures.** The debug build turns on
   `VK_LAYER_KHRONOS_validation`; any `VUID` on stderr fails the test. This is what
   makes "it rendered correctly" checkable without capturing a single pixel.
6. **A training e2e trains for seconds, not minutes.** Two envs, a handful of
   steps, two updates. It is not checking that PPO learns — `test_ppo.py` does
   that — it is checking that a run produces `metrics.csv`, `evals.csv`,
   `config.json`, `model.json`, a checkpoint and a recording, and that the console
   and the game can read every one of them.

### Where they live, and how they run

```
python/tests/e2e/
  harness.py       # process drivers, deadlines, skip guards. No tests.
  test_game.py     # md_app
  test_training.py # md-train
  test_console.py  # md.ui, offscreen
  test_journey.py  # the whole thing, in the order a user meets it
```

pytest rather than Catch2, and Python rather than C++, because an e2e here
orchestrates *both* halves — a C++ binary writing a file a Python console reads,
and a Python trainer writing a recording a C++ binary plays. The existing
`core/tests/e2e/` keeps its meaning (whole-simulation invariants in-process) and
is not moved.

Marked `@pytest.mark.e2e` so `poe pytest` stays the fast inner loop and
`poe test-e2e-app` runs these. The full gate runs both.

**Headless.** Linux CI has no GPU and no display; both are solvable with what
the distribution already ships, and neither needs a new project dependency:

* **Vulkan** — `mesa-vulkan-drivers` provides lavapipe, a software device. Point
  `VK_ICD_FILENAMES` at `lvp_icd.json` and the renderer runs on the CPU.
* **A display** — `xvfb-run`, with `QT_QPA_PLATFORM=xcb`. The console needs no X
  at all: `QT_QPA_PLATFORM=offscreen` is enough for Qt Widgets.

A machine with neither skips the game tests and still runs the console and
training ones, which is the right behaviour on a build box that has no graphics
stack.

### Retro-fitting the three that already exist

Before Task 6, and independently of it — these test code that is already written,
so they are pure additions and can land at once:

- [ ] **The game boots and exits cleanly** — `--frames`, menu, `--play`,
      `--watch`; exit 0, no `VUID`, a report showing it advanced.
- [ ] **A recording plays back to the state it was recorded in** — the strongest
      single assertion in the suite, because it crosses every boundary in the
      project: a Python trainer wrote it, C++ replays it, and the score has to
      match what the trainer said. `--replay X --until-done --report`.
- [ ] **A short real training run writes every artifact** — and each is readable
      by the thing that consumes it.
- [ ] **The console attaches to that directory** and shows the run: curves
      populated, recordings listed, model panel filled, Start enabled.
- [ ] **Runtime setup installs, health-checks and is then used** — against a
      local wheel directory rather than the network, so the test is hermetic.

### What the first run of this layer found

Both of these were already there, in a plain debug run of the game with none of
the new flags, and nothing in the tree was catching either. They are baselined in
`harness.KNOWN_VALIDATION_ERRORS` so that *new* validation errors fail
immediately rather than being buried under them — which is not the same as
accepting them.

- [ ] **`VUID-vkAcquireNextImageKHR-semaphore-01779`** — the swapchain acquire
      semaphore is reused while a previous signal or wait on it is still
      pending. One semaphore where there must be one per frame in flight; it
      happens to work on this driver and is undefined behaviour. Ten occurrences
      in fifteen seconds of play, so it is every frame, not an edge case.
- [ ] **`VUID-VkShaderModuleCreateInfo-pCode-08740`** — a SPIR-V capability is
      declared whose environment requirement the instance does not satisfy: the
      shaders are compiled against a newer target than the instance asks for.

Neither belongs to this program of work — they are renderer bugs, and fixing
them is a separate change with its own tests. They are recorded here because the
e2e layer is what surfaced them, on the day it was written.

### Then: one per task, from here on

Every task below gains an **e2e step** stating the user-visible claim it must
make true. They are listed with their tasks; the rule is that a task is not done
until its e2e passes, and that the e2e is written when the task starts rather
than after it works.

---

### Task 1: Define the portable learned-policy format

**E2E:** none of its own — a file format has no shipped surface until Task 3
loads one in the game, and a round-trip through its own reader is a unit test by
definition. The claim it is really making is checked in Task 2, across a process
boundary, against a different language's implementation.

**Files:**
- Create: `python/md/policy_format.py`
- Create: `python/tests/test_policy_format.py`
- Modify: `python/md/__init__.py`
- Modify: `docs/API.md`

- [ ] **Step 1: Write failing round-trip and rejection tests**

Define tests that construct a two-layer fixture with NumPy arrays, round-trip it,
verify the checksum, and reject an unknown schema, truncated tensor, non-finite
weight, duplicate tensor name, and dimension mismatch:

```python
def test_policy_round_trip(tmp_path: Path) -> None:
    policy = fixture_policy()
    path = policy_format.write(tmp_path / "policy.mdp", policy)
    assert policy_format.read(path) == policy


@pytest.mark.parametrize("mutation", [
    unknown_schema, truncate_payload, duplicate_tensor, wrong_dimensions,
])
def test_policy_rejects_invalid_payload(tmp_path: Path, mutation: Mutation) -> None:
    path = write_mutated_fixture(tmp_path, mutation)
    with pytest.raises(policy_format.PolicyFormatError):
        policy_format.read(path)
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest python/tests/test_policy_format.py -q`
Expected: collection fails because `md.policy_format` does not exist.

- [ ] **Step 3: Implement the format**

Use a fixed magic header, little-endian numeric payloads, UTF-8 JSON manifest,
SHA-256 payload checksum, explicit tensor offsets, and `allow_pickle=False`.
Expose these stable types:

```python
@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class NativePolicy:
    schema: int
    observation_size: int
    action_count: int
    architecture: str
    tensors: tuple[Tensor, ...]
    metadata: Mapping[str, str | int | float]


def write(path: Path, policy: NativePolicy) -> Path: ...
def read(path: Path) -> NativePolicy: ...
```

- [ ] **Step 4: Run focused Python quality checks**

Run: `pytest python/tests/test_policy_format.py -q`
Expected: all tests pass.

Run: `ruff check python/md/policy_format.py python/tests/test_policy_format.py`
Expected: no findings.

- [ ] **Step 5: Document the compatibility promise**

Document magic, schema, dimensions, checksums, tensor ordering, and the rule that
the format stores data only. State that `.pt` is never the import format.

- [ ] **Step 6: Commit**

```bash
git add python/md/policy_format.py python/md/__init__.py \
  python/tests/test_policy_format.py docs/API.md
git commit -m "Define the portable learned-policy format"
```

### Task 2: Add native C++ policy loading and inference

**E2E:** *the same policy chooses the same move in both languages.* Export a
checkpoint, then have `md_agent_eval` play a fixed seed with it and the Python
evaluator play the same seed with the same file, and assert the action logs are
identical tick for tick. Parity asserted in one process proves the maths; parity
asserted across two proves the *file*, which is the thing being shipped.

**Files:**
- Create: `agent/include/md/agent/policy.hpp`
- Create: `agent/src/policy.cpp`
- Create: `agent/tests/unit/test_policy.cpp`
- Modify: `agent/CMakeLists.txt`
- Modify: `agent/tests/CMakeLists.txt`
- Modify: `python/md/export_policy.py`
- Create: `python/tests/test_export_policy.py`

- [ ] **Step 1: Add a Python exporter fixture**

Build a deterministic `PolicyValueNet`, save a checkpoint, export it, and write
fixed observations, legal-action masks, logits, values, and chosen actions to a
JSON parity fixture. Assert export refuses missing tensors and unsupported
architectures.

- [ ] **Step 2: Verify exporter tests fail**

Run: `pytest python/tests/test_export_policy.py -q`
Expected: import fails because `md.export_policy` does not exist.

- [ ] **Step 3: Implement checkpoint export**

Load checkpoints only through the existing trusted training path, map state-dict
names to `NativePolicy`, and expose:

```python
def export_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    metadata: Mapping[str, str | int | float],
) -> Path: ...
```

The exporter writes to a sibling temporary file and replaces the destination
only after validation.

- [ ] **Step 4: Add failing C++ parser and inference tests**

Test a checked-in small fixture for manifest parsing, tensor bounds, forward
output, legal-action masking, tie-breaking, corrupt checksum, and incompatible
schema. The public API is:

```cpp
class Policy {
  public:
    static Policy load(const std::filesystem::path& path);
    [[nodiscard]] Decision act(std::span<const float> observation,
                               std::span<const std::uint8_t> legal) const;
};
```

- [ ] **Step 5: Implement C++ inference**

Parse with bounds-checked offsets, reject non-finite values, perform the fixed
MLP forward pass, apply the legal mask before deterministic argmax, and keep
all policy state immutable after loading.

- [ ] **Step 6: Verify Python/C++ parity**

Run: `pytest python/tests/test_export_policy.py -q`
Expected: pass.

Run: `cmake --build --preset debug && ctest --preset debug -R policy --output-on-failure`
Expected: parser, corruption, masking, and parity tests pass.

- [ ] **Step 7: Commit**

```bash
git add agent python/md/export_policy.py python/tests/test_export_policy.py
git commit -m "Run exported learned policies in native code"
```

### Task 3: Bundle and watch the pretrained model

**E2E:** *the shipped game plays well with the shipped model, and needs no
Python to do it.* `md_app --watch-model <installed path> --frames N --report`
from the staged install tree, with Python removed from `PATH`: exit 0, no `VUID`,
and a report showing a game that advanced and scored. Repeat with
`--watch-scripted` so both bundled agents are covered.

**Files:**
- Create: `models/pretrained.mdp`
- Create: `models/pretrained.json`
- Modify: `app/CMakeLists.txt`
- Modify: `app/game_window.hpp`
- Modify: `app/game_window.cpp`
- Modify: `app/main.cpp`
- Modify: `app/renderer.cpp`
- Modify: `agent/eval_main.cpp`
- Modify: `core/tests/e2e/` golden/evaluation tests

- [ ] **Step 1: Add failing installed-resource and menu tests**

Assert the release resource exists, validates, and produces the recorded
canonical score summary. Add menu tests for scripted and learned watch choices
without changing human high-score eligibility.

- [ ] **Step 2: Verify the focused C++ tests fail**

Run: `cmake --build --preset debug && ctest --preset debug -R "policy|menu|eval" --output-on-failure`
Expected: failures for the missing bundled resource and learned watch action.

- [ ] **Step 3: Promote the release model**

Export one reviewed checkpoint, generate `pretrained.json` with provenance,
license, policy schema, simulator version, seed-set version, and canonical
metrics, then verify it with the Task 2 parity harness.

- [ ] **Step 4: Add game watch selection**

Turn **WATCH AI** into a choice between **SCRIPTED** and **PRETRAINED**. Add
`--watch-scripted` and `--watch-model <path>` for package/E2E tests. Both drivers
must use the same `Action` timing path as human input.

- [ ] **Step 5: Install the resource on every platform**

Embed it in Qt resources or install it beside the executable using one
cross-platform lookup function. Package tests must locate it from the installed
layout, not the source tree.

- [ ] **Step 6: Verify game-only behavior**

Run the Debug and Release E2E suites and a package smoke test that removes Python
from `PATH`, launches both watch modes, and confirms no training menu entry.

- [ ] **Step 7: Commit**

```bash
git add models app agent core/tests
git commit -m "Ship a pretrained agent with the standalone game"
```

### Task 4: Package and discover the existing console

**E2E:** *the two packages are two different products.* From the game-only
staging tree with Python off `PATH`: the game launches and its menu has no
TRAIN AI entry. From the full tree: the console launcher resolves, `md-console`
starts and exits cleanly under `QT_QPA_PLATFORM=offscreen`, and TRAIN AI appears.
The negative half matters more than the positive one — it is the promise that the
game stays Python-free.

**Files:**
- Modify: `debian/control`
- Modify: `debian/rules`
- Create: `debian/python3-md.install`
- Create: `debian/missile-defense-training.install`
- Create: `packaging/missile-defense-training.desktop`
- Modify: `CMakeLists.txt`
- Modify: `app/CMakeLists.txt`
- Modify: `app/game_window.cpp`
- Modify: `python/md/ui/runner.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `python/tests/test_packaging.py`

- [ ] **Step 1: Write failing packaging-contract tests**

Assert the game-only staging tree contains no Python, console, or **TRAIN AI**
launcher; the full staging tree contains `md-console`, its desktop entry, native
binding, and a discoverable console launcher.

- [ ] **Step 2: Split Debian binaries**

Keep one source package and produce `missile-defense`, `python3-md`, and
`missile-defense-training` exactly as specified. Ensure installing only
`missile-defense` succeeds with dependency checks.

- [ ] **Step 3: Add Windows/macOS console payloads**

Bundle an isolated console application and native binding. Add installer
component metadata so game-only selection omits them. On macOS, ship the console
as its own `.app` in the disk image.

- [ ] **Step 4: Make console discovery authoritative**

Implement one lookup shared in behavior by installer tests and the game.
**TRAIN AI** is added to the main menu only when that lookup resolves an
executable; activating it starts the existing console and returns to the menu.

- [ ] **Step 5: Verify package matrices in CI**

Build/stage game-only and full variants on Windows/macOS; build all Debian
binaries and run `lintian`. Smoke-launch both applications from staged paths.

- [ ] **Step 6: Commit**

```bash
git add debian packaging CMakeLists.txt app python/md/ui/runner.py \
  python/tests/test_packaging.py .github/workflows/ci.yml
git commit -m "Package the training console as an optional companion"
```

### Task 5: Implement one-click managed runtime setup

**E2E:** *a console with no torch can end up training.* Drive the setup dialog
against a **local wheel directory** rather than the network (hermetic, and fast),
then assert the console starts a real short run with the managed interpreter and
the run writes `metrics.csv`. Also: cancelling leaves nothing behind, and a
console with a broken runtime still browses and replays.

**Files:**
- Create: `python/md/runtime.py`
- Create: `python/md/ui/runtime_dialog.py`
- Create: `python/tests/test_runtime.py`
- Modify: `python/md/ui/app.py`
- Modify: `python/md/ui/runner.py`
- Modify: `python/md/paths.py`
- Modify: `python/tests/test_ui_runner.py`

- [x] **Step 1: Write state-machine tests**

Use fake probes, downloader, and subprocess runner to cover absent, recommended,
installing, ready, broken, repairable, cancelled, and removable states. Assert
that only allow-listed package indexes and compatible backend choices produce an
install plan.

- [x] **Step 2: Implement runtime planning**

Expose pure planning separately from effects:

```python
@dataclass(frozen=True)
class RuntimePlan:
    backend: str
    python: Path
    target: Path
    packages: tuple[str, ...]
    index_url: str


def recommend(system: SystemInfo, probes: Sequence[BackendProbe]) -> RuntimePlan: ...
```

- [x] **Step 3: Implement transactional installation**

Install into a new versioned directory, run an import/native-binding health
check, write a signed-by-checksum local manifest, then atomically switch the
`current` marker. Cancellation removes only the incomplete directory.

- [x] **Step 4: Add the setup dialog**

Show recommendation, backend, source, disk requirement, progress, cancel, retry,
repair, and remove. Keep logs behind disclosure. On success, return directly to
the new-run dialog.

- [x] **Step 5: Route training through the managed interpreter**

Replace `can_train()`'s current-interpreter assumption with runtime health.
Browsing and replay actions remain enabled when runtime setup is absent/broken.

- [x] **Step 6: Verify**

Run: `pytest python/tests/test_runtime.py python/tests/test_ui_runner.py -q`
Expected: all state, cancellation, recovery, and interpreter-routing tests pass.

- [x] **Step 7: Commit**

```bash
git add python/md/runtime.py python/md/ui/runtime_dialog.py python/md/ui/app.py \
  python/md/ui/runner.py python/md/paths.py python/tests
git commit -m "Install the training runtime from the console"
```

### Task 6: Build the run library and explicit model promotion

**E2E:** *promotion produces something the rest of the system can actually
use.* Train briefly, promote from the library, and assert the league directory
holds a `policy.mdp` that the **native** evaluator loads and plays a seed with —
not merely a file that the Python that wrote it can read back.

**Files:**
- Create: `python/md/library.py`
- Create: `python/md/league.py`
- Create: `python/md/ui/library.py`
- Create: `python/md/ui/league.py`
- Create: `python/tests/test_library.py`
- Create: `python/tests/test_league.py`
- Modify: `python/md/ui/app.py`
- Modify: `python/md/ui/forms.py`
- Modify: `python/md/paths.py`

- [ ] **Step 1: Write run discovery and storage tests**

Cover named runs, incomplete runs, active runs, byte totals, best evaluated
checkpoint selection, pinned recordings, and paths outside the managed root.

- [ ] **Step 2: Add the run library screen**

Make the library the console landing view. Each row shows name, state, best
score, last activity, backend, disk use, promotion, and archive status. Opening
a row displays the current dashboard unchanged.

- [ ] **Step 3: Write atomic promotion tests**

Assert promotion defaults to the highest canonical evaluated checkpoint, accepts
an explicitly selected evaluated checkpoint, validates native export/parity,
copies rather than references the run, and leaves no league entry on failure.

- [ ] **Step 4: Implement the league store**

Use `models/<stable-id>/policy.mdp` plus `model.json`. Write into a temporary
directory, fsync/close, validate, then rename. Reserve immutable IDs separately
from editable display names.

- [ ] **Step 5: Add promotion UI**

On stopped runs, **Enter Model League** shows default checkpoint, alternatives,
name, note, metrics, compatibility, and export validation. Success navigates to
the new league entry.

- [ ] **Step 6: Verify**

Run: `pytest python/tests/test_library.py python/tests/test_league.py -q`
Expected: discovery, selection, atomicity, and protection tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/md/library.py python/md/league.py python/md/ui/library.py \
  python/md/ui/league.py python/md/ui/app.py python/md/ui/forms.py \
  python/md/paths.py python/tests
git commit -m "Promote trained policies into a local Model League"
```

### Task 7: Evaluate league models and create paired matches

**E2E:** *the contest is fair and complete.* Promote two models, run a canonical
head-to-head, and assert: every contestant was given the identical seed set, the
ranking only appears once every seed is in, a cancelled tournament replaces no
ranked result, and one match manifest exists per paired seed. Fairness is a
property of the whole orchestration, so it is only checkable from outside it.

**Files:**
- Create: `python/md/tournament.py`
- Create: `python/tests/test_tournament.py`
- Modify: `agent/eval_main.cpp`
- Modify: `agent/CMakeLists.txt`
- Modify: `python/md/league.py`
- Modify: `python/md/ui/league.py`
- Modify: `python/md/ui/runner.py`

- [ ] **Step 1: Write canonical-completeness tests**

Use a small seed fixture to assert every contestant receives identical seeds,
partial evaluations never replace ranked results, metric aggregation matches the
C++ protocol, and incompatible simulator/seed versions are not co-ranked.

- [ ] **Step 2: Extend the native evaluator**

Accept scripted or native policy contestants, emit one machine-readable row per
seed plus a final aggregate, and optionally save an action log for each side.
Keep the canonical seed definition in one existing C++ source of truth.

- [ ] **Step 3: Implement tournament orchestration**

Run quick or canonical head-to-head, capture progress/cancellation, write results
atomically to league metadata, and create one match manifest per paired seed.

- [ ] **Step 4: Add Model League actions**

Provide **Evaluate**, **Head to head**, contestant selection, progress, result
cards, and history. Rank only complete canonical results; label quick matches as
unranked.

- [ ] **Step 5: Verify**

Run: `pytest python/tests/test_tournament.py python/tests/test_league.py -q`
Expected: fairness, completeness, cancellation, and ranking tests pass.

Run: `ctest --preset debug -R agent_eval --output-on-failure`
Expected: native aggregate/action-log tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/md/tournament.py python/md/league.py python/md/ui/league.py \
  python/md/ui/runner.py python/tests agent
git commit -m "Run fair local tournaments between learned policies"
```

### Task 8: Add synchronized split-screen spectator mode

**E2E:** *a match plays back as one thing.* `md_app --match <manifest>
--until-done --report`: both sides run to their own ends, the report names both
final scores and they match the tournament's record of that seed. Then the same
manifest with a seek and a restart, asserting the two sides are still on the same
tick — desynchronisation is the entire failure mode of this feature.

**Files:**
- Create: `replay/include/md/replay/match.hpp`
- Create: `replay/src/match.cpp`
- Create: `replay/tests/unit/test_match.cpp`
- Create: `app/match_window.hpp`
- Create: `app/match_window.cpp`
- Modify: `replay/CMakeLists.txt`
- Modify: `replay/tests/CMakeLists.txt`
- Modify: `app/CMakeLists.txt`
- Modify: `app/main.cpp`
- Modify: `app/renderer.cpp`
- Modify: `python/md/ui/runner.py`
- Modify: `python/md/ui/league.py`

- [ ] **Step 1: Write failing paired-player tests**

Cover manifest checksums, same-seed/config enforcement, shared tick/seek/restart,
different recording lengths, frozen completed side, and corrupt/missing side.

- [ ] **Step 2: Implement `MatchPlayer`**

Own two replay players and one shared transport:

```cpp
class MatchPlayer {
  public:
    static MatchPlayer load(const std::filesystem::path& manifest);
    void tick();
    void seek(std::uint64_t tick);
    void restart();
    [[nodiscard]] const replay::Player& left() const;
    [[nodiscard]] const replay::Player& right() const;
};
```

- [ ] **Step 3: Add the dedicated match window**

Use two viewport rectangles, one renderer/device, shared controls, per-side HUD,
round/seed context, and live delta. A finished side renders its final state while
the other advances.

- [ ] **Step 4: Add launch paths**

Support `md_app --match <manifest>` and a console **Watch match** action. Add a
game-only exhibition entry for scripted versus bundled pretrained using stored
paired recordings or freshly generated native action logs.

- [ ] **Step 5: Verify controls and rendering**

Run replay/match unit tests in Debug and Release. Add an offscreen render test
for two viewports and a scripted UI smoke test for pause, 1×/8×, seek, restart,
expand-side, unequal endings, and exit.

- [ ] **Step 6: Commit**

```bash
git add replay app python/md/ui/runner.py python/md/ui/league.py
git commit -m "Spectate model matches in synchronized split screen"
```

### Task 9: Add safe cleanup, archive, restore, export, and import

**E2E:** *an archived run comes back.* Archive a run, verify, remove the
original, restore it to a new directory, and assert the console draws the same
curves from the restored copy as it did from the original. Plus the refusals:
a traversal entry, a corrupt archive, and an existing target are each rejected
without having deleted anything.

**Files:**
- Create: `python/md/archive.py`
- Create: `python/tests/test_archive.py`
- Modify: `python/md/library.py`
- Modify: `python/md/league.py`
- Modify: `python/md/ui/library.py`
- Modify: `python/md/ui/league.py`
- Modify: `python/md/paths.py`

- [ ] **Step 1: Write archive security tests**

Cover checksums, duplicate entries, absolute paths, `..` traversal, links,
declared-size limits, corruption, incompatible schema, existing restore target,
and no source deletion before verification.

- [ ] **Step 2: Implement archive services**

Expose planning separately from mutation:

```python
@dataclass(frozen=True)
class CleanupPlan:
    keep: tuple[Path, ...]
    remove: tuple[Path, ...]
    reclaim_bytes: int


def plan_cleanup(run: Run, pins: Collection[Path]) -> CleanupPlan: ...
def create_archive(run: Run, destination: Path, selection: ArchiveSelection) -> Path: ...
def verify_archive(path: Path) -> ArchiveManifest: ...
def restore_archive(path: Path, destination: Path) -> Run: ...
```

- [ ] **Step 3: Add cleanup UI**

Show category counts and exact reclaimed bytes. Keep summary artifacts,
best/promoted checkpoint, and pinned recordings. Manual is default; retention is
an advanced opt-in setting that uses the same visible plan.

- [ ] **Step 4: Add archive/restore UI**

Create and verify first; only then enable **Remove original**. Restore to a new
directory and report compatibility without overwriting.

- [ ] **Step 5: Add model export/import**

Export `.mdp` plus metadata. Import validates data-only format, checksum,
dimensions, schema, and native inference before league registration.

- [ ] **Step 6: Verify**

Run: `pytest python/tests/test_archive.py python/tests/test_library.py python/tests/test_league.py -q`
Expected: safety, byte accounting, protection, restore, and import tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/md/archive.py python/md/library.py python/md/league.py \
  python/md/ui/library.py python/md/ui/league.py python/md/paths.py python/tests
git commit -m "Manage training artifacts without losing promoted models"
```

### Task 10: Make the UI journey the documented and packaged default

**E2E:** *the whole journey, in the order a person meets it* — install, set up
the runtime, start a named run, watch it, promote it, hold it against another
model, spectate the match, clean up, archive, restore. This is `test_journey.py`,
and it is the one test that would have caught every integration failure the
tasks above can produce individually.

**Files:**
- Modify: `README.md`
- Modify: `docs/TRAINING.md`
- Modify: `docs/PACKAGING.md`
- Modify: `docs/WINDOWS.md`
- Modify: `docs/MACOS.md`
- Modify: `docs/TESTING.md`
- Modify: `docs/ROADMAP.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `python/tests/test_tools_quickstart.py`
- Modify: `python/tests/test_packaging.py`

- [ ] **Step 1: Write installed-user acceptance scripts**

Create CI scripts that exercise game-only, full install, runtime bootstrap with a
fake local package source, named training smoke run, promotion, two-model quick
match, spectator launch, cleanup dry-run, archive, restore, and imported-model
validation.

- [ ] **Step 2: Rewrite README training entry**

Add **Train your own AI** before Development using the eight-step UI journey in
the design. Keep source/CLI commands in developer documentation, not the primary
download path.

- [ ] **Step 3: Update platform and packaging documentation**

Document installer choices, Debian binary split, managed runtime location,
supported backends, game-only contract, model/match/archive formats, and
recovery/remove actions.

- [ ] **Step 4: Run the full local gate**

Run: `poe check`
Expected: format, Python/C++ lint and types, clang-tidy, Debug/Release tests,
pytest, and coverage pass. On machines with clang-format newer than the pinned
version, use CI's clang-format-21 result as authority and do not reformat
unrelated files.

- [ ] **Step 5: Run packaging CI**

Push a branch and require green Linux gate, Debian multi-binary build/lintian,
Windows installer/ZIP matrix, macOS DMG matrix, and installed-user smoke jobs.

- [ ] **Step 6: Commit**

```bash
git add README.md docs .github/workflows python/tests
git commit -m "Document and verify the UI-only AI training journey"
```

### Task 11: Complete per-run simulation statistics

**E2E:** *the numbers reach the files people read.* A short real run's
`evals.csv` carries every new column with plausible values, and `poe eval` prints
the full block — asserted by running both, not by calling `summarize`.

**Why:** Today a run reports score, wave, cities-left and kills-per-shot only.
"Why did that policy plateau?" needs the full picture — how long episodes
survived, how much damage they took, and *how* the ammunition was spent — as
first-class, aggregated numbers rather than something inferred from a replay.
This is the data layer the console analysis in Task 12 draws on, so it lands
first.

**Files:**
- Modify: `core/include/md/sim.hpp`, `core/src/sim.cpp` (kills-per-shot histogram in `StepResult`)
- Modify: `agent/include/md/agent/eval.hpp` (extend `EpisodeResult` + `Summary`)
- Modify: `agent/src/eval.cpp` (`run_episode` counting + `summarize` aggregation)
- Modify: `bindings/vec_env.hpp`, `bindings/vec_env.cpp` (populate the new fields per env)
- Modify: `bindings/module.cpp`, `python/md/_md_native.pyi` (expose the fields)
- Modify: `python/md/eval.py` (`format_summary`), `python/md/train.py` (`evals.csv` columns)
- Modify: `agent/tests/unit/test_heuristic.cpp` (or a new `test_eval.cpp`), `python/tests/test_env.py`
- Modify: `docs/TRAINING.md`, `docs/API.md`

**The complete per-episode stat set** (sourced from the deterministic event
stream and end-state, the same way `shots`/`kills` already are):

| Stat | Source |
|---|---|
| `ticks` (survival time; ÷60 = seconds) | `sim.tick()` *(exists)* |
| `wave_reached` (last level) | `sim.wave()` *(exists)* |
| `waves_cleared` | count `WaveCleared` |
| `score` | `sim.score()` *(exists)* |
| `cities_left` / `cities_lost` | alive count / count `CityLost` |
| `bases_left` / `bases_lost` | alive count / count `BaseLost` |
| `bonus_cities` (rebuilt) | count `BonusCity` |
| `mirv_splits` | count `MirvSplit` |
| `shots` (fired) | count `Fire` *(exists)* |
| `kills` (targets destroyed) | count `ThreatKilled` *(exists)* |
| `hits` (shots that killed ≥1) = `shots − wasted` | derived |
| `wasted` (shots that killed nothing) | `StepResult.wasted` |
| **`kills_per_shot[]`** histogram (0,1,2,3,4+) | new: binned at blast expiry from `Blast.kills` |

`Summary` gains the per-episode means plus the summed `kills_per_shot`
histogram over the whole seed set. `evals.csv` widens to carry them; the eval
printout grows a full block. The histogram is the one new piece of core
instrumentation — every other field is event-counting already proven in
`run_episode`.

- [x] Steps: TDD each layer (C++ Catch2 for counting + `summarize` + histogram
      where `bin[0] == wasted`; pytest for the surfaced fields and `evals.csv`
      columns), rebuild bindings, `poe check`, verify with `poe eval` + a short
      GPU run. Commit: `Report the full statistics of a simulated run`.
- [x] **E2E** — `test_a_runs_evaluation_carries_the_full_statistics` reads a real
      run's `evals.csv` and asserts every new column is present *and plausible*
      (a column of zeroes is the failure a header check would pass on), and
      `test_the_evaluator_prints_the_full_statistics_block` drives the
      `md_agent_eval` binary — the half with no Python in it — for the printout.

### Task 12: Statistical analysis of runs in the console

**E2E:** *the analysis view renders from a real run.* Point the console at the
directory the training e2e produced and assert the distribution and the curves
are populated rather than showing their empty states, and that the compare picker
overlays a second run.

**Why:** Numbers in `evals.csv` are only as useful as what you can see in them.
The console already tails those files for the score curve; this turns it into an
instrument for *understanding* a policy — distributions, not just trend lines —
so "it stopped catching clusters" or "it dies to base loss, not city loss"
becomes visible at a glance and comparable across runs.

**Files (new UI service + view, composing existing sources):**
- Create: `python/md/ui/stats.py` (aggregate/derive distributions from run stats — no Qt)
- Create: `python/md/ui/analysis.py` (the analysis view/widgets)
- Create: `python/tests/test_ui_stats.py`
- Modify: `python/md/ui/app.py`, `python/md/ui/sources.py`, `python/md/ui/theme.py`

- [ ] **Kills-per-shot distribution** — the histogram from Task 11 as a bar
      chart (0/1/2/3/4+), the single clearest read on "is it catching clusters".
- [ ] **Survival & damage over the run** — mean ticks-survived, cities/bases lost,
      waves cleared as curves alongside the score, so a plateau's *cause* shows.
- [ ] **Per-run summary card** — the full latest-eval stat block in one panel.
- [ ] **Compare across runs** — reuse the existing vs-run picker so a distribution
      or curve overlays the run being beaten (same hue, lower opacity, as Phase 5).
- [ ] Follows the M8 design intent: one screen, live, dark, empty-states handled;
      `md.ui` still never imports `torch`. Charts via the existing `CurveView`
      escape hatch / Qt Charts. Build charts with the `dataviz` guidance.
- [ ] Steps: TDD `stats.py` under pytest (no display), then the view; `poe check`.
      Commit: `Analyse a run's statistics in the console`.

## Program completion checklist

- [ ] Game-only packages contain both bundled agents and no Python/training UI.
- [ ] Full Windows/macOS packages launch the existing console directly and from
      the game.
- [ ] Debian produces three binary packages from one source.
- [ ] One-click runtime setup, repair, and removal work without a terminal.
- [ ] Promotion exports a parity-checked native policy atomically.
- [ ] The local league ranks only complete compatible canonical evaluations.
- [ ] Synchronized split-screen matches remain aligned through every transport
      control and unequal endings.
- [ ] Cleanup protects promoted/pinned artifacts and reports exact reclaimed
      bytes.
- [ ] Archives verify before deletion and restore without unsafe extraction.
- [ ] README and platform docs describe the installed UI flow as the default.
- [x] A run reports the full per-episode statistics (survival, damage, spend,
      kills-per-shot distribution), aggregated and in `evals.csv`.
- [ ] The console can analyse those statistics — distributions and cause-of-plateau
      views — within a run and across runs.
- [ ] `poe check` and all packaging/installed-user CI jobs pass.
