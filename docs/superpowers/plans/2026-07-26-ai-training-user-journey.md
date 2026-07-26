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

Existing large UI files should compose these services; they should not absorb
their persistence, package-management, archive, or tournament logic.

### Task 1: Define the portable learned-policy format

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

**Files:**
- Create: `python/md/runtime.py`
- Create: `python/md/ui/runtime_dialog.py`
- Create: `python/tests/test_runtime.py`
- Modify: `python/md/ui/app.py`
- Modify: `python/md/ui/runner.py`
- Modify: `python/md/paths.py`
- Modify: `python/tests/test_ui_runner.py`

- [ ] **Step 1: Write state-machine tests**

Use fake probes, downloader, and subprocess runner to cover absent, recommended,
installing, ready, broken, repairable, cancelled, and removable states. Assert
that only allow-listed package indexes and compatible backend choices produce an
install plan.

- [ ] **Step 2: Implement runtime planning**

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

- [ ] **Step 3: Implement transactional installation**

Install into a new versioned directory, run an import/native-binding health
check, write a signed-by-checksum local manifest, then atomically switch the
`current` marker. Cancellation removes only the incomplete directory.

- [ ] **Step 4: Add the setup dialog**

Show recommendation, backend, source, disk requirement, progress, cancel, retry,
repair, and remove. Keep logs behind disclosure. On success, return directly to
the new-run dialog.

- [ ] **Step 5: Route training through the managed interpreter**

Replace `can_train()`'s current-interpreter assumption with runtime health.
Browsing and replay actions remain enabled when runtime setup is absent/broken.

- [ ] **Step 6: Verify**

Run: `pytest python/tests/test_runtime.py python/tests/test_ui_runner.py -q`
Expected: all state, cancellation, recovery, and interpreter-routing tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/md/runtime.py python/md/ui/runtime_dialog.py python/md/ui/app.py \
  python/md/ui/runner.py python/md/paths.py python/tests
git commit -m "Install the training runtime from the console"
```

### Task 6: Build the run library and explicit model promotion

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
- [ ] `poe check` and all packaging/installed-user CI jobs pass.
