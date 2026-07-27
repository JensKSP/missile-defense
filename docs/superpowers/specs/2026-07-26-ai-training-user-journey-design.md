# AI Training User Journey

**Date:** 2026-07-26
**Status:** Approved direction
**Scope:** Packaging, training-console journey, learned-policy deployment, local
model competition, spectator playback, and artifact management

## Purpose

Missile Defense is a learning project. A downloaded copy should therefore let a
person do more than watch the bundled AI: without using a terminal, they should
be able to train a policy, change meaningful parameters, see what it is
learning, promote a useful checkpoint, compare models fairly, and watch them
compete.

The project already has most of the training console. This design extends and
packages that existing PySide6 application; it does not replace it or move
training into the C++ game process.

## Product principles

1. **One discoverable flow.** The game exposes **TRAIN AI** when the training
   console is installed, and the console also has its own desktop entry.
2. **No command line in the user journey.** Building from source may still use
   developer commands, but installed users train, evaluate, watch, export,
   archive, restore, and clean up through UI.
3. **The game remains a small, complete game.** A game-only installation
   includes the scripted agent and one pretrained learned policy, but no Python
   or training affordances.
4. **Fair comparisons.** Models compete by playing the same canonical seeds
   under the same simulation, timing, action-mask, and scoring rules.
5. **Spectating explains the numbers.** Every tournament result can be watched
   as synchronized paired replays.
6. **User work is durable.** Promoted models are protected from run cleanup;
   archives are self-describing, checksummed, and restorable.
7. **Heavy dependencies are explicit.** The console remains useful without
   PyTorch. Training support is installed once, on demand, into an app-managed
   environment.

## What exists today

The existing `python/md/ui/` console already provides:

- start, pause, resume, graceful stop, and reset;
- a parameter dialog with four primary settings and progressive disclosure for
  advanced settings;
- live score, return, entropy, and value-loss plots;
- run-to-run curve overlays;
- model and checkpoint inspection;
- replay browsing, playback through the C++ game, and deletion;
- CPU, memory, and optional GPU monitoring;
- attachment to runs started elsewhere.

The Python wheel already exposes `md-console` and `md-train` with optional
`console` and `train` extras. Downloadable game artifacts currently contain only
the C++ game. The game can watch the scripted heuristic and play recorded action
logs, but it does not yet load a learned policy directly. There is no model
registry, local AI league, checkpoint-promotion flow, paired spectator mode, or
run archive manager.

## Packaging contract

### Game-only installation

The game-only product is complete without Python:

- C++ game and desktop/menu integration;
- scripted algorithmic agent;
- one bundled pretrained learned model;
- **WATCH AI** choices for the scripted and pretrained agents;
- replay browser and spectator controls;
- human arcade high scores;
- no **TRAIN AI** entry and no training-related settings.

The absence of training is represented by absence, not by disabled controls that
send the user to a shell.

### Windows and macOS

The normal full installer/image contains:

- the game-only payload;
- the existing training console as a separately launchable application;
- the Python package, native environment binding, runtime bootstrapper, and
  console dependencies;
- a **TRAIN AI** launcher in the game;
- a standalone **Missile Defense Training Console** desktop/application entry.

PyTorch is not embedded in the base package. The first action that requires
training opens a one-click runtime setup described below.

Installers may offer a game-only component selection. If that is selected, the
installed result obeys the game-only contract exactly.

### Debian

One Debian source package produces separate binary packages:

| Binary package | Responsibility |
|---|---|
| `missile-defense` | Game, scripted agent, pretrained model, game desktop entry |
| `python3-md` | Python environment API and compiled native binding |
| `missile-defense-training` | Trainer, existing console, runtime bootstrapper, console desktop entry |

`missile-defense-training` depends on `python3-md` and the game, so replay and
spectator launching always work. The game package does not depend on Python.
The existing `debian/` build is the Debian authority; CPack's `.deb` remains a
developer convenience and does not need to express the multi-binary split.

## First-run training runtime

The packaged console starts without importing PyTorch. It detects:

- operating system and CPU architecture;
- supported GPU vendor/backend;
- available disk space;
- whether a managed training runtime already exists and is healthy.

When training is first requested, a setup screen recommends one supported
runtime:

- CPU when no supported accelerator is detected;
- the normal macOS PyTorch build with Metal support on Apple silicon;
- a compatible vendor build on supported Windows/Linux hardware.

The screen shows the selected backend, download size before confirmation, target
directory, and expected capability. **Install training runtime** performs the
download and installation into an application-owned environment without showing
a terminal. Progress, cancellation, retry, repair, and removal are UI actions.
Downloaded packages are obtained from configured official package sources and
verified using the package manager's integrity metadata.

A failed or cancelled setup does not damage the console. Run browsing, charts,
league viewing, archives, and replay playback remain available. Diagnostic text
is copyable and names the failed stage; a log is available behind a disclosure
control.

Advanced users may choose among compatible backends, but the default path is one
recommended button rather than a package/version form.

## Learned-policy deployment

The game-only package cannot depend on Python or PyTorch, so promoted policies
need a native inference artifact.

### Chosen approach

Export the project's fixed policy architecture into a small, versioned,
self-describing model file containing:

- format and policy-schema versions;
- observation and action dimensions;
- ordered tensor names, shapes, dtypes, and weights;
- normalization and action-mask metadata;
- training/evaluation provenance;
- payload checksum.

The C++ side implements inference for the supported architecture using the same
observation construction and action mask as training. This is preferred over:

- bundling Python/PyTorch with the game, which violates the game-only contract;
- adding ONNX Runtime for one known multilayer perceptron, which adds a large
  cross-platform dependency and another packaging surface.

Export runs a parity check on fixed observations: Python and C++ must choose the
same masked action and produce numerically bounded logits/value outputs before a
model can be promoted. Unsupported future architectures fail with an explicit
compatibility message rather than being guessed.

The release process promotes one validated model as the bundled pretrained
policy. Its license/provenance and canonical evaluation summary ship beside it.

## Training journey

### 1. Open

The user launches the existing console directly or selects **TRAIN AI** in the
game. The console opens the run library, not an empty plotting dashboard.

### 2. Create a run

**New training run** asks for:

- model/run name;
- environments, rollout steps, updates, and learning rate;
- recommended hardware/runtime;
- estimated working disk use;
- optional resume checkpoint.

The current advanced-parameter disclosure remains. Tooltips continue to explain
why a parameter matters. The command preview may remain as a reproducibility
detail, but no user action depends on copying or executing it.

### 3. Learn

The existing one-button state machine remains:

**Start → Pause → Resume**, with graceful **Stop** and non-destructive **Reset**.

Live curves, system meters, log, model card, checkpoints, and recordings update
from run artifacts exactly as they do today. The UI adds:

- training-runtime/backend identity;
- current and peak disk use;
- the best canonical evaluation so far;
- a clear **Watch latest** action when a recording appears.

### 4. Promote

Every evaluation is associated with its checkpoint. The console tracks the best
canonical score automatically. When a run stops, **Enter Model League** offers:

- the best evaluated checkpoint by default;
- any other evaluated checkpoint;
- an editable model name and optional note;
- the parity/export validation result;
- the canonical score card.

Promotion is explicit. Merely finishing a run does not populate the league.

The promoted model is copied into the league store as an immutable native
inference artifact with its metadata. It no longer depends on the continued
existence of the run directory.

## Local Model League

AI results are separate from the human arcade high-score table.

The local league initially contains:

- the scripted algorithmic agent;
- the bundled pretrained model;
- each user-promoted model.

The league ranks the latest valid canonical tournament result and shows:

- mean score;
- mean wave reached;
- mean surviving cities;
- kills per interceptor;
- evaluation seed-set and simulation versions;
- model origin, training parameters, and evaluation date.

No online account, upload, or community leaderboard is included in this phase.
Incompatible results remain visible but are not ranked against a different
simulation or seed-set version.

Users can rename their own model display name, add notes, reevaluate it, export
it, archive its source run, or explicitly remove it. The scripted and bundled
entries cannot be deleted.

## Tournament and spectator experience

### Competition semantics

Models do not share one battlefield. For each canonical seed, every contestant
plays an independent simulation with identical configuration. Results are
aggregated over the same seed set. This makes the tournament deterministic,
repeatable, and fair.

The first release supports:

- quick head-to-head between any two league entries;
- full canonical head-to-head for ranked results;
- optional round-robin across selected local models after head-to-head is
  stable.

Evaluation happens through shared C++ scoring code using exported native
policies, so the model being ranked is the model the game can actually show.

### Paired recordings

A match produces two action logs plus a small match manifest containing model
identities, seed, configuration, result, and checksums. Replays remain the source
of truth; the viewer never reruns policy inference while spectating.

### Synchronized split-screen

The approved spectator layout is two battlefields side by side:

- same seed and simulation clock;
- shared pause, resume, speed, restart, and seek controls;
- model name, live score, wave, cities, and ammunition on each side;
- persistent score delta and round/tournament context;
- an ended side remains frozen at game over while the other side continues;
- selecting either side can expand it temporarily without losing synchronization.

This belongs in the C++ game as a dedicated match-replay mode. The console starts
the match and opens the spectator window; game-only installations can use the
same mode for the scripted agent versus the bundled pretrained model.

## Storage, cleanup, archive, and export

The run library shows total and per-run disk usage, last activity, promotion
state, and archive state.

### Cleanup

Cleanup is manual by default. An advanced opt-in retention policy may prune
intermediate artifacts after showing what it will keep.

The safe cleanup preset keeps:

- configuration, model card, metrics, evaluations, and useful summary metadata;
- the promoted/best checkpoint when present;
- user-pinned recordings;
- enough log tail to diagnose the run.

It offers removal of:

- superseded intermediate checkpoints;
- unpinned periodic recordings;
- bulky transient logs and caches.

Before confirmation, the dialog lists categories, counts, and reclaimed bytes.
League artifacts are outside the run directory and are never cleanup targets.

### Archive

**Archive run** creates a standard compressed container with a project-specific
extension. It contains:

- a versioned manifest;
- all selected run artifacts and their relative paths;
- checksums, uncompressed sizes, and creation metadata;
- compatibility versions for the simulator, observation, policy, and replay
  formats.

The console verifies the archive before offering **Archive and remove original**.
Restore extracts into a new run directory, verifies every entry, prevents path
traversal, and never overwrites an existing run silently.

### Export

**Export model** writes the immutable native policy and metadata to a location
chosen by the user. **Import model** validates format, checksum, compatibility,
and inference shape before offering league registration. Importing does not
execute code from the archive or model.

## Data locations

Existing per-user data-location rules remain authoritative. Within the
`MissileDefense` data directory:

```text
runs/                 mutable training runs
models/               protected promoted native policies
matches/              paired tournament recordings and manifests
archives/             optional default archive destination
runtime/              app-managed training environment and package cache
```

The console may let users choose an external archive/export location. Runtime
files are replaceable; runs and promoted models are user data.

## Failure and recovery rules

- Runtime setup failures are retryable and never prevent non-training use.
- A trainer crash leaves the last complete checkpoint and artifacts readable.
- Promotion is atomic: export to a temporary file, validate, then rename into
  the league store.
- Evaluation writes a result only after all required seeds complete.
- Match manifests reference checksums, so mismatched or missing paired replays
  are reported rather than played out of sync.
- Cleanup and archive dialogs identify promoted/pinned artifacts.
- Archive creation is verified before source deletion is offered.
- Model and replay compatibility failures explain which version differs.

## Security boundaries

- Model and archive formats contain data, never executable Python or pickle
  objects accepted from untrusted sources.
- A `.pt` checkpoint is only handled inside the managed training environment and
  is not a portable import format.
- Archive extraction rejects absolute paths, parent traversal, links, duplicate
  entries, and unreasonable declared sizes.
- Runtime package sources are allow-listed and visible in the setup details.
- The game never imports Python or loads arbitrary shared libraries for policy
  inference.

## Documentation changes

The README gains a first-class **Train your own AI** section before development
instructions. It explains the installed UI journey:

1. install the full package;
2. open **TRAIN AI**;
3. complete one-click runtime setup;
4. start a run and change the four meaningful parameters;
5. watch curves and replays;
6. promote the best checkpoint;
7. enter a head-to-head tournament and spectate it;
8. clean up or archive the run.

CLI instructions remain in `docs/TRAINING.md` for developers and automation, but
they are no longer the primary user path. Packaging documentation states the
game-only/full-install matrix and Debian binary split exactly.

## Verification and acceptance

The feature is complete only when these user-level scenarios pass:

1. A clean game-only install launches, watches both bundled agents, and exposes
   no training feature or Python dependency.
2. A full Windows/macOS install exposes the console directly and through
   **TRAIN AI**.
3. A Debian build produces the three binary packages from one source and the
   game package works alone.
4. First-run setup installs a recommended runtime with one click; cancellation,
   failure, repair, and removal leave the console usable.
5. A user creates a named run, edits primary parameters, watches live progress,
   pauses/resumes, and opens a generated replay without a terminal.
6. The best checkpoint is suggested, parity-validated, promoted, and protected
   from run cleanup.
7. Two league models are evaluated on identical seeds and ranked from complete
   canonical results.
8. Their paired action logs play in synchronized split-screen through pause,
   speed, seek, restart, unequal episode endings, and completion.
9. Cleanup reports and reclaims the promised bytes without harming pinned,
   promoted, or summary artifacts.
10. A full archive verifies, restores into a new run, and reproduces its model,
    metrics, evaluations, and selected recordings.
11. Imported models and archives reject corruption, incompatibility, unsafe
    paths, and executable payloads with actionable messages.

## Delivery sequence

The work should land in dependency order:

1. native policy format, exporter, parity tests, and C++ inference;
2. bundled pretrained policy and game-only watch flow;
3. packaging split and console discovery/launcher;
4. one-click managed runtime setup;
5. run library, storage accounting, and explicit checkpoint promotion;
6. local Model League and canonical evaluator;
7. paired match artifacts and synchronized split-screen spectator mode;
8. cleanup, archive, restore, model export/import;
9. README, platform documentation, packaging CI, and end-to-end installed-user
   tests throughout the phases above.

The detailed implementation record is stored beside this specification in
`docs/superpowers/plans/2026-07-26-ai-training-user-journey.md`. The canonical
remaining scope, educational journey, and release gates are in
`docs/superpowers/plans/2026-07-27-v1.0-completion-plan.md`.
