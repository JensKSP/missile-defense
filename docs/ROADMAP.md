# Roadmap

Milestones, ordered so that **a human can play first**. Each builds on the pure,
headless, deterministic `md::core` simulation.

> **Acceptance gate:** a milestone is *passed* only when **the human confirms it** by
> playing/using it. "Implementation complete + gate green" is not "passed" — it means
> *ready for human sign-off*.

## M1 — Human can play ✅ *(passed — accepted by Jens, 2026-07-24)*

A person plays a fresh game of Missile Command in the Vulkan UI: waves of descending
threats, three batteries with limited ammo, interceptors with travel time + expanding
blast, six cities to defend, score, and win/lose.

Path:
1. ✅ Core primitives — `Vec2`, `Pcg32` RNG — + TDD / zero-warning quality harness.
2. ✅ Entity state model — POD, fixed-capacity (`Config`, enums, entities, `Action`).
3. ✅ `Sim` core loop — `reset(seed)` / `step(action)`, waves & spawning, motion,
   collisions (blast ↔ threat, threat ↔ city), scoring, termination. Unit + e2e +
   **determinism** tests (Debug == Release, `-ffp-contract=off`). **The game is fully
   playable headless.** *(MIRV / smart-bomb threat variants deferred to a follow-up.)*
4. ✅ Vulkan / Qt app — `QVulkanWindow` bring-up → instanced-quad/circle renderer under
   an orthographic world→screen projection → fixed-timestep sim loop → Qt mouse input
   mapped to the shared `Action` (crosshair aim, click to fire) → HUD (score / wave /
   ammo) via a quad-based pixel font.

→ **Passed** — played and accepted by the human on 2026-07-24.

## M2 — Polished game *(in progress)*

A complete, fun arcade game. **Acceptance gate: the human plays it and confirms.**

**Presentation & shell**
- ✅ Menu (start / help / highscores / exit)
- ✅ Restart — new game from the menu
- ✅ Pause → menu (RESUME / NEW GAME), Help screen
- ✅ Rocket trails — missile lines from launch point to a glowing head
- ✅ Blast glow + dangerous fireball explosions (ground impacts + interceptor blasts)
- ✅ Game-over screen (dim overlay, spaced text)
- ✅ **Sound** — procedural retro SFX via vendored miniaudio, driven by the
  deterministic `md::core` event stream (which also gives AI observation parity)

**Gameplay completeness** (faithful mechanics)
- ✅ **Destructible bases** — threats target/destroy batteries (render as rubble)
- ✅ **MIRV threats** — warheads that split into several mid-descent (purple)
- ✅ **Smart-bomb threats** — decoys that steer to dodge blasts (green)
- ✅ **Bonus cities** — rebuild a destroyed city at score thresholds

**Also requested**
- ✅ **Highscores** — persistent top-10 table (pre-seeded with defaults) + arcade
  initials entry on a qualifying score
- ✅ **Options** — menu screen with Audio on/off (mutes the SFX) and Music on/off
  (looping procedural FM-synth soundtrack)
- ✅ **Polish** — animated starfield, missile-shaped threats with a distinct look
  per type (ICBM / MIRV / smart bomb), thunder on city loss, WWII E57 siren on
  each wave, crosshair reticle

## M3 — Record & replay in the UI ✅ *(implemented — ready for sign-off)*

Record every run as `(seed, action-log)`; load a run and replay it with pause / scrub /
variable speed; **take over** from any point and continue playing.

Brought forward out of order, because M6 needs it: watching the policy play is how you
see it learn, and a scalar reward curve does not show you *why* a run went badly.

- ✅ **Recording** — `md::replay::Recording` is `(seed, config, obs-spec, frame-skip,
  action indices)`. Only the discrete action *index* is stored: `decode_action` is a pure
  function of state, so seed + indices replay bit-identically at four bytes per agent
  step. A full episode is ~80 kB, which is what makes it cheap to drop one on disk every
  few training updates.
- ✅ **From training** — `VecEnv.record(i)` logs one environment of the batch;
  `save_recording(i, path, update=…, label=…)` writes an episode once it completes.
  Episodes are only handed over whole.
- ✅ **Playback** — `md_app --replay <file>` plays one, with the label and a progress bar
  on the HUD, `[` / `]` speed control, and `T` to take over from where it has reached.
- ✅ **Scrub** — arrow keys seek ±5 s, `R` restarts. There is no state to interpolate
  to, so seeking forward plays forward; seeking back rewinds to the nearest snapshot
  (every 600 ticks — `Sim` state is a `memcpy`) and plays from there, which bounds a
  backwards seek instead of replaying from tick zero.
- ✅ **Browser** — a **REPLAYS** menu entry lists what is in `runs/`, newest first,
  rescanned on each visit so a training run in progress shows its latest episodes.
- ⬜ **Recording human play** — the recorder logs discrete action indices, which is what
  a policy emits; a human's mouse aim is a continuous `Action` and does not fit that
  format. Deferred until the player model is calibrated (see the note in `config.hpp`).

> **Recordings are build-local artifacts, not an archive format.** The file embeds the
> simulation `Config` it was recorded with — not the app's audio/fullscreen settings,
> which cannot affect the simulation — so a recording *does* survive a change to a
> `Config` default: it replays with its own values. The stored struct size is checked on
> load to catch the layout changing (a field added or reordered), where the raw bytes
> would otherwise be misread as garbage. What no guard here can catch is a change to the
> simulation *logic* with `Config` untouched; that is the golden checksum's job, and it
> is the real reason these files are tied to the build that wrote them.

## M4 — Algorithmic reference AI ✅ *(implemented — ready for sign-off)*

A hand-coded Missile Command agent — no learning — playing headless through the shared
`Action` interface, as the apples-to-apples yardstick for the ML agent. Lives in
`agent/` (`md::agent`), a sibling of the core: the simulation never depends on an agent.
Deterministic, allocation-free, unit-tested. Run it with `poe eval`.

**Fairness.** The baseline is held to *the same information as the neural policy*: it
reads only what `md::encode` exposes and never touches the simulation's internal
bookkeeping — in particular not `Threat::target_index`, which would reveal for free which
city a warhead is aimed at. It infers that from the trajectory, exactly as a policy must.
It is equally bound by the player model (crosshair travel, trigger interval), because
those live in `Sim::step`, not in the driver.

Heuristics, per tick: discard threats already doomed by an in-flight interceptor or a
live blast; score every (threat, battery) pair by what the threat would destroy, how many
*other* threats one blast would catch, how soon it lands, and how long the shot takes to
set up (cursor travel included); engage the best pair. Scoring is a pure function of
observable state, so target commitment falls out for free — whatever the crosshair is
already near is cheapest to shoot, which stops it oscillating between rivals.

### Baseline results — the numbers to beat

32 canonical seeds (`md::agent::default_seeds`), default `Config`:

| Metric | Value |
|---|---|
| Mean score | **18,036** (range 15,275 – 20,610) |
| Mean wave reached | **16.0** |
| Mean cities surviving | **0.00** of 6 |
| Kills per interceptor | **1.10** |
| Episodes surviving the cap | 0 / 32 |

**This settles the question the design turned on.** A perfect-marksmanship agent — one
that solves the lead-intercept exactly and never misses — still loses *every* game, with
every city gone, around wave 16. The difficulty is not aiming; it is allocation under an
ammunition budget that goes negative as waves grow. Exact velocity information does not
trivialise the game, which is why the observation gives raw state rather than hiding it.

Kills-per-interceptor of 1.10 is where the visible headroom is: blasts are catching a
cluster only occasionally. Waiting for MIRV spreads to converge, and spending the
end-of-wave ammo bonus wisely, are exactly the judgement calls a learned policy can beat a
greedy one at.

*(Deliberately reasonable, not optimal — a yardstick, not a champion. It is never tuned
against the evaluation seeds.)*

## M5 — ML infrastructure

nanobind bindings; Gymnasium environment; `VecSim` (N parallel sims, thread pool, GIL
released, zero-copy batched observations); reward design. Reuses the **M4 evaluation
protocol** so learned and scripted agents are compared identically.

## M6 — Train

Custom PPO (PyTorch) with a curriculum; **beat the M4 algorithmic baseline**.

## M7 — Watch the AI / takeover

Export the policy → in-process C++ inference; live spectator mode; human takeover from any
point in an AI-played game (scripted or learned).

## M8 — Training console (PySide6)

A desktop UI for running and understanding training: start / pause / stop / reset a
run, tweak the parameters, watch the curves, browse and play back recordings, inspect
the model, and launch the game — without leaving the window.

**Why it earns a milestone.** The project already bets on interpretability: recordings
exist because a return curve cannot tell you the policy has learned to ignore MIRVs.
This puts the curve and the episode side by side, which is the whole loop in one place.

### Design intent — modern, simple, elegant

This is a constraint, not a mood — and *simple* here means **the least complexity the
job actually needs, not minimalism as a style**. This is an engineering instrument:
density earns its place when it is telling you something. The thing to cut is
ornament and indirection, never information you would otherwise go hunting for.

Concretely:

**One screen, not a tab bar.** The run is the subject. A tabbed console makes you
hunt for the thing you came to look at. The curve gets the space; everything else is
a strip around it.

**Control is small so monitoring can be large.** Three affordances, not a dashboard
of them: one primary button that changes meaning (Start → Pause → Resume), Stop, and
Reset tucked away because it is destructive. If control needs more than a slim bar,
the design is wrong.

**Progressive disclosure over a wall of fields.** There are ~20 hyperparameters and
their defaults are good and reasoned. Show the four that change a run's character
(envs, steps, updates, learning rate); everything else lives behind *Advanced*.
Each field carries the reasoning already written beside it in code as its tooltip —
the UI should teach, since that is what this project is for.

**The baseline is the hero.** 18,036 drawn as a horizontal line the return curve is
climbing toward. That single element is the difference between "a number going up"
and "am I winning yet". Everything else on the plot is secondary.

**It should look like it belongs to the game.** The game already has a palette —
deep navy field, orange fireballs, cool blue cities, a pixel font. The console
should feel like the same product seen from the other side, not a generic tool that
happens to point at it. Dark by default.

**Live, quietly.** Values update smoothly and continuously; nothing flickers, nothing
needs a refresh button, and a long run left open overnight should still be readable
at a glance from across the room.

**Empty states are part of the design.** No run yet, no recordings yet, no GPU probe
installed — each says what it is and what to do, rather than showing a blank panel or
a zeroed meter.

### The decision that shapes everything: training stays out of process

The UI **must not** train inside its own event loop. Training saturates the CPU for
hours; in a Qt event loop that means a frozen window, and a UI crash would take the run
with it.

Instead the run is a subprocess and the existing artifacts *are* the interface:

| Artifact | The UI's use |
|---|---|
| `runs/metrics.csv` | tail → live curves (return, entropy, value, clip fraction) |
| `runs/update-*.mdr` | list → double-click launches `md_app --replay` |
| `runs/checkpoints/*.pt` | list → `--load` to score, or `--resume` to continue |
| stdout | streamed into a log pane |

This buys more than responsiveness: the UI can **attach to a run started from a
terminal**, a crashed UI leaves training untouched, and the whole thing works against a
synced directory from the Debian box.

### Control: a file the loop polls, not a signal

Start is just spawning the process. The other three need the training loop to
cooperate, and the simplest mechanism that works identically on Windows and Linux is a
small control file checked **once per update** (an update is seconds, so that
granularity is plenty):

* **Pause** — the loop blocks between updates rather than being SIGSTOP'd, so it stays
  responsive, keeps its allocations, and resumes exactly where it was. An OS-level
  suspend would freeze it mid-update in an unknown state.
* **Stop** — *graceful*: finish the current update, write a final checkpoint, flush
  metrics, exit. Killing the process instead would throw away everything since the last
  checkpoint.
* **Reset** — start a fresh run directory. Destructive (it abandons checkpoints), so it
  must confirm, and should default to a new `--out-dir` rather than deleting.

Because it is a file, `touch runs/STOP` from a shell does the same thing — the UI is a
convenience over the mechanism, never the only way to reach it.

### What it shows

Capabilities, **not a tab layout** — see the design intent above. Most of this lives
on one screen; only the parameter form and the recording browser justify their own
surface.

1. **Run** — parameter form (the `TrainConfig` / `PPOConfig` fields, each with the
   reasoning already written beside them in code as tooltips), start/pause/stop/reset,
   live status.
2. **Curves** — return, entropy, value loss, clip fraction, from `metrics.csv`. The
   **baseline drawn as a horizontal line**, because "beat 18,036" is the actual goal and
   a curve without it is just a number going up.
3. **Model** — parameter count, layer shapes, observation/action sizes, the iteration a
   checkpoint came from, and its eval summary against the baseline.
4. **Recordings** — browse `runs/`, play in the app, delete. Newest first.
5. **System** — CPU and RAM via `psutil`; GPU through a **pluggable probe**.

   There is no cross-vendor Python API for GPU telemetry, so rather than pick one
   vendor or skip the feature, the panel talks to a small protocol and discovers
   whichever backend is installed:

   ```python
   class GpuProbe(Protocol):
       name: str                       # "NVIDIA RTX 4090", "AMD RX 7900 XTX"
       def sample(self) -> GpuSample: ...   # utilisation %, memory used/total, temp
   ```

   | Backend | Package | Covers |
   |---|---|---|
   | NVIDIA | `pynvml` | CUDA cards |
   | AMD | `amdsmi` (ROCm's own Python bindings; `pyrsmi` as fallback) | ROCm cards |
   | none | — | panel hides the GPU row |

   All optional: each backend is one module behind a soft import, so a missing
   package is a normal state and not an error. This matters because the interesting
   runs will eventually happen on a machine with a real accelerator — on a laptop
   training on CPU the meter reads idle and tells you nothing, but that is an
   argument for it degrading quietly, not for leaving the hook out. Adding a vendor
   later should be one file, not a refactor of the panel.

### Notes

* **PySide6**, installed into the *native* interpreter that already has torch and
  `_md_native`. Its wheels are MSVC-built, so this needs no MSYS2 involvement and no
  second toolchain — and it is consistent with the game already being Qt 6.
* Lives in `python/md/ui/`, launched by `poe ui`. It imports `md.env` only for shapes;
  it never steps a simulation itself.
* Depends on M6 being genuinely runnable — a console for a run that does not learn is a
  pretty window. Sequence it after the first real training run, not before.

### Execution plan

Sequenced so that **every phase ends with something usable**. Nothing here is a
big-bang: if work stops after any phase, what exists still earns its place.

```
python/md/ui/
  __init__.py
  app.py         # QApplication bootstrap, window, `poe ui` entry point
  runner.py      # spawn/attach to training; the control file; stdout stream
  sources.py     # tail metrics.csv, list recordings + checkpoints (no Qt)
  charts.py      # the curve widget, baseline line included
  system.py      # psutil sampler + the GpuProbe protocol
  probes/        # nvidia.py, amd.py — soft imports, one file per vendor
  theme.py       # palette lifted from the game, dark by default
```

`sources.py` and `runner.py` hold **no Qt**, so both are unit-testable under
pytest without a display — which is what keeps this from being a milestone with no
tests.

**Phase 1 — Monitor (read-only).** Tail `metrics.csv`, draw return / entropy /
value with the baseline line, list recordings newest-first, double-click to launch
`md_app --replay`. Attaches to a run started from a terminal; cannot start one.
*Usable on its own: this is the panel you leave open during a run.*

**Phase 2 — Control.** The control file protocol (`runs/control.json`), polled once
per update by the training loop, plus start / pause / resume / graceful stop in the
UI. **The training-loop half ships first and independently** — `touch runs/STOP`
must work before any button does, because that is what keeps the UI a convenience
over the mechanism rather than the only way in.

**Phase 3 — Parameters.** The four headline fields plus *Advanced*, tooltips
sourced from the dataclass docstrings so they cannot drift from the code. Launches
a configured run; writes the config next to the checkpoints so a run records what
produced it.

**Phase 4 — Model & system.** Parameter count, layer shapes, checkpoint iteration,
last eval against the baseline. CPU/RAM row; GPU row when a probe imports.

**Phase 5 — Compare.** Two checkpoints side by side, their eval summaries and
curves overlaid. The point at which the console starts answering *"did that change
help?"* rather than only *"what is happening now?"*.

### Risks worth naming up front

* **Charting.** Qt has no built-in plot. `pyqtgraph` is the pragmatic pick (fast,
  live-updating, MIT); QtCharts is heavier and its licence is murkier for PySide.
  Decide in Phase 1, because everything visual rests on it.
* **Tailing a file being written.** Read incrementally from a held offset rather
  than re-reading the CSV each tick, or a long run degrades as the file grows.
* **Scope creep into a trainer.** The UI must never step a simulation or hold model
  state. If a feature needs that, it belongs in `md.train`, exposed as an artifact.

### Before this: onboarding

Independent of the console, and higher leverage for reach. The README is currently a
reference, so a newcomer meets a dependency matrix before a missile. Add a **quick
start** — clone, build, play, watch the AI — with the requirement tables demoted to
reference below it. Small, self-contained, and the thing most likely to decide
whether someone runs this at all.
