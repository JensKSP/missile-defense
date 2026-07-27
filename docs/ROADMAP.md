# Roadmap

Milestones, ordered so that **a human can play first**. Each builds on the pure,
headless, deterministic `md::core` simulation.

> **1.0 completion:** The canonical remaining scope and release gates are in
> [`docs/superpowers/plans/2026-07-27-v1.0-completion-plan.md`](superpowers/plans/2026-07-27-v1.0-completion-plan.md).
> The product goal is a fun, self-directed ML learning experience, not merely a
> game plus training controls.
>
> **Acceptance gate:** a milestone is *passed* only when **the human confirms it** by
> playing/using it. "Implementation complete + gate green" is not "passed" — it means
> *ready for human sign-off*.

## M1 — Human can play ✅ *(passed — accepted by Jens, 2026-07-24)*

A person plays a fresh game of Missile Defense in the Vulkan UI: waves of descending
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

A hand-coded Missile Defense agent — no learning — playing headless through the shared
`Action` interface, as the apples-to-apples yardstick for the ML agent. Lives in
`agent/` (`md::agent`), a sibling of the core: the simulation never depends on an agent.
Deterministic, allocation-free, unit-tested. Run it with `poe eval`.

**Fairness.** The baseline is held to *the same information as the neural policy*: it
reads only what `md::encode` exposes and never touches the simulation's internal
bookkeeping — in particular not `Threat::target_index`, which would reveal for free which
city a warhead is aimed at. It infers that from the trajectory, exactly as a policy must.
Blast lifetime phase is included beside radius, so the heuristic no longer knows when a
full-size blast expires unless the model can know it too. It is equally bound by the
player model (15 Hz decisions, crosshair travel, trigger interval), because those limits
live in `Sim::step`, not in the driver.

At each sampled decision, the heuristic discards threats already doomed by an in-flight
interceptor or a live blast; scores every (threat, battery) pair by what the threat would
destroy, how many *other* threats one blast would catch, how soon it lands, and how long
the shot takes to set up (cursor travel included); then engages the best pair. Scoring is
a pure function of observable state, so target commitment falls out for free — whatever
the crosshair is already near is cheapest to shoot, which stops it oscillating between
rivals.

### Reference results — the learning ladder

32 held-out canonical seeds (deterministic stream offset 32), default `Config`,
15 Hz decisions, and an exact 120,000-tick cap:

| Metric | Value |
|---|---|
| Mean score | **98,542.34375** (range 83,525–108,920) |
| Mean wave reached | **15.75** |
| Mean cities surviving | **0.00** of 6 |
| Kills per interceptor | **1.0853** |
| Episodes surviving the cap | 0 / 32 |

#### The skill ladder, and what each behaviour is worth

`md_agent_eval --skill low|medium|high`, same protocol. These are progressive
challenges: LOW is the first target, MEDIUM is the normal trained-policy
benchmark, and HIGH is the expert challenge.

| Skill | Mean score | Wave | Kills/shot | Wasted |
|---|---|---|---|---|
| `low` | 19,585.5 | 8.38 | 0.50 | 56% |
| `medium` | 63,295.6 | 13.16 | 0.75 | 33% |
| `high` | **98,542.3** | 15.75 | 1.09 | 4% |

The dial is `Params::coverage_horizon` — how many seconds ahead the agent
remembers the shots it has already fired — calibrated by sweeping it against
this block (`medium` = 0.36 s). Two results worth keeping:

* **Ammunition memory is worth ~78,000 points; MIRV-cluster planning is worth
  ~1,500.** Switching off `cluster_bonus`, which deliberately waits for spreads
  to converge, barely moves the score. The whole baseline is one idea: *do not
  shoot what is already dead.*
* **The response is a cliff, not a slope.** 0.30 s scores ~34k and 0.40 s ~85k,
  because that is where the dial crosses a typical interceptor's flight time.
  Re-sweep it if `interceptor_speed` or the world box changes.

#### The learned policy, on the same block

One relational run (`--architecture entity`, 1,024 envs × 256 steps, 1,000
updates), checkpoint selected on the validation split and scored **once** here:

| Metric | Scripted | Learned |
|---|---|---|
| Mean score | **98,542.3** | **90,865.9** (range 76,550–101,945) |
| Mean wave reached | 15.75 | 15.38 |
| Kills per interceptor | 1.09 | 0.86 |
| Wasted shots | 4% | 23% |

It matches the depth and loses the 7,676 entirely on marksmanship. It **easily
beats MEDIUM** and approaches HIGH: it rediscovered much of the ammunition
discipline nobody told it about, but not all of it. That is the 1.0 progression
shown to learners; HIGH remains the challenge rather than a milestone the
bundled model is falsely claimed to have beaten. `models/pretrained.mdp` is this
checkpoint, and the game runs it natively.

**This settles the question the design turned on.** A perfect-marksmanship agent — one
that solves the lead-intercept exactly and never misses — still loses *every* game, with
every city gone, around wave 16. The difficulty is not aiming; it is allocation under an
ammunition budget that goes negative as waves grow. Exact velocity information does not
trivialise the game, which is why the observation gives raw state rather than hiding it.

Kills-per-interceptor of about 1.09 is where the visible headroom is: blasts are catching a
cluster only occasionally. Waiting for MIRV spreads to converge, and spending the
end-of-wave ammo bonus wisely, are exactly the judgement calls a learned policy can beat a
greedy one at.

*(Deliberately reasonable, not optimal — a yardstick, not a champion. Routine
policy evaluation and checkpoint selection use the disjoint stream prefix at
offset 0, not this held-out block.)*

## M5 — ML infrastructure

nanobind bindings; Gymnasium environment; `VecSim` (N parallel sims, thread pool, GIL
released, zero-copy batched observations); reward design. Reuses the **M4 evaluation
protocol** so learned and scripted agents are compared identically.

## M6 — Train

Custom PPO (PyTorch) with a curriculum and a guided experimental loop. The
learner progresses through LOW and MEDIUM, diagnoses the remaining behavior gap,
and can take on HIGH as the expert challenge. The bundled relational policy
already scores 90,865.9 on the held-out protocol: clearly above MEDIUM
(63,295.6), below HIGH (98,542.3).

## M7 — Watch the AI / takeover

Export the policy → in-process C++ inference; live spectator mode; human takeover from any
point in an AI-played game (scripted or learned).

## M8 — Training console (PySide6)

A desktop UI for running and understanding training: start / pause / stop / reset a
run, tweak the parameters, watch the curves, browse and play back recordings, inspect
the model, and launch the game — without leaving the window.

**Why it earns a milestone.** The product is a fun way to teach yourself machine
learning. The console guides the learner through predict → change one thing →
train → watch → measure → explain → challenge. The project already bets on
interpretability: recordings exist because a return curve cannot tell you the
policy has learned to ignore MIRVs. Putting the curve, visible behavior, fair
comparison, and explanation side by side turns training into a learning loop
rather than an unexplained dashboard.

### Design intent — modern, simple, elegant

This is a constraint, not a mood — and *simple* here means **the least complexity the
job actually needs, not minimalism as a style**. This is an engineering instrument:
density earns its place when it is telling you something. The thing to cut is
ornament and indirection, never information you would otherwise go hunting for.

Concretely:

**One screen, not a tab bar.** The run is the subject. A tabbed console makes you
hunt for the thing you came to look at. The curve gets the space; everything else is
a strip around it.

> **Amended 2026-07-26, at the human's request: two tabs, and only two.**
> **TRAINING** is the screen above — what you *watch*, live, from across the
> room. **STATISTICS** is the full per-episode stat block, the kills-per-shot
> distribution and the cause curves, which is what you *read* when a run has
> plateaued and you want to know why.
>
> The original rule stands for everything else, and the reason it does not stand
> here is that those are two different activities, not two views of one. Fourteen
> numbers and a histogram reward being studied; putting them on the monitoring
> screen would have made the thing you glance at fifty times an hour permanently
> crowded by the thing you open twice a run. Everything *around* the plots — the
> tiles, the run picker, the **vs** picker, the recordings, the model, the log —
> is shared and does not move, so switching tabs never changes which run you are
> looking at, which is the failure mode the rule was written against.

**Control is small so monitoring can be large.** Three affordances, not a dashboard
of them: one primary button that changes meaning (Start → Pause → Resume), Stop, and
Reset tucked away because it is destructive. If control needs more than a slim bar,
the design is wrong.

**Progressive disclosure over a wall of fields.** There are ~20 hyperparameters and
their defaults are good and reasoned. Show the four that change a run's character
(envs, steps, updates, learning rate); everything else lives behind *Advanced*.
Each field carries the reasoning already written beside it in code as its tooltip —
the UI should teach, since that is what this project is for.

**The baseline is the hero — on the held-out benchmark.** 98,542 is the final
number to beat. Routine training curves use a separate validation seed split so
selecting `policy-best.pt` cannot tune against the headline test; the console
labels the split and only draws a baseline that belongs to the same protocol.

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
| `runs/evals.csv` | tail → the score curve the baseline is drawn across |
| `runs/update-*.mdr` | list → ▶ Play (or double-click) launches `md_app --replay` |
| `runs/checkpoints/*.pt` | list → `--load` to score, or `--resume` to continue |
| `runs/model.json` | the network being trained — layers, shapes, parameter count |
| `runs/train.log` | tail → a log pane, for a run this console never started |
| stdout | the same pane, when the run *is* this console's child |

> **`evals.csv` was added for this.** The plan said to draw the scripted baseline across the
> *return* curve, and that turned out to be wrong: the return in `metrics.csv` is
> shaped, scaled and summed undiscounted, so it reads in the tens and has no fixed
> relationship to a game score. The only number in a run that *is* comparable is
> an evaluation score — deterministic seeds, shared C++ `summarize`, greedy play —
> and it was being printed and thrown away. It is now appended to `runs/evals.csv`
> with its seed split, cadence, and inference device. Periodic rows use validation
> seeds for checkpoint selection; the canonical 32 seeds remain untouched until a
> final held-out score. Return, entropy and value loss keep their own axes below it.

This buys more than responsiveness: the UI can **attach to a run started from a
terminal**, a crashed UI leaves training untouched, and the whole thing works against a
synced directory from the Debian box.

### Control: a file the loop polls, not a signal ✅ *(implemented)*

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

> **Built as marker files, not `control.json`.** `runs/PAUSE` and `runs/STOP`: the
> presence of a file rather than its contents. A JSON file has to be *parsed*, and the
> reader can catch the writer mid-write — the same torn read the metrics tail goes to
> trouble to avoid, except that here the damage is an unreadable *command* rather than
> one wrong data point. Existence needs no protocol, and `touch` produces it. Each is a
> *state* rather than an event, so `ls runs/` explains why nothing is happening, and both
> are cleared when a run starts and when one ends — a stale `STOP` must not kill
> tomorrow's run. Passing a *value* rather than a signal is the one thing this cannot
> do, so `runs/TUNING.json` sits beside them for the eval cadence: parsed, written to a
> temporary name and renamed into place, and unreadable-is-absent so a typo cannot kill
> a run that is hours old. The console's **eval every** box writes it. It
> lives in `md.control`, which the trainer and the console both import and which pulls in
> neither of them.

### What it shows

Capabilities, **not a tab layout** — see the design intent above. Most of this lives
on one screen; only the parameter form and the recording browser justify their own
surface.

1. **Run** — parameter form (the `TrainConfig` / `PPOConfig` fields, each with the
   reasoning already written beside them in code as tooltips), start/pause/stop/reset,
   live status.
2. **Curves** — return, entropy, value loss, clip fraction, from `metrics.csv`, plus
   split-labelled evaluation scores. The **98,542 canonical baseline** is drawn only
   for held-out rows produced by that same protocol.
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
   | NVIDIA | `nvidia-ml-py` (imports as `pynvml`) | CUDA cards |
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
  __main__.py    # `python -m md.ui`; explains itself when PySide6 is absent
  app.py         # QApplication bootstrap, window, `poe ui` entry point
  runner.py      # spawn the game and the trainer; read the run's output (no Qt)
  sources.py     # tail metrics.csv, list recordings + checkpoints (no Qt)
  params.py      # the trainer's knobs, read out of its source (no Qt, no torch)
  forms.py       # the parameter dialog
  charts.py      # the curve widget, baseline line included
  system.py      # psutil sampler + the GpuProbe protocol (no Qt)
  probes/        # nvidia.py, amd.py — soft imports, one file per vendor
  meters.py      # the CPU / RAM / GPU strip
  theme.py       # palette lifted from the game, dark by default
```

The control protocol itself is **not** here: it is `python/md/control.py`, because
the trainer polls it and the console writes it, and neither should have to import
the other to agree on what a pause is.

`sources.py` and `runner.py` hold **no Qt**, so both are unit-testable under
pytest without a display — which is what keeps this from being a milestone with no
tests.

**Phase 1 — Monitor (read-only).** ✅ *(implemented — ready for sign-off)* Tails
`metrics.csv` and `evals.csv`, draws the score against the baseline line with
return / entropy / value beneath it, lists recordings newest-first, and
double-click launches `md_app --replay`. Attaches to a run started from a
terminal; cannot start one. Run it with `poe ui [run-dir]`.
*Usable on its own: this is the panel you leave open during a run.*

Two things the plan did not anticipate, both now in the code with the reason
beside them: the baseline needed a comparable curve to be drawn across (hence
`evals.csv`, above), and launching the game from the *native* interpreter on
Windows needs the CLANG64 prefix put back on `PATH`, or the MinGW build dies
looking for `libc++.dll` before it can show a window.

**Phase 2 — Control.** ✅ *(implemented — ready for sign-off)* `md.control` and the
loop's once-per-update check landed first and on their own, so `touch runs/STOP`
and `touch runs/PAUSE` work with no console anywhere. The bar then reads: one
button that changes meaning (Start → Pause → Resume), Stop, and Reset, which
attaches to the next free run directory and deletes nothing. Pause and Stop work
on a run this console never started, because they are the same two files. A
started run's stdout streams into a log pane that opens itself if the run dies.

**Phase 3 — Parameters.** ✅ *(implemented — ready for sign-off)* The four headline
fields plus *Advanced*, each tooltip the `#:` reasoning already written beside the
field — read out of `train.py` and `ppo.py` with `ast` and `tokenize` rather than
by importing them, since `md.ui` may not import torch. Only changed values are
passed and the resulting command line is shown, so the dialog teaches the CLI
rather than replacing it. Every `PPOConfig` field also became a real flag,
generated from the dataclass: the form could not otherwise offer the learning
rate at all. `runs/config.json` records what a run was started with — written by
the trainer, so a terminal-started run gets one too.

**Phase 4 — Model & system.** ✅ *(implemented — ready for sign-off)* CPU and
RAM from psutil, and a GPU row from whichever vendor backend imports — both
optional, each saying which package would fill it in rather than showing a dead
meter. The strip sits at the foot of the recordings column, which is where the
space already was; no fly-out or tab was needed. **The vendor backends are
written from the published APIs and tested against stand-in modules, not against
real hardware** — there is no NVIDIA or ROCm machine in this project's loop,
which is why every field is read defensively and a probe that starts failing is
dropped rather than raised.

The model half sits above it: architecture, parameter count, the observation and
action sizes, a line per layer, and which checkpoint is newest with what it
scored. The rule above ("anything needing model state belongs in `md.train`,
surfaced as an artifact the UI reads") decided its shape — the console cannot
open a `.pt` without torch, so the trainer writes `runs/model.json` at start-up
and the console reads that. `md.modelcard` holds the format and imports torch
*nowhere*, not even lazily: `describe()` takes a state dict's **shapes**, and a
shape is a tuple of ints. Like `md.control`, it is a file both sides agree on
and neither has to import the other for.

> **One file per run, not a sidecar per checkpoint.** Within a run the
> architecture, the layer shapes and the observation and action sizes never
> change — only the iteration does, and that is already in each checkpoint's
> *name*. A file beside every `policy-*.pt` would be the same bytes repeated once
> per hundred updates. It lands beside `config.json` for the symmetry: one says
> what the run was started with, the other what it is training.

A checkpoint's score is looked up by *its own* update rather than taken from the
most recent evaluation. Those are usually the same row and occasionally are not,
and labelling update 750's score as update 800's would make the one panel whose
job is "this is the model you have" the one that lies about it.

> On Windows an AMD card has no supported Python telemetry API at all (`amdsmi`
> and `pyrsmi` are ROCm, so effectively Linux). The row therefore stays empty on
> the development machine — correctly, and with the reason on screen. A
> vendor-neutral Windows backend over the PDH performance counters would fit the
> same protocol as one more file in `probes/`.

**Phase 5 — Compare.** ✅ *(implemented — ready for sign-off)* A second picker
beside the run picker — *this run* **vs** *that one* — and every curve gains the
other run's line in the same colour at a third of the opacity, with its latest
value under each stat tile. The point at which the console starts answering
*"did that change help?"* rather than only *"what is happening now?"*.

Overlaid rather than in a second window beside it. Two plots with independent
axes make you compare by eye across a gap and do arithmetic between two y-scales;
one plot answers it by which line is higher. Same hue at lower opacity for the
same reason a different colour was rejected — it has to read as *this metric,
the other run*, not as a fourth thing on the chart. The comparison is only ever
the two curve files: an experiment is worth holding against another on what it
scored and how it got there, and the rest of the window stays about the run you
are actually driving.

It came out much smaller than planned, because the run picker had already paid
for most of it: candidate runs, re-attaching every panel, and a tail that
survives a file being appended to were all there. What was missing was a second
series per chart and a second line per tile.

**Not in the plan, but the runs asked for it: a run picker.** Experiments
accumulate one `--out-dir` each, so `runs/` stopped being a run and became a
container of them — at which point *which run the window is showing* is something
you change several times an hour, and it belongs in the window rather than in the
command that started it. The header carries a dropdown of the runs inside and
beside the attached directory; choosing one re-attaches every panel, which is the
same act Reset already performed. It also makes Phase 5 half a step shorter: two
runs are now two clicks apart rather than two terminals.

### Risks, and how each is mitigated structurally

A risk with only "be careful" behind it is a worry. Each of these gets a mechanism.

**1. Charting — no plot widget in Qt itself.**

Use **Qt Charts**, which ships with PySide6 under **LGPLv3**, same as PySide6. No extra
dependency, native look, nothing to vet. (It was GPL-only under Qt 5, which is where its
reputation for awkward licensing comes from; that is out of date. The GPL trap that *is*
still real is `PyQt6-Charts` from Riverbank — a different binding to the one used here.)

The performance worry that would push toward `pyqtgraph` does not apply at this data
rate: an update takes **seconds**, so the plot gains a point every few seconds, not
thousands per second. Either library would cope; the bundled one wins on having nothing
to install.

Still keep it behind our own `CurveView` (`append`, `set_baseline`) so nothing else
imports the charting module — cheap now, and it keeps the escape hatch open if the look
turns out wrong.

> **Licence note:** PySide6 and Qt Charts are LGPLv3 while this project is MIT. Dynamic
> linking keeps that clean, but the console must stay an *optional* component — never a
> dependency of the game or the `.deb`.

**2. Reading a file while something else is writing it.**

Training appends one line to `metrics.csv` per update; the UI reads it to draw the
curve. Three ways the naive version breaks:

* **Re-reading the whole file each refresh.** After 5,000 updates that is 5,000 lines
  re-parsed every tick, forever. *Fix:* remember the byte position where reading
  stopped and resume from there.
* **The half-written line.** Training may be mid-write when the UI reads, yielding
  `247,32768,4.87,1.6` with no newline because the rest is not flushed yet. Parsed as
  a row, that is a wrong value or a crash. *Fix:* only parse lines ending in a newline;
  keep the fragment and prepend it to the next read. **This is the one that actually
  bites** — the performance issue is merely slow, this one is wrong.
* **The file was reset.** A fresh run empties it, but the saved position still says
  byte 40,000, so reading resumes past the end and returns nothing forever. *Fix:* if
  the file is now shorter than the saved position, treat it as a new file and start
  from zero.

None of this is hard; it just has to be deliberate. `sources.py` has no Qt so pytest
can write a CSV incrementally — torn final line and truncation included — and assert
the reader yields exactly the new rows.



**3. Scope creep into a trainer.**

Make it structural instead of a rule people remember:

* **`md.ui` must never import `torch`.** That is a one-line test — import the package
  and assert `torch` is absent from `sys.modules` — so the boundary is enforced by CI
  rather than by discipline. It also keeps the console startable in an environment
  with no torch at all, which is a feature: you can watch a remote run from anywhere.
* The UI's only writes are the control file and spawning subprocesses. Everything
  else is read-only.
* Any feature that seems to need model state belongs in `md.train`, surfaced as an
  artifact the UI reads.

### Before this: onboarding

Independent of the console, and higher leverage for reach. The README is currently a
reference, so a newcomer meets a dependency matrix before a missile. Add a **quick
start** — clone, build, play, watch the AI — with the requirement tables demoted to
reference below it. Small, self-contained, and the thing most likely to decide
whether someone runs this at all.
