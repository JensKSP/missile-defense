# Missile Defense — Game Design Specification

> **Status:** v0.1 strawman — a starting point to react to and edit, *not* frozen.
> This document is the single source of truth for the game rules. It also **doubles
> as the reward specification** for reinforcement learning (Step 3), so decisions here
> propagate all the way through training. Nothing below is trained against until the
> **"freeze mechanics"** gate is passed.

## 1. Overview

A faithful clone of **Atari's Missile Command** (1980, Dave Theurer). The player defends
six cities from waves of descending enemy missiles by launching interceptors from three
ground batteries. Interceptors travel to an aim point and detonate into an expanding
blast that destroys any enemy warhead caught inside it. Ammunition is limited per wave.
The game is an **endless, escalating survival** contest — there is no victory, only a
higher score and a later "THE END".

The same simulation is driven by three clients: a human player (Qt + Vulkan), a headless
Python training environment (Gymnasium via nanobind), and — later — a trained policy run
in-process. See `docs/ROADMAP.md` for the build order.

## 2. World & simulation model

| Property | Value (default) | Notes |
|---|---|---|
| Dimensionality | 2D | |
| World size | `320 × 180` world units | 16:9 virtual field; renderer scales to window |
| Origin | bottom-left, **y-up** | ground at `y = 0`, sky at top |
| Timestep | fixed `dt = 1/60 s` | sim is fully decoupled from wall clock |
| Time compression | step without rendering/sleeping | headless training runs as fast as CPU allows |
| Numerics | `float` (32-bit) | **[decision]** same-machine reproducible, not cross-platform bitwise |
| Threading | single-threaded core update | determinism; parallelism is *N independent sim instances*, not intra-sim threads |
| RNG | per-instance PCG, seeded in constructor | no global RNG state; randomness enters only at spawn |

**Determinism contract:** given the same `(seed, action-sequence)`, the sim produces a
bitwise-identical trajectory **on the same build/machine**. Enforced by a CI test.
Compile with no fast-math and consistent FP settings. This is what makes replays
(Step 4) reduce to just `(seed, actions)`.

## 3. Entities

### 3.1 Cities (defended assets)
- **6 cities**, arranged in two groups of three along the bottom, interleaved with the
  bases: `[ALPHA] city city city [DELTA] city city city [OMEGA]`.
- One-hit-kill. A destroyed city stays destroyed for the rest of the episode.
- All cities equal value to start (weighting is a later knob).

### 3.2 Bases (interceptor batteries)
| Base | Position | Ammo / wave | Notes |
|---|---|---|---|
| ALPHA | left | 10 | |
| DELTA | center | 10 | |
| OMEGA | right | 10 | |

- **30 interceptors total per wave**, replenished at the start of each wave.
- Minimum inter-launch cooldown per base: `~6 ticks (0.1 s)` — prevents spamming one tick.
- Bases **are destructible**: a warhead that lands on one knocks it out, and it stops
  firing for the remainder of the wave. Like the arcade, batteries are **rebuilt at the
  start of the next wave** — losing one costs its remaining ammo and its coverage, but is
  not permanent. (Permanent loss meant three dead batteries left the player unable to
  fire ever again, watching a decided game finish itself: no agency for a human, and pure
  noise in a training rollout.)

### 3.3 Interceptors (player projectiles)
- Launched from a base toward an aim point `P`; fly in a straight line at fixed speed.
- On reaching `P` (or on proximity), **detonate** into a circular blast that **expands to
  a max radius, lingers, then collapses** over a short lifetime.
- Any enemy warhead entering the blast during its lifetime is destroyed — one blast can
  **chain-kill** several warheads.
- Defaults: speed `~220 u/s`, blast max radius `~14 u`, blast lifetime `~0.9 s`.
- The fireball's colour/opacity phase is driven by normalized age
  (`age / blast_lifetime`). The observation exposes that same visible phase alongside
  radius; a full-size lingering blast otherwise gives no indication how close it is to
  expiry.

### 3.4 Threats (enemies)
| Type | Behavior | Appears |
|---|---|---|
| **ICBM** | straight-line descent from top toward a random city or base | wave 1+ |
| **MIRV** | an ICBM that **splits** into `k = 2–3` warheads at a random altitude, each retargeting | wave 2+ |
| **Smart bomb** | descends but **steers to dodge** nearby active blasts (a "decoy" that wastes ammo) | wave ≥ 5 |

- Threat speed and count scale with wave number (see §4).
- Damage: a warhead reaching a city destroys it; reaching the ground harmlessly otherwise.

## 4. Episode structure & scoring

### 4.1 Waves
- Play proceeds in **waves**. Wave `n` spawns `N(n)` threats; speed, MIRV probability, and
  (from wave 5) smart-bomb count all scale with `n`.
- Starting point: `N(1) ≈ 8`, growing roughly linearly; speed multiplier `1.0 → …` per wave.
  *(All scaling curves are tuning knobs, finalized during playtest.)*
- Ammo is refilled to 30 at the start of each wave.

### 4.2 Termination
- The episode ends when **all 6 cities are destroyed** ("THE END"). No win condition —
  it is endless survival. Canonical evaluation also truncates at exactly 120,000
  simulation ticks; a final partial four-tick decision window stops at the cap
  instead of rounding the episode up.

### 4.3 Scoring (also the reward seed)
| Event | Points |
|---|---|
| Enemy warhead destroyed | `+25 × wave_multiplier` |
| End-of-wave: unused interceptor | `+5 each` |
| End-of-wave: surviving city | `+100 each` |
| Bonus city | every `10,000` points (up to a cap) |

The **reward function** for RL (Step 2) is derived from this score — likely per-step score
delta plus shaping for intercept timing/efficiency. Reward shaping is its own iteration
loop and is *not* fixed by this document; this table only fixes the human-facing score.

## 5. Control interface — the shared `Action`

A single action primitive is shared by the human and the AI:

```
Action = { aim: vec2, base_id ∈ {ALPHA, DELTA, OMEGA}, move: bool, fire: bool }
```

The **crosshair is simulation state**, not a free parameter of the action — the arcade
cabinet's trackball, modelled honestly. An action asks to *steer* it toward `aim`, and
optionally to launch from `base_id` at wherever the crosshair actually ended up.

- **Preconditions** for a launch: the chosen base is alive, has ammo `> 0` and is off its
  own cooldown, **and** the global trigger interval has elapsed. A rejected launch is a
  no-op (no crash, no ammo spent, no penalty beyond the wasted opportunity).
- **Human client:** the current mouse position is submitted as `aim`; a click is
  latched until the next 15 Hz decision tick, then sets `fire` once.
- **AI policy:** action space stays **target-selection** — "assign an interceptor from
  base *j* to threat *i*" — with a helper turning that into steer-then-fire over several
  ticks. Discrete-over-entities, friendly to a from-scratch PPO, same underlying primitive.
- **Cadence:** the simulation samples a new action every 4 ticks (15 Hz) and holds it
  between decisions. Training's frame-skip uses the same interval, so neither the
  scripted nor learned driver gets faster reactions by how its outer loop is written.

### 5.1 The player model — why the limits exist

A machine given exact state also gets *perfect mechanics* for free. Without constraints an
agent can name opposite corners of the map on consecutive ticks and empty every battery in
one second — capabilities no hand has. That is not an information asymmetry (§13 covers
that); it is an **execution asymmetry**, and it makes any human-vs-AI score comparison
meaningless. The standard remedy in the field (AlphaStar, OpenAI Five) is to **constrain
actuation, not perception** — so we cap the hands and leave the eyes alone:

| Knob | Models | Default |
|---|---|---|
| `aim_max_speed` | How fast a hand moves the cursor (world units/s; `0` = instant) | `1200` |
| `fire_interval` | The trigger finger — min seconds between **any** two launches | `0.33` |
| `base_cooldown` | The battery's own reload, applied on top | `0.1` |
| `decision_interval` | Reaction rate — simulation ticks per newly sampled action | `4` |

These player limits are enforced inside `Sim::step`, so **every driver obeys them identically** —
human, scripted baseline, and learned policy. Consequences worth noting:

- The two limits cover different regimes: the trigger interval binds on *clustered* targets
  (a MIRV splitting into three), the crosshair cap on *cross-field* repositioning. Neither
  subsumes the other.
- Deliberately **not** modelled: aim *error*. Noise would blur the analytic-intercept
  advantage, but it is not where the game's difficulty lives, and it would make the reward
  signal noisier for no design gain.
- Defaults are strawman values. `aim_max_speed` is set generously so it is invisible during
  human play (a mouse rarely exceeds it) while still denying the agent free teleportation;
  calibrate both from recorded human play before the mechanics freeze.
- Setting `aim_max_speed` or `fire_interval` to `0` disables that limit, which is how
  tests of *other* mechanics avoid being paced by the player model.

## 6. Observation model (data availability, finalized in Step 2)

The core sim state must be fully introspectable so the observation vector can be built
**inside the C++ core** (shared by training and by live inference — no train/serve skew).
The observation is *set-structured* (variable threat/interceptor counts) and includes:
threats (position, velocity, type), in-flight interceptors, active blasts (position,
radius, and their visibly rendered lifetime phase), per-base ammo/cooldown, per-city
alive flags, and wave/time. It is encoded as a fixed-capacity, zero-padded vector
of 1,959 floats. Event counts returned to a policy cover every underlying tick in
its four-tick decision window, so frame skip cannot erase an audible cue.

## 7. Deviations from the arcade original (v0.1)
- No bomber/satellite enemies yet (only ICBM / MIRV / smart bomb).
- Motion is straight-line; a **ballistic-arc variant** is deferred to `MissileDefense-v1`.

### 7.1 Ground damage is resolved by impact point
A warhead destroys whatever installation actually stands where it lands (within half a
ground slot), **not** whatever it was aimed at when it spawned. The two differ for smart
bombs, which steer sideways to dodge blasts and can drift clear of their original target.

This is a parity requirement, not a detail: if the stored assignment decided the outcome,
a bomb that visibly drifted away would still level the city it started towards, and
neither the player nor a policy could predict that from what is on screen. `Threat`'s
`target_kind`/`target_index` therefore record *intent at launch* — they fix the initial
heading and nothing more. A dodging bomb can now miss entirely, which is the honest
consequence of letting it dodge.

## 8. Open decisions & tuning knobs
- Final wave-scaling curves (count, speed, MIRV/smart-bomb rates) — set during playtest.
- Whether to add bomber/satellite threats.
- City weighting (equal vs. valued).
- Reward shaping details (Step 2), observation encoding (Step 2), curriculum (Step 3).

## 9. Architecture & performance

**One sim, three drivers.** The pure C++ core (`md::core`) is driven by the Qt+Vulkan app
(human), the nanobind/Python layer (training), and recorded-action playback. `core/` depends
on nothing — no Qt, no Vulkan, no Python.

**A `Sim` is a self-contained POD value.** All state lives inline at fixed capacity (bounded by
`max_threats`, `max_interceptors`, `max_blasts`, …), with no pointers, no heap allocation in
`step()`, no globals, and its own `Pcg32`. Consequences:
- snapshot = `memcpy`; serialize = write the bytes;
- cheap to construct/copy/move → **massive parallelism** by stepping N independent sims;
- combined with determinism, a full run is just `(seed, actions)`.

**Loop ownership.** `Sim::step(action)` advances exactly one fixed `dt` tick and owns no timing.
*Drivers* own the loop: the human app uses a fixed-timestep accumulator with render
interpolation; training calls `step` as fast as possible; replay feeds recorded actions.

**Hot-path rules (production).** No virtual dispatch — threat variants are an enum tag, not a
class hierarchy. No allocation and no exceptions in `step()`.

**Parallelism.** A `VecSim` owns N independent sims and steps them across a thread pool with the
Python GIL released; observations are written into one contiguous buffer exposed zero-copy to
NumPy (nanobind/DLPack). Sims run on CPU; the policy trains on GPU (EnvPool-style).

**Determinism, including floating point.** Same `(seed, actions)` ⇒ identical trajectory on this
machine. FP behavior is pinned consistently across Debug/Release (consistent `-ffp-contract`, no
fast-math) so a run recorded by Release training replays bit-identically in the UI. The
determinism test asserts Debug == Release.

## 10. Recording & replay

- **Primary format:** `(seed, config-version, action-log)` — kilobytes per episode, so millions
  of training runs can be recorded and later filtered by outcome (best score, worst loss).
- **Playback:** deterministic re-simulation. For pause / scrub / variable-speed the UI
  materializes the trajectory into a buffer of POD snapshots on load (each snapshot a `memcpy`).
- **Takeover:** because a `Sim` is a value, the UI re-sims a recording to any tick, then switches
  the action source from the recording to human input and continues — the timeline diverges from
  that point on.
- *(Optional)* full per-tick state recording for debugging when determinism is in question.

## 11. UI modes (one mechanism)

Every mode is just "a driver feeds `Action`s into a `Sim`; the renderer draws the `Sim`":
1. **Fresh play** — human, from `reset(seed)`.
2. **Replay** — recorded actions, no input.
3. **Takeover** — replay to tick T, then hand control to the human.

## 12. Milestones

**M1 passed: a human can play a fresh game in the Vulkan UI.** The current
milestone status and remaining work are in [`ROADMAP.md`](ROADMAP.md).

## 13. Audio & game events

Discrete game events — interceptor launched, detonation, threat destroyed, city destroyed, wave
cleared, game over — are **first-class, deterministic outputs of `md::core`**, accumulated during
each `step()` and available from the simulation. One event stream feeds three consumers:

- **Audio (human):** the app plays a sound effect per event (launch *thunk*, intercept *boom*, a
  city lost, …).
- **AI observation (parity):** the same events are counted in the agent's observation. A direct
  encoder reports one tick; the vector environment sums all four ticks before the next 15 Hz
  decision, so the policy receives every cue a human *hears*. This is a deliberate fairness
  principle — **no information asymmetry between human and AI.** What you hear, the model
  "hears."
- **Replay:** events derive deterministically from `(seed, actions)`, so replays reproduce
  identical audio and cues.

The sim computes these moments internally (firing, kills, city hits, wave clear,
termination) and surfaces them as a fixed-capacity event list. Audio uses miniaudio
(preferring the system header, otherwise fetched at build time), so it does not
need Qt Multimedia. Recordings capture the game's audio from the PipeWire sink
monitor (`scripts/record.sh`).

---
*Freeze this spec before Step 2. Every mechanics change after the freeze invalidates
trained models and bumps the env version.*
