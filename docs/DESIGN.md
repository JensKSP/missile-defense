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
- **[deviation, revisit]** Bases are **not destructible** in v0.1. In the arcade original
  a base can be knocked out by incoming fire; we start simpler and may add this back.

### 3.3 Interceptors (player projectiles)
- Launched from a base toward an aim point `P`; fly in a straight line at fixed speed.
- On reaching `P` (or on proximity), **detonate** into a circular blast that **expands to
  a max radius, lingers, then collapses** over a short lifetime.
- Any enemy warhead entering the blast during its lifetime is destroyed — one blast can
  **chain-kill** several warheads.
- Defaults: speed `~220 u/s`, blast max radius `~14 u`, blast lifetime `~0.9 s`.

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
  it is endless survival. (For RL we also impose a max-steps truncation.)

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
Action = NoOp
       | Fire { base_id ∈ {ALPHA, DELTA, OMEGA}, target: vec2 P }
```

- **Preconditions** for `Fire`: chosen base has ammo `> 0` and is off cooldown.
  An invalid `Fire` is treated as `NoOp` (no crash, no penalty beyond wasted opportunity).
- **Human client:** mouse position → `P`; a key/button per base selects `base_id`
  (the trackball + three fire buttons of the original).
- **AI policy:** action space is **target-selection** — "assign an interceptor from base
  *j* to threat *i*" — and a helper computes the lead-intercept point `P` from the threat's
  state. This keeps the AI's action space discrete-over-entities (friendly to a
  from-scratch PPO) while producing the *same* underlying `Fire` primitive.
- **Cadence:** the agent may act **every tick** (`K = 1`), and `NoOp` is a first-class
  choice. Frame-skip may be introduced later as a training optimization.

## 6. Observation model (data availability, finalized in Step 2)

The core sim state must be fully introspectable so the observation vector can be built
**inside the C++ core** (shared by training and by live inference — no train/serve skew).
The observation is *set-structured* (variable threat/interceptor counts) and will include:
threats (position, velocity, type), in-flight interceptors and active blasts, per-base
ammo/cooldown, per-city alive flags, and wave/time. Exact encoding (fixed-max padding vs.
set/attention encoder) is a Step-2 decision.

## 7. Deviations from the arcade original (v0.1)
- Bases are **not destructible** (§3.2) — simplification to revisit.
- No bomber/satellite enemies yet (only ICBM / MIRV / smart bomb).
- Motion is straight-line; a **ballistic-arc variant** is deferred to `MissileDefense-v1`.

## 8. Open decisions & tuning knobs
- Final wave-scaling curves (count, speed, MIRV/smart-bomb rates) — set during playtest.
- Whether to restore destructible bases and bomber/satellite threats.
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

**M1 (current target): a human can play a fresh game in the Vulkan UI.** Full milestone list and
the path to M1 are in [`ROADMAP.md`](ROADMAP.md).

## 13. Audio & game events

Discrete game events — interceptor launched, detonation, threat destroyed, city destroyed, wave
cleared, game over — are **first-class, deterministic outputs of `md::core`**, accumulated during
`step()` and returned in `StepResult`. One event stream feeds three consumers *identically*:

- **Audio (human):** the app plays a sound effect per event (launch *thunk*, intercept *boom*, a
  city lost, …).
- **AI observation (parity):** the same events are encoded into the agent's observation, so the
  policy receives the cues a human *hears*. This is a deliberate fairness principle — **no
  information asymmetry between human and AI.** What you hear, the model "hears."
- **Replay:** events derive deterministically from `(seed, actions)`, so replays reproduce
  identical audio and cues.

The sim already computes these moments internally (firing, kills, city hits, wave clear,
termination); surfacing them as an event list is a small, pure, test-first core addition. The
audio backend (Qt 6 Multimedia `QSoundEffect` vs. a vendored zero-dependency library such as
miniaudio) is decided when audio is implemented — `qt6-multimedia-dev` is not currently installed.
Recordings capture the game's audio from the PipeWire sink monitor (`scripts/record.sh`).

---
*Freeze this spec before Step 2. Every mechanics change after the freeze invalidates
trained models and bumps the env version.*
