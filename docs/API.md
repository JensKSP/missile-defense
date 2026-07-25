# API — driving the simulation from an agent

How a policy — scripted or learned — sees the game and acts on it. Everything here
lives in `md::core`: no Qt, no rendering, no Python. The human client and the
training loop drive the *same* code, so there is no train/serve skew.

Companion documents: [DESIGN.md](DESIGN.md) (mechanics, §5 the control interface,
§5.1 the player model) and [ROADMAP.md](ROADMAP.md) (milestones).

## 1. The loop

`Sim::step` advances exactly one fixed `dt` tick and owns no timing — drivers own
the loop. A run is fully determined by `(seed, config, actions)`.

```cpp
md::Sim sim{config};
sim.reset(seed);
const md::StepResult r = sim.step(action);   // r.reward = score delta, r.terminated
```

- **Tick rate** is fixed at 60 Hz (`Config::dt = 1/60`). Training runs as fast as
  the CPU allows; only the app throttles to wall-clock.
- **The field** is a fixed 320 × 180 world box, origin bottom-left, y-up. Window
  size is decoupled by the renderer's letterbox projection, so screen resolution
  never leaks into the simulation or the observation.
- **A `Sim` is a value** — trivially copyable, ~12 KB, no heap, no pointers. A
  snapshot is a `memcpy`; a fork is an assignment.

## 2. Observation — raw state, never analysis

`md/observation.hpp` encodes the simulation into one flat `float` vector, written
into a caller-owned buffer so a batched `VecSim` can fill a row zero-copy.

```cpp
constexpr md::ObsSpec spec;                  // defaults to full simulation capacity
std::vector<float> obs(spec.size());
md::encode(sim, spec, obs);                  // allocation-free, noexcept
```

### The admission rule

> **A feature belongs in the observation only if a human perceives it directly on
> screen.**

This is the concrete form of the fairness principle in DESIGN §13. It cuts both
ways, and the second direction is the one that matters:

| Included (perceived) | Excluded (computed) |
|---|---|
| threat position, velocity, type | time-to-impact |
| interceptor position, velocity, detonation point | which city a threat is aimed at |
| blast position and current radius | danger / priority ranking |
| per-battery ammo, cooldown, alive | intercept points |
| per-city alive, position | "recommended target" |
| crosshair position, trigger cooldown | anything a heuristic would compute |
| wave, score, this tick's event counts | |

Deriving those excluded quantities **is** the job we want the policy to learn.
Supplying them would be doing the hard part and then claiming the network
discovered it.

Two consequences worth stating explicitly:

- **Velocity is included.** It is perceived (you see motion), and withholding it
  changes nothing: threats travel in straight lines, so `pos_t − pos_{t−1}`
  recovers it exactly. Hiding it would only force frame-stacking and burn training
  time relearning subtraction.
- **Ordering is neutral.** Slots are emitted in the simulation's own array order,
  deliberately *not* sorted by urgency — any ranking is itself a triage heuristic,
  and sorting would smuggle it back in through the ordering.

### Layout and scaling

Fixed-size, zero-padded, in this order. Each entity slot starts with a `present`
flag; empty slots are all-zero.

| Block | Slots × features |
|---|---|
| threats | `spec.threats` × 9 — present, pos(2), vel(2), type one-hot(4) |
| interceptors | `spec.interceptors` × 7 — present, pos(2), vel(2), target(2) |
| blasts | `spec.blasts` × 4 — present, pos(2), radius |
| batteries | 3 × 4 — alive, x, ammo, cooldown |
| cities | 6 × 2 — alive, x |
| globals | 5 — crosshair(2), trigger cooldown, wave, score |
| events | 10 — count per `EventType` this tick |

Positions map to `[-1, 1]` against the world box, velocities scale by
`interceptor_speed`, timers by their own intervals, score by `bonus_city_score`
(so `1.0` is one bonus city's worth — the threshold the policy must actually reason
about). The event block is what delivers audio parity: the model "hears" what you
hear.

`ObsSpec` defaults to the simulation's own capacities, so the policy is never blind
to a threat you can see. Lowering the caps speeds training but truncates by slot
index and can hide a live threat — an information asymmetry. Do that knowingly.

## 3. Action — a discrete space over entities

The control primitive is `Action` (DESIGN §5): steer a crosshair that lives in the
simulation, and optionally pull the trigger. The crosshair is speed-capped and
shots are paced by a trigger interval, so **naming a distant point costs time** —
the limits apply to human and agent alike (DESIGN §5.1).

Policies do not emit raw `Action`s. They pick an index in a discrete space, and
`md/intercept.hpp` turns it into the primitive:

```cpp
const std::uint32_t n = md::action_count(spec);      // 1 + 3 × spec.threats
std::array<bool, n> mask{};
md::action_mask(sim, spec, mask);                    // mask illegal logits
const md::Action a = md::decode_action(sim, spec, index);
sim.step(a);
```

- **Index 0** is NoOp — a first-class choice.
- **Index ≥ 1** maps to *(battery, threat slot)* in row-major order: "engage threat
  *i* from battery *j*".
- **The mask** hides only what can never work: an empty threat slot, or a dead /
  out-of-ammo battery. **Cooldowns deliberately do not mask an action out** —
  while the trigger recovers, steering toward the next target is still useful work,
  and the engagement fires as soon as it legally can.

Masking matters more than it looks: from-scratch PPO otherwise spends a lot of
early training discovering that firing an empty battery does nothing.

## 4. Intercept — the actuation helper

`solve_intercept` answers "where and when would a shot from this battery meet this
threat". With the crosshair speed-capped this is no longer the classic quadratic,
because the flight cannot start until the *cursor* has arrived. The point must
satisfy

```
t = |X(t) − crosshair| / aim_max_speed  +  |X(t) − base| / interceptor_speed
X(t) = threat.pos + threat.velocity · t
```

solved by fixed-point iteration, which contracts whenever threats are slower than
both the cursor and the interceptor — true for any sane config. With
`aim_max_speed == 0` the aim term vanishes and it degenerates to the ordinary lead
intercept.

`engage(sim, base, slot)` wraps this into the macro-action `decode_action` uses:
steer toward the point, and fire on the tick the crosshair actually arrives. It is
shared by the scripted baseline and the learned policy so the two cannot drift
apart.

This helper is *actuation*, not perception — which is why it may compute things the
observation withholds. The policy chooses **which** threat to engage; the helper
only carries out that choice.

## 5. Reward

`StepResult::reward` is the per-tick score delta, derived from the scoring table in
DESIGN §4.3 (kills, end-of-wave ammo and city bonuses). Shaping is a training-time
concern and is layered on top in Python, not baked into the core.

Episodes terminate when the last city falls (`StepResult::terminated`). Training
imposes its own max-steps truncation; the simulation has no opinion about it.

## 6. Determinism

Same `(seed, config, actions)` ⇒ bit-identical trajectory, Debug and Release alike
(`-ffp-contract=off`, no fast-math), enforced by a golden-checksum test. Two
consequences the API leans on:

- A full run is `(seed, config, action log)` — kilobytes, not frames.
- Replay, scrubbing and human takeover are re-simulation plus `memcpy`, because a
  `Sim` is a value.

## 7. Status

| Piece | State |
|---|---|
| `Sim`, `Action`, player model | done |
| `md/observation.hpp`, `md/intercept.hpp` | done |
| Scripted baseline + evaluation protocol (`md::agent`, M4) | done — `poe eval` |
| nanobind bindings, Gymnasium env, `VecSim` (M5) | planned |
| Episode recording / replay / takeover (M3) | planned |

The learned agent is scored by the **same** `md::agent::evaluate` over the **same**
`default_seeds`, so "beat the baseline" is a concrete claim. Current baseline:
mean score 15,583, mean wave 14.4, 0/6 cities surviving, 1.10 kills per
interceptor ([ROADMAP.md](ROADMAP.md#m4--algorithmic-reference-ai--implemented--ready-for-sign-off)).
