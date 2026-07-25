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

### There is no win condition

The episode ends only when **all six cities are gone**. There is no victory state —
this is endless survival, as in the 1980 arcade original, which simply flashes
"THE END". So the objective is **score**, and survival is instrumental: more waves
survived means more score. Defeat is not merely likely but *arithmetically
certain* — wave *N* sends `(8 + 2(N−1)) × (1 + 2·p_mirv)` warheads against a fixed
30 interceptors, so from about wave 8 you must average more than one kill per shot,
and past wave 20 more than three. See ROADMAP M4 for the measured baseline.

That shapes the RL problem: it is a **maximise-return-before-inevitable-death**
task, not a goal-reaching one. There is no sparse success signal to discover, which
is good — the score is dense and always informative.

### The primary reward is the score delta

`StepResult::reward` is the per-tick score delta (DESIGN §4.3: +25 per kill, and at
each wave end +5 per unused interceptor and +100 per surviving city). Keep the RL
objective identical to the benchmark metric, or you will optimise something other
than what `md::agent::evaluate` reports.

### The trap: at 60 Hz, discounting erases the city bonus

Losing a city produces **no immediate reward change at all**. Its cost appears only
at the next wave boundary, as a smaller `+100 × surviving cities`. Now discount it:

| γ | Effective horizon `1/(1−γ)` | at 60 Hz |
|---|---|---|
| 0.99 | 100 steps | **1.7 s** |
| 0.997 | 333 steps | 5.6 s |
| 0.999 | 1000 steps | 16.7 s |

A wave boundary is ~18 s away. With γ = 0.99 it is discounted by `0.99^1080 ≈ 2×10⁻⁵`
— **mathematically invisible**. An agent trained that way learns "shoot things for
+25" and will happily let every city die. This is the single largest reward-design
hazard in the project.

Two fixes, and the recommendation is to use both.

**1. Frame-skip.** Act every 4 ticks (15 Hz — decisions are ~100 ms scale anyway,
and most ticks are NoOp). That shortens every horizon by 4× in step counts, making
γ = 0.997 span a whole wave.

**2. Potential-based shaping**, which moves the deferred bonus to the instant it is
earned *without* changing the optimal policy (Ng, Harada & Russell, 1999):

```
Φ(s) = 100 · live_cities + 5 · total_ammo
r′   = Δscore + γ·Φ(s′) − Φ(s)
```

This is exact rather than ad-hoc: those weights are the *same* 100-per-city and
5-per-interceptor the end-of-wave bonus already pays — the shaping merely delivers
them continuously. Losing a city costs ≈ −100 the moment it happens, a bonus city
pays +100 on the spot, and wasted ammo is felt immediately. Because it is a
potential difference, the optimal policy is provably unchanged: you fix credit
assignment without biasing the objective.

### Practicalities

- **Normalise.** Divide by ~100 so the value head sees O(1) targets.
- **Truncation is not termination.** On a time-limit cutoff you must bootstrap
  `V(s′)`; on real game-over you must not. `StepResult` carries only `terminated`,
  so the Gym wrapper owns truncation and must keep the two distinct — conflating
  them teaches the agent that running out of clock is as bad as dying.
- **Do not reward survival directly.** A per-tick "still alive" bonus competes with
  the score objective and encourages stalling; termination handling already makes
  dying expensive, since a dead agent collects nothing more.

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
mean score 18,036, mean wave 16.0, 0/6 cities surviving, 1.09 kills per
interceptor ([ROADMAP.md](ROADMAP.md#m4--algorithmic-reference-ai--implemented--ready-for-sign-off)).
