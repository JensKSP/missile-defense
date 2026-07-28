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
- **Decision rate** defaults to every 4 ticks (15 Hz). `Sim` samples and holds
  the action itself, so human, scripted and learned drivers cannot gain reaction
  speed by calling `step` more often. The app keeps a mouse click pending until
  the next sampling tick.
- **The field** is a fixed 320 × 180 world box, origin bottom-left, y-up. Window
  size is decoupled by the renderer's letterbox projection, so screen resolution
  never leaks into the simulation or the observation.
- **A `Sim` is a value** — trivially copyable, ~12 KB, no heap, no pointers. A
  snapshot is a `memcpy`; a fork is an assignment.

`StepResult` carries the reward and the termination flag, plus three tallies of
what happened *this tick* that cannot be recovered afterwards:

```cpp
std::int32_t wasted;         // blasts that expired having killed nothing
std::int32_t multi_kills;    // kills beyond a blast's first
std::array<std::int32_t, md::kills_per_shot_bins> kills_per_shot;  // 0,1,2,3,4+
```

`kills_per_shot` bins each blast that expired this tick by its lifetime kill
count, so `kills_per_shot[0] == wasted` by construction and the top bin absorbs
anything above three. It is filled at blast *expiry* — the one moment a blast's
kill count is final — which is why it has to be instrumentation in the core and
cannot be derived from the event stream by a caller counting `ThreatKilled`:
those events say a threat died, not which interceptor's blast is responsible for
how many. Aggregated over an episode and a seed set (`md::agent::EpisodeResult`,
`Summary`) this is the distribution behind the kills-per-shot average, and the
only thing that distinguishes an agent catching clusters from one whose mean is
propped up by luck.

The core counts and prices nothing: `wasted` and `multi_kills` are what the
Python shaping puts a number on (`missile_defense.sim.env.Shaping.waste_penalty` and
`multikill_bonus`), while `kills_per_shot` is reporting only and no reward reads
it. Keeping the two apart is deliberate — a statistic that fed the objective
would stop being a measurement of it.

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
| blast position, current radius, lifetime phase | danger / priority ranking |
| per-battery ammo, cooldown, alive | intercept points |
| per-city alive, position | "recommended target" |
| crosshair position, trigger cooldown | anything a heuristic would compute |
| wave, score, event counts for the agent decision window | |

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
| blasts | `spec.blasts` × 5 — present, pos(2), radius, lifetime phase |
| batteries | 3 × 4 — alive, x, ammo, cooldown |
| cities | 6 × 2 — alive, x |
| globals | 5 — crosshair(2), trigger cooldown, wave, score |
| events | 10 — count per `EventType` in the current agent window |

At the default full capacities this is exactly **1,959 floats**.

Positions map to `[-1, 1]` against the world box, velocities scale by
`interceptor_speed`, timers by their own intervals, score by `bonus_city_score`
(so `1.0` is one bonus city's worth — the threshold the policy must actually reason
about). A blast's lifetime phase is `age / blast_lifetime`, the same value that
drives its rendered fireball; radius alone is ambiguous while a full-size blast
lingers. A direct `encode` call reports the current simulation tick's events;
`VecEnv` sums them across every tick in the frame-skip window so an early cue is
not lost before the policy's next decision. The event block is what delivers audio
parity: the model "hears" what you hear.

`ObsSpec` defaults to the simulation's own capacities, so the policy is never blind
to a threat you can see. Lowering the caps speeds training but truncates by slot
index and can hide a live threat — an information asymmetry. Such a policy may be
trained and inspected, but the canonical evaluator rejects its reduced spec rather
than comparing it with the full-state scripted baseline.

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

`StepResult::reward` is the per-tick score delta. Keep the RL objective identical
to the benchmark metric, or you will optimise something other than what
`md::agent::evaluate` reports.

The scoring follows the 1980 arcade original:

| Event | Points |
|---|---|
| Missile destroyed (ICBM, MIRV, split warhead) | 25 |
| Smart bomb destroyed | 125 |
| Unused interceptor, at wave end | 5 each |
| Surviving city, at wave end | 100 each |
| Bonus city | every 10,000 points |

**All of it is multiplied by the wave multiplier**, which steps up every two waves
and caps at ×6 from wave 11: ×1 for waves 1–2, ×2 for 3–4, and so on. This is the
single most important fact about the objective, because it is what makes the game
about *surviving deep* rather than playing early waves cleanly — at the cap a
surviving city is worth 600 and a smart bomb 750. A version of this simulation
without the multiplier flattened that incentive completely, and the scripted
baseline's score increased several-fold when it was restored. On the held-out
canonical protocol, the current handicapped baseline averages 13,687.28125.

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

**1. Frame-skip.** Act every 4 ticks (15 Hz — decisions are ~67 ms apart,
and most ticks are NoOp). That shortens every horizon by 4× in step counts, making
γ = 0.997 span a whole wave. `VecEnv` makes this value the copied Config's
`decision_interval`, aggregates events across the four ticks, and stops a final
window at the exact episode tick cap rather than rounding its duration up.

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
- **Survival is not something you can farm here.** An earlier version of this page
  warned against rewarding survival directly, on the usual grounds that it
  encourages stalling. That warning does not apply to this game: the wave schedule
  is not under the agent's control, there is no way to slow the game down, and
  doing nothing kills you fastest of all. The only way to last longer is to defend
  better. If a survival term ever earns its place, the objection to answer is a
  different one — that a per-tick bonus pays the same whether the last shot was
  brilliant or wasted, so it carries far less per-action credit than the score.

## 6. Determinism

Same `(seed, config, actions)` ⇒ bit-identical trajectory, Debug and Release alike
(`-ffp-contract=off`, no fast-math), enforced by a golden-checksum test. Two
consequences the API leans on:

- A full run is `(seed, config, action log)` — kilobytes, not frames.
- Replay, scrubbing and human takeover are re-simulation plus `memcpy`, because a
  `Sim` is a value.

## 7. `.mdp` — a trained policy as data

The format both the game and the evaluator load a learned policy from. Written by
`missile_defense.sim.policy_format`, read by `missile_defense.sim.policy_format` and `agent/src/policy.cpp`.

**Why it is not a `.pt`.** A PyTorch checkpoint is a pickle, and loading one runs
whatever its author put in it. That is fine for a file you trained yourself and
unacceptable for one the game finds in an install directory or one somebody
downloaded. The second reason is harder: the game is C++ with no Python in it at
all — that is the promise `debian/control` keeps — so the only format it *could*
load is one that needs no torch to read. **`.pt` is never an import format here.**
It is what training writes; `missile_defense.sim.export_policy` converts it.

```
magic            8 bytes, "MDPOLICY"
container        uint32 LE — how to parse the rest of this header
manifest length  uint32 LE
manifest         UTF-8 JSON, exactly that many bytes
payload          the tensors, back to back, at the offsets the manifest gives
```

**Two version numbers, answering different questions.** `container` is how to
parse the file. `schema` is what the numbers *mean* — the observation encoding
and action space of §2 and §3 — and moves whenever `md::encode` does. So a reader
that understands the container can always get far enough into the manifest to say
*"this policy is for a different simulation"*, which is a far better failure than
a parse error on a byte offset.

The compatibility promise, in full:

| Rule | Why |
|---|---|
| **Data only** — no code, no pickles, no Python type references | a reader never has to trust the file to avoid executing something |
| **Little-endian float32**, explicitly, everywhere | these files travel between machines; "it worked on mine" is not a format |
| **Every tensor's offset and length are in the manifest and bounds-checked against the payload before a byte is read** | the C++ reader has no other way to be safe, and the Python one checks identically so both agree on which files are valid |
| **SHA-256 over the payload**, in the manifest | a flipped bit in a weight is otherwise a policy that plays slightly worse and nobody ever finds out |
| **Tensor order and names are fixed by the architecture** | a file may not add, drop or rename one and still claim that architecture; readers look tensors up by name and check the set is exactly right |
| **Dimensions are resolved against each other and against the manifest** | a manifest claiming a different observation size than its weights have is a rejection, not a policy that reads off the end of a row |
| **Non-finite weights are refused on read *and* on write** | a NaN propagates to every logit, so the policy plays uniformly at random and merely looks bad |
| **Unknown metadata keys are preserved and ignored** | that is the extension point — display name, run id, canonical score — and it never affects how the weights are read |

`metadata` is where a model's **display name** lives, which is what lets the game
put `SCRIPTED` or the model's own name on the HUD while it plays. A path is not a
name: `policy-best.pt` says nothing about which run produced it.

Writing is validated *before* the file exists and lands by atomic rename, so a
promotion that would have shipped an unreadable model fails where someone can
still do something about it, and a crash mid-write never replaces a good file
with half of a bad one.

Adding an architecture means one entry in `ARCHITECTURES` in
`python/missile_defense/sim/policy_format.py` and the matching forward pass in
`agent/src/policy.cpp`. Both sides then refuse what the other cannot run — which
is the entire reason the architecture is named in the file.

### Running one: `md::agent::Policy`

```cpp
const auto policy = md::agent::Policy::load("models/learned-high.mdp");
md::agent::PolicyDriver driver{policy, md::ObsSpec{}};
const auto result = md::agent::run_episode(config, seed, driver);
```

`PolicyDriver` and `ScriptedDriver` are both `md::agent::Driver`, so a learned
policy and the M4 baseline go through **the same** `run_episode`, the same event
tallying and the same `summarize`. A second loop for learned agents is how two
contestants end up measured by two subtly different rulers.

`md_agent_eval --policy <file.mdp>` is that path from the command line; without
the flag it is the scripted baseline, exactly as before. `--action-log <file>`
writes one action index per *sampled decision*.

The game runs the same driver:

```
md_app --watch-scripted            # the M4 baseline
md_app --watch-model <file.mdp>    # a learned policy, from anywhere on disk
```

`--watch-model` **refuses** a file it cannot run rather than falling back to the
scripted agent, because watching the wrong agent and not being told is worse than
not watching at all. A policy trained against a different observation encoding is
one of those refusals, and not a rare one: the encoding gained a feature in
`a100aec` and every checkpoint written before it is unrunnable.

While an agent plays, the HUD names it — `SCRIPTED`, or the model's `display_name`
upper-cased for the pixel font — and `--report` carries the same string in its
`driver` field, so a test can assert who was playing instead of a human squinting
at a screenshot.

A **bundled** model is `models/pretrained.mdp` in the source tree; `app/CMakeLists.txt`
installs it beside the executable (`models/` on Windows, `Contents/Resources/models`
on macOS, `/usr/share/missile-defense/models` on Debian) under component `game`,
not `python` — playing against a learned agent needs no interpreter, which is the
entire reason this format exists. It is **optional**: none ships today, and with
none the WATCH AI menu entry starts the scripted agent directly rather than
offering a choice of one.

**The observation is per decision, not per tick.** `md::encode` writes the
current tick's events into the observation's event suffix; a driver that only
ever encodes on decision ticks is blind to the three ticks in between, where most
events happen. `VecEnv` accumulates across the window and overwrites that suffix,
and `PolicyDriver` does the identical thing. This is not a subtlety anyone
reasoned their way to — the cross-process parity test caught the two diverging at
decision 401, on the first event that fell in a skipped tick.

### Proving the two agree

| Test | What it proves | Where |
|---|---|---|
| `test_export_policy.py` | the NumPy forward pass matches torch | one process |
| `test_policy.cpp` | the C++ forward pass matches a checked-in fixture, logit for logit | one process |
| `test_parity.py` | **both languages play the same seed identically, decision for decision** | two processes, one file |

Only the third makes the claim that matters, because only it exercises the
*file*. The fixture (`agent/tests/fixtures/`) is checked in rather than generated
at build time on purpose: one produced by both sides at build time would be two
implementations of the same bug. Regenerate it with
`python -m tools.make_policy_fixture` when the format or the reference forward
pass changes, and expect the diff to be reviewed.

## 8. `.json` — a match, as two recordings on one seed

A **match** is two agents playing the *same* problem, shown side by side. It is
a manifest naming two `.mdr` recordings, the seed they share, and the scores the
tournament measured — nothing more, because the recordings already carry
everything needed to replay them.

```
python -m missile_defense.runs.tournament         write_manifest(match, path, recordings)
                                record_pair(match, directory)    -> both sides
                                record_episode(model, seed, path) -> one side

md_app --match match.json                 # a manifest, with names and scores
md_app --match-left a.mdr --match-right b.mdr   # ad hoc, no claimed scores
```

`record_episode` sets the seed with `VecEnv.reset_seeds`, not the constructor's
`seed=` — the constructor's is a *starting point* the environment derives
per-episode seeds from, and a match is only a match if both sides played the
seed the manifest claims. The seed lands in the recording's header, which is
what the game checks before it will pair two files.

**Same seed, or it is refused.** `md::replay::MatchPlayer` compares the two
headers and throws rather than opening a window. Two agents on two different
problems, drawn side by side, is not a comparison — and it looks exactly like
one, which is why the refusal is in the loader and not in a caller's discipline.

**One transport, never two.** `tick`, `seek` and `restart` move both sides or
neither, and neither `Player` is reachable in a way that lets a caller advance
one alone. Two players on two timers drift within seconds, and tick 900 beside
tick 913 is two videos.

**Unequal endings are the interesting case.** One agent dying at wave 9 while
the other reaches wave 14 is what the comparison is *for*. The shorter side
freezes on its final state while the clock keeps running, so both stay on the
same tick number even when only one is still doing anything.

| Test | What it proves | Where |
|---|---|---|
| `test_match.cpp` | the transport keeps both sides on one tick, however it is driven | one process |
| `test_tournament.py` | a pair is only recorded on a seed the match was played on | one process |
| `tests/e2e/test_match.py` | **the real binary plays a real manifest as one screen, and refuses the rest** | two processes, three files |

## 9. Status

| Piece | State |
|---|---|
| `Sim`, `Action`, player model | done |
| `md/observation.hpp`, `md/intercept.hpp` | done |
| Scripted baseline + evaluation protocol (`md::agent`, M4) | done — `poe eval` |
| nanobind bindings and batched Python `VecEnv` (M5) | done |
| PPO training, split evaluation and checkpoints (M6) | done — `poe train` |
| Episode recording / replay / takeover (M3) | done |
| Native in-game inference for a learned checkpoint (M7) | planned |

The fixed deterministic seed stream is split before any policy is selected.
Routine learned-policy evaluation uses offset 0, count 32 as validation and may
choose `policy-best.pt`; the final benchmark uses the disjoint held-out block at
offset 32, count 32. That canonical run is pinned to 4 ticks per decision,
exactly 120,000 ticks, and CPU inference. Both drivers feed the same C++
episode records into the same `summarize` implementation, so "beat the
baseline" remains a concrete claim without using the test set to choose a model.
Current held-out baseline: mean score 13,687.28125 (8,040–20,270), mean wave
7.16, 0/6 cities surviving, and 0.73 kills per interceptor, all under the
published handicap
([ROADMAP.md](ROADMAP.md#m4--algorithmic-reference-ai--implemented--ready-for-sign-off)).
