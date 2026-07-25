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

## M3 — Record & replay in the UI

Record every run as `(seed, action-log)`; load a run and replay it with pause / scrub /
variable speed; **take over** from any point and continue playing.

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
| Mean score | **15,583** (range 9,790 – 18,510) |
| Mean wave reached | **14.4** |
| Mean cities surviving | **0.00** of 6 |
| Kills per interceptor | **1.10** |
| Episodes surviving the cap | 0 / 32 |

**This settles the question the design turned on.** A perfect-marksmanship agent — one
that solves the lead-intercept exactly and never misses — still loses *every* game, with
every city gone, around wave 14. The difficulty is not aiming; it is allocation under an
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
