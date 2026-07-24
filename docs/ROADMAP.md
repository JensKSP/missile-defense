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

## M4 — ML infrastructure

nanobind bindings; Gymnasium environment; `VecSim` (N parallel sims, thread pool, GIL
released, zero-copy batched observations); a scripted baseline agent; reward design and
an evaluation protocol. *(The headless sim is already sufficient to start this — it does
not depend on M2/M3.)*

## M5 — Train

Custom PPO (PyTorch) with a curriculum; beat the scripted baseline.

## M6 — Watch the AI / takeover

Export the policy → in-process C++ inference; live spectator mode; human takeover from any
point in an AI-played game.
