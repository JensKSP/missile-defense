# Roadmap

Milestones, ordered so that **a human can play first**. Each builds on the pure,
headless, deterministic `md::core` simulation.

## M1 — Human can play *(current target)*

A person plays a fresh game of Missile Command in the Vulkan UI: waves of descending
threats, three batteries with limited ammo, interceptors with travel time + expanding
blast, six cities to defend, score, and win/lose.

Path:
1. ✅ Core primitives — `Vec2`, `Pcg32` RNG — + TDD / zero-warning quality harness.
2. **Entity state model** — POD, fixed-capacity (`Config`, enums, entities, `Action`). *(in progress)*
3. **`Sim` core loop** — `reset(seed)` / `step(action)`, waves & spawning, motion,
   collisions (blast ↔ threat, threat ↔ city), scoring, termination. Ships with unit +
   e2e + **determinism** tests → the game is fully playable *headless*.
4. **Vulkan / Qt app** — `QVulkanWindow` device bring-up → 2D renderer (instanced quads
   for entities, a line/circle pipeline for trajectories & blasts) → fixed-timestep loop
   with render interpolation → input → `Action` + HUD (score / ammo / wave).

→ **M1 reached.**

## M2 — Record & replay in the UI

Record every run as `(seed, action-log)`; load a run and replay it with pause / scrub /
variable speed; **take over** from any point and continue playing.

## M3 — ML infrastructure

nanobind bindings; Gymnasium environment; `VecSim` (N parallel sims, thread pool, GIL
released, zero-copy batched observations); a scripted baseline agent; reward design and
an evaluation protocol.

## M4 — Train

Custom PPO (PyTorch) with a curriculum; beat the scripted baseline.

## M5 — Watch the AI / takeover

Export the policy → in-process C++ inference; live spectator mode; human takeover from any
point in an AI-played game.
