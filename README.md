# Missile Defense

A faithful clone of Atari's **Missile Command** (1980), built as a personal
project for learning AI / machine learning. The same C++ simulation is played by
humans (Qt + Vulkan), simulated headlessly for training, and — eventually —
controlled by a reinforcement-learning agent.

See [docs/DESIGN.md](docs/DESIGN.md) for the game specification (which doubles as
the reward spec) and `docs/ROADMAP.md` for the build order.

## Architecture

One simulation, three drivers:

```
            ┌──────────────────────────────────────────────┐
            │        core/   pure C++ sim (md::core)         │
            │  deterministic · headless · fixed timestep      │
            └───────┬───────────────┬───────────────┬────────┘
                    │               │               │
              app/ (Qt+Vulkan)  bindings/ (nanobind)   trained policy
              human plays       python/ RL training     (Step 5)
```

## Layout

| Path        | Contents                                                        |
|-------------|-----------------------------------------------------------------|
| `core/`     | Pure C++ simulation library + tests (no Qt, no rendering)        |
| `bindings/` | nanobind → Python module (Step 2)                               |
| `app/`      | Qt 6 + Vulkan human client / replay viewer (deferred)           |
| `python/`   | Gymnasium env + training (Step 2–3)                            |
| `docs/`     | Design spec, roadmap                                             |

## Toolchain

clang-21 · C++23 · CMake + Ninja · Qt 6 · Vulkan (glslangValidator) ·
Python (venv + pip + poe).

## Build

```bash
# one-time: dev tooling for tasks/lint
pip install poethepoet ruff pytest

poe configure   # cmake --preset debug   (clang-21, Ninja, ASan+UBSan)
poe build       # cmake --build --preset debug
poe test        # ctest --preset debug
```

Or drive CMake directly:

```bash
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
```
