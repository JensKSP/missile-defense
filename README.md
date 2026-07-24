# Missile Defense

A faithful clone of Atari's **Missile Command** (1980), built as a personal
project for learning AI / machine learning. The same deterministic C++
simulation is played by humans (Qt 6 + Vulkan) and — as a headless, fast,
reproducible environment — used to train a reinforcement-learning agent.

- Game design (and reward spec): [docs/DESIGN.md](docs/DESIGN.md)
- Milestones / roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Testing & quality gate: [docs/TESTING.md](docs/TESTING.md)

## Requirements

Built and tested on Debian (trixie). Adjust package names for other distros.

### Required — to build and run the game

| Purpose | Packages |
|---|---|
| C++23 compiler | `clang-21 lld-21` *(or any C++23 compiler — see note below)* |
| Build system | `cmake` (≥ 3.25), `ninja-build` |
| GUI toolkit | `qt6-base-dev qt6-base-dev-tools` |
| Vulkan (dev + loader) | `libvulkan-dev` |
| Vulkan driver | `mesa-vulkan-drivers` *(or your GPU vendor's Vulkan driver)* |
| Shader compiler | `glslang-tools` (provides `glslangValidator`) |

```bash
sudo apt update
sudo apt install clang-21 lld-21 cmake ninja-build \
  qt6-base-dev qt6-base-dev-tools \
  libvulkan-dev glslang-tools mesa-vulkan-drivers
```

> **Compiler note:** the CMake preset pins **clang-21** (Debian package). To use
> a different compiler, edit `CMakePresets.json` (the `CMAKE_CXX_COMPILER` cache
> variable) or pass `-DCMAKE_CXX_COMPILER=<your-c++23-compiler>`.

### Optional — development, tests, and the task runner

| Purpose | Packages / tools |
|---|---|
| Task runner + Python tests | `python3 python3-pip python3-venv` then `pip install poethepoet ruff pytest mypy` |
| Vulkan validation (debugging) | `vulkan-validationlayers`, `vulkan-tools` (for `vulkaninfo`) |
| Editor tooling | `clangd-21 clang-format-21 clang-tidy-21` |
| Screenshot / video capture | `imagemagick` (`import`), `ffmpeg`, `xdotool` |

```bash
sudo apt install python3 python3-pip python3-venv \
  vulkan-validationlayers vulkan-tools \
  clangd-21 clang-format-21 clang-tidy-21 \
  imagemagick ffmpeg
```

## Build & run

### With `poe` (recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install poethepoet ruff pytest mypy

poe app        # build (Release) and launch the game
```

### With CMake directly (no Python needed)

```bash
cmake --preset release
cmake --build --preset release
./build/release/app/md_app
```

There is no system install step — run the game from the build output above (or
copy `build/release/app/md_app` wherever you like; it has no data files, shaders
are baked into the binary).

To build only the headless simulation + tests (no Qt/Vulkan):

```bash
cmake --preset release -DMD_BUILD_APP=OFF
cmake --build --preset release
```

## How to play

Defend your six cities from incoming missiles by launching interceptors from
your three batteries.

| Input | Action |
|---|---|
| Mouse | Move the crosshair (aim) |
| Left click | Fire from the nearest battery with ammo |
| Arrow keys / W,S + Enter | Navigate the menu |
| `Esc` | Pause → menu (resume with `Esc` or the RESUME item) |
| `P` | Pause → menu |

Menu: **START** a new game, **HELP**, **HIGHSCORES**, **EXIT**.

## Development

```bash
poe test        # C++ unit + e2e tests (Debug)
poe check       # full local gate: format, lint, types, tidy, tests (Debug + Release)
```

The project enforces a **zero-warning** policy (`-Werror`, clang-tidy as errors,
ruff, mypy). See [docs/TESTING.md](docs/TESTING.md).

Handy tasks: `poe build`, `poe test-unit`, `poe test-release`, `poe shot`
(screenshot the running app), `poe rec` (record video), `poe format`.

## Project layout

| Path | Contents |
|---|---|
| `core/` | Pure C++ simulation library (`md::core`) + tests — no Qt, no rendering |
| `app/` | Qt 6 + Vulkan human client (renderer, input, HUD, menu) |
| `bindings/` | Python bindings (nanobind) — *planned* |
| `python/` | Gymnasium env + RL training — *planned* |
| `docs/` | Design spec, roadmap, testing |
| `scripts/` | Screenshot / video capture helpers |
