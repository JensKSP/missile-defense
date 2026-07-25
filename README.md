# Missile Defense

A faithful clone of Atari's **Missile Command** (1980), built as a personal
project for learning AI / machine learning. The same deterministic C++
simulation is played by humans (Qt 6 + Vulkan) and — as a headless, fast,
reproducible environment — used to train a reinforcement-learning agent.

![A MIRV splitting mid-descent over the cities, with interceptor trails and a fireball](docs/images/gameplay.png)

*By Jens Köhler · [MIT License](LICENSE) · developed with [Claude Code](https://claude.com/claude-code) (Anthropic).*

- Game design (and reward spec): [docs/DESIGN.md](docs/DESIGN.md)
- Milestones / roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Testing & quality gate: [docs/TESTING.md](docs/TESTING.md)
- Training the agent (M6): [docs/TRAINING.md](docs/TRAINING.md)

## Features

- **Faithful gameplay** — waves of ICBMs, splitting **MIRVs** (into re-entry
  warheads), blast-dodging **smart bombs**, three ammo-limited batteries, six
  destructible cities, bonus cities, and a rising difficulty curve.
- **Vulkan renderer** — instanced quads under an orthographic world→screen
  projection: rocket trails, glow, dangerous fireball explosions, distinct
  shapes per threat type, an animated twinkling starfield, and a pixel HUD/menu.
- **Procedural audio** — retro SFX *and* a looping FM-synth soundtrack, all
  generated in code (no asset files), driven by the core's deterministic event
  stream (which will also give the AI observation parity).
- **Full arcade shell** — menu, pause, help, **options** (audio / music /
  fullscreen), and a persistent **top-10 highscore** table with arcade initials
  entry.
- **Deterministic core** — fixed-timestep, `-ffp-contract=off`, seed + action
  replays are bit-identical (Debug == Release), gated by a golden checksum test.
- **Zero-warning, tested** — `-Werror`, strict clang-tidy, ruff + mypy, and
  ≥ 80 % core line coverage — all enforced by one `poe check` gate.

| | |
|:---:|:---:|
| ![The title menu, drawn in the game's own pixel font](docs/images/menu.png) | ![Interceptor blasts expanding over the skyline](docs/images/intercept.png) |
| **Full arcade shell** — menu, options, help, highscores | **Interceptors** — travel time, then an expanding blast |

## Requirements

Built and tested on Debian (trixie); adjust package names for other distros. It
also builds and runs on **Windows** via MSYS2 (see [Windows](#windows-msys2-clang64) below).

### Required — to build and run the game

| Purpose | Packages |
|---|---|
| C++23 compiler | `clang-21 lld-21` *(or any C++23 compiler — see note below)* |
| Build system | `cmake` (≥ 3.25), `ninja-build` |
| GUI toolkit | `qt6-base-dev qt6-base-dev-tools` |
| Vulkan (dev + loader) | `libvulkan-dev` |
| Vulkan driver | `mesa-vulkan-drivers` *(or your GPU vendor's Vulkan driver)* |
| Shader compiler | `glslang-tools` (provides `glslangValidator`) |
| Audio (single-header) | `libminiaudio-dev` *(else fetched at build; see note)* |

```bash
sudo apt update
sudo apt install clang-21 lld-21 cmake ninja-build \
  qt6-base-dev qt6-base-dev-tools \
  libvulkan-dev glslang-tools mesa-vulkan-drivers \
  libminiaudio-dev
```

> **Audio note:** miniaudio is a single-header library. The build prefers the
> system copy (`libminiaudio-dev`); if it is absent it fetches it via CMake
> (disable with `-DMD_FETCH_MINIAUDIO=OFF`). Nothing is vendored in-tree.

> **Compiler note:** the CMake preset pins **clang-21** (Debian package). To use
> a different compiler, edit `CMakePresets.json` (the `CMAKE_CXX_COMPILER` cache
> variable) or pass `-DCMAKE_CXX_COMPILER=<your-c++23-compiler>`.

### Optional — development, tests, and the task runner

| Purpose | Packages / tools |
|---|---|
| Task runner + Python tooling | `python3 python3-pip python3-venv` then `pip install poethepoet ruff pytest mypy pyright` |
| Vulkan validation (debugging) | `vulkan-validationlayers`, `vulkan-tools` (for `vulkaninfo`) |
| Editor tooling | `clangd-21 clang-format-21 clang-tidy-21` |
| Coverage | `llvm-21` (provides `llvm-cov-21`, `llvm-profdata-21`) |
| Debian package build | `cpack` (ships with `cmake`), `dpkg-dev` |
| Screenshot / video capture | `imagemagick` (`import`), `ffmpeg`, `xdotool` |

```bash
sudo apt install python3 python3-pip python3-venv \
  vulkan-validationlayers vulkan-tools \
  clangd-21 clang-format-21 clang-tidy-21 llvm-21 \
  dpkg-dev imagemagick ffmpeg xdotool
```

### Windows (MSYS2 CLANG64)

The same Clang + CMake + Ninja build runs on Windows through
[MSYS2](https://www.msys2.org/)'s **CLANG64** environment — no Visual Studio
needed. Install MSYS2 (`winget install -e --id MSYS2.MSYS2`), then from the
**MSYS2 CLANG64** shell:

```bash
pacman -Syu                       # if it closes the shell, reopen CLANG64 and repeat
pacman -S --needed \
  mingw-w64-clang-x86_64-clang \
  mingw-w64-clang-x86_64-lld \
  mingw-w64-clang-x86_64-cmake \
  mingw-w64-clang-x86_64-ninja \
  mingw-w64-clang-x86_64-qt6-base \
  mingw-w64-clang-x86_64-vulkan-headers \
  mingw-w64-clang-x86_64-vulkan-loader \
  mingw-w64-clang-x86_64-vulkan-validation-layers \
  mingw-w64-clang-x86_64-glslang
```

This set maps onto the Debian deps above: `clang`+`lld` (compiler), `qt6-base`
(Qt 6), `vulkan-headers`+`vulkan-loader` (Vulkan loader/headers), `glslang`
(`glslangValidator`). miniaudio has no MSYS2 package, so it is fetched at build
time just like on Linux. Then build and run with the same CMake commands as below
(the binary is `build/release/app/md_app.exe`) — **from the CLANG64 shell**, so
Qt/Vulkan DLLs resolve on `PATH`.

> **Windows notes:** the presets auto-detect Clang here (the `clang++-21` pin is
> Linux-only), so `cmake --preset release` / `debug` work unchanged. ASan/UBSan
> are not enabled on Windows (no MinGW LeakSanitizer runtime), so the `debug`
> preset builds as a plain Debug build. Optional dev tools:
> `mingw-w64-clang-x86_64-clang-tools-extra` (clang-tidy / clang-format),
> `mingw-w64-clang-x86_64-vulkan-tools` (`vulkaninfo`).

## Build & run

### With `poe` (recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install poethepoet ruff pytest mypy pyright

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
| Arrow keys / W,S + Enter | Navigate the menu (mouse works too — hover + click) |
| `Esc` | Pause → menu (resume with `Esc` or the RESUME item) |
| `P` | Pause → menu |
| `F` | Toggle fullscreen |
| `M` | Toggle music |
| `A` | Toggle audio (SFX) |

Menu: **START** a new game, **WATCH AI**, **HELP**, **OPTIONS** (audio / music /
fullscreen), **HIGHSCORES**, **ABOUT**, **EXIT**. Beat a high score to enter your
initials, arcade style.

### Watching the AI play

**WATCH AI** hands the controls to the scripted baseline agent (M4) and lets you
watch it defend — same game, same crosshair travel and trigger interval a hand is
held to, just a different driver. `md_app --watch` boots straight into it.

| Input | Action |
|---|---|
| `T` | Take over — you get the crosshair, the game continues from there |
| `]` / `[` | Fast-forward 1× → 8× (audio mutes above 1×) |
| `Esc` | Pause → menu |

Because both the simulation and the agent are deterministic, watching seed *N* is
bit-identical to the run `poe eval` measured for that seed — no recording needed,
the seed alone reproduces it.

### Watching a recorded run

Pick one from the **REPLAYS** menu entry, which lists what is in `runs/` (newest
first), or name it directly:

```bash
md_app --replay runs/update-1200.mdr
```

A *learned* policy is not reproducible from a seed the way the scripted agent is, so
training runs record episodes instead — from Python, `env.record(0)` then
`env.save_recording(0, path, update=n, label=f"update-{n}")`. What is stored is the
action index per agent step, four bytes each, so an episode is ~80 kB and can be
dropped every few updates to watch the policy improve.

| Input | Action |
|---|---|
| `[` / `]` | Fast-forward 1x -> 8x |
| Left / Right | Seek back / forward 5 s |
| `R` | Restart the recording |
| `T` | Take over from where it has reached |

## Development

```bash
poe test        # C++ unit + e2e tests (Debug)
poe coverage    # md::core line coverage; fails under 80% (currently ~96%)
poe check       # full local gate: format, lint, types, tidy, tests, coverage
```

The project enforces a **zero-warning** policy (`-Werror`, clang-tidy as errors,
ruff, mypy) plus a **≥ 80 % coverage** gate on the core. See
[docs/TESTING.md](docs/TESTING.md).

Handy tasks: `poe build`, `poe test-unit`, `poe test-release`, `poe shot`
(screenshot the running app), `poe rec` (record video), `poe format`.

## Building a Debian package

```bash
poe deb         # -> build/release/missile-defense_<version>_<arch>.deb
sudo apt install ./build/release/missile-defense_*.deb   # installs `missile-defense`
```

The package installs the game to `/usr/games/missile-defense` with a desktop
entry, and declares its runtime dependencies (Qt 6, the Vulkan loader). It is
produced by CPack's DEB generator directly from the CMake build.

## Project layout

| Path | Contents |
|---|---|
| `core/` | Pure C++ simulation library (`md::core`) + tests — no Qt, no rendering |
| `app/` | Qt 6 + Vulkan human client (renderer, input, HUD, menu) |
| `bindings/` | Python bindings (nanobind) — *planned* |
| `python/` | Gymnasium env + RL training — *planned* |
| `docs/` | Design spec, roadmap, testing |
| `tools/` | Cross-platform Python dev tooling (coverage, format/tidy, capture) |

## License & credits

Copyright © 2026 Jens Köhler. Released under the [MIT License](LICENSE).

This game was designed and written by Jens Köhler and **developed with
[Claude Code](https://claude.com/claude-code)** (Anthropic) as an AI pair
programmer. Audio uses [miniaudio](https://miniaud.io/) (public domain / MIT-0)
via the system package or CMake fetch — nothing is vendored. *Missile Command*
is a trademark of Atari; this is an independent, non-commercial homage.
