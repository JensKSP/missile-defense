# Missile Defense

A faithful clone of Atari's **Missile Command** (1980), built as a personal
project for learning AI / machine learning. The same deterministic C++
simulation is played by humans (Qt 6 + Vulkan) and — as a headless, fast,
reproducible environment — used to train a reinforcement-learning agent.

![A MIRV splitting mid-descent over the cities, with interceptor trails and a fireball](docs/images/gameplay.png)

*By Jens Köhler · [MIT License](LICENSE) · developed with [Claude Code](https://claude.com/claude-code) (Anthropic).*

## Download

Prebuilt for Debian trixie, Ubuntu 24.04, Windows and macOS (Apple silicon):

| | |
|---|---|
| **[Latest release](https://github.com/JensKSP/missile-defense/releases/latest)** | versioned, and someone has looked at it |
| [Nightly](https://github.com/JensKSP/missile-defense/releases/tag/nightly) | the newest `master` that passed CI; nobody has played it |

Every build is checksummed in `SHA256SUMS`. The macOS bundle is ad-hoc signed
rather than notarised, so macOS asks you to clear quarantine first —
[docs/MACOS.md](docs/MACOS.md#signing-it-for-other-people) has the one command.
How the versions are numbered, and what a nightly's `0.1.0~dev128` means to
`apt`, is in [docs/RELEASING.md](docs/RELEASING.md#versioning).

## Quick start

Or build it — clone to watching an AI defend six cities, in about ten minutes.
On Debian / Ubuntu — for Windows see [docs/WINDOWS.md](docs/WINDOWS.md), for
macOS [docs/MACOS.md](docs/MACOS.md), and for other distros adjust the package
names using [Requirements](#requirements) below:

```bash
# 1 — dependencies (a few hundred MB: Qt 6, Vulkan, a C++23 compiler)
sudo apt update
sudo apt install -y g++-14 cmake ninja-build \
  qt6-base-dev qt6-base-dev-tools \
  libvulkan-dev glslang-tools mesa-vulkan-drivers libminiaudio-dev

# 2 — build (no Python needed, and no install step afterwards)
git clone https://github.com/JensKSP/missile-defense.git
cd missile-defense
# g++-14 explicitly: it has C++23 <print>, which Ubuntu 24.04's default g++-13
# does not, and CXX makes CMake use it instead of hunting for clang.
CXX=g++-14 cmake --preset release && cmake --build --preset release

# 3 — play
./build/release/app/md_app
```

**Play it.** The mouse aims, left click fires from the nearest battery with
ammo. Six cities, three batteries, and less ammunition than you would like.
→ [Full controls](#how-to-play)

**Watch the AI play it.** `./build/release/app/md_app --watch` boots straight
into a game driven by the scripted agent — held to the same crosshair speed and
trigger interval as your hand. `]` fast-forwards to 8×; `T` takes the controls
back mid-game. It averages **113,834** points and still loses every game around
wave 17, because this game is about spending ammunition, not about aiming.
→ [More](#watching-the-ai-play)

**Train one that beats it.** 113,834 is the number a learned policy has to beat,
scored on the same 32 seeds by the same code. → [docs/TRAINING.md](docs/TRAINING.md)

On **Windows** the same toolchain runs under MSYS2 — its own ten-minute path is
in [docs/WINDOWS.md](docs/WINDOWS.md). On **macOS** it is Homebrew and MoltenVK:
[docs/MACOS.md](docs/MACOS.md), which is built and tested in CI but — unlike the
other two — has never been run by a human.

Deeper reading: [design & reward spec](docs/DESIGN.md) ·
[the agent API](docs/API.md) · [milestones / roadmap](docs/ROADMAP.md) ·
[testing & quality gate](docs/TESTING.md) ·
[simulation throughput](docs/PERFORMANCE.md) ·
[training on an NVIDIA GPU](docs/NVIDIA.md) ·
[packaging & file locations](docs/PACKAGING.md)

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

Reference — the [quick start](#quick-start) above already covers the common
case. Read on for what each package is for and the optional development tools.

Built and tested on Debian (trixie); adjust package names for other distros.
**Windows** builds through MSYS2 with its own instructions in
[docs/WINDOWS.md](docs/WINDOWS.md), and **macOS** through Homebrew in
[docs/MACOS.md](docs/MACOS.md).

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

The `apt install` line for exactly these is in the
[quick start](#quick-start) above, so there is only one copy to keep current.

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
| Screenshot / video capture | `imagemagick` (`import`), `ffmpeg`, `xdotool` — on Linux/X11; Windows and macOS use what they ship with |

```bash
sudo apt install python3 python3-pip python3-venv \
  vulkan-validationlayers vulkan-tools \
  clangd-21 clang-format-21 clang-tidy-21 llvm-21 \
  dpkg-dev imagemagick ffmpeg xdotool
```

## Build & run

### With CMake directly (no Python needed)

What the [quick start](#quick-start) uses:

```bash
cmake --preset release
cmake --build --preset release
./build/release/app/md_app
```

There is no system install step — run the game from the build output above (or
copy `build/release/app/md_app` wherever you like; it has no data files, shaders
are baked into the binary).

### With `poe`

`poethepoet` wraps that same build together with the tests, linters, and
coverage gate behind one-word tasks — worth setting up as soon as you start
changing code:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install poethepoet ruff pytest mypy pyright

poe app        # build (Release) and launch the game
```

### Headless only

To build only the simulation + tests (no Qt/Vulkan):

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
(screenshot the running app), `poe rec` (record video), `poe format`, `poe ui`
(start, watch and stop a training run — see [docs/TRAINING.md](docs/TRAINING.md)).

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
| `docs/` | Design spec, roadmap, testing, training, Windows + macOS notes |
| `tools/` | Cross-platform Python dev tooling (coverage, format/tidy, capture) |

## License & credits

Copyright © 2026 Jens Köhler. Released under the [MIT License](LICENSE).

This game was designed and written by Jens Köhler and **developed with
[Claude Code](https://claude.com/claude-code)** (Anthropic) as an AI pair
programmer. Audio uses [miniaudio](https://miniaud.io/) (public domain / MIT-0)
via the system package or CMake fetch — nothing is vendored. *Missile Command*
is a trademark of Atari; this is an independent, non-commercial homage.
