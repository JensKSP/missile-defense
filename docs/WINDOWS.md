# Windows

The game builds on Windows with the same Clang + CMake + Ninja toolchain as
Linux, through [MSYS2](https://www.msys2.org/)'s **CLANG64** environment — no
Visual Studio needed. Training is the one thing that needs a second, native
toolchain; see [Training on Windows](#training-on-windows) for why.

Everything that is not platform-specific — how to play, what the scripted AI is,
the project layout — is in the [main README](../README.md). This page is only the
Windows delta.

## Quick start

```powershell
# 1 — MSYS2, from PowerShell
winget install -e --id MSYS2.MSYS2
```

Then from the **MSYS2 CLANG64** shell (not the MSYS or UCRT64 one):

```bash
# 2 — dependencies
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

# 3 — build
git clone https://github.com/JensKSP/missile-defense.git
cd missile-defense
cmake --preset release && cmake --build --preset release

# 4 — play
./build/release/app/md_app.exe
```

**Run it from the CLANG64 shell**, so the Qt and MSYS2 runtime DLLs resolve on
`PATH`. Launched from Explorer or PowerShell it will usually fail to load them —
see [Running outside the CLANG64 shell](#running-outside-the-clang64-shell) if
you want a bundle that does not need the shell.

If `git` is not on the path in that shell, `pacman -S git` installs it (that one
is an MSYS package, not a `mingw-w64-clang-x86_64-` one).

From here the [main README](../README.md#quick-start) applies unchanged — play
it, then `./build/release/app/md_app.exe --watch` to hand the crosshair to the
scripted agent.

## What the packages are

The set maps onto the Debian dependencies in the
[main README](../README.md#requirements):

| MSYS2 package | Debian equivalent | Purpose |
|---|---|---|
| `clang`, `lld` | `clang-21 lld-21` | C++23 compiler and linker |
| `cmake`, `ninja` | `cmake ninja-build` | Build system |
| `qt6-base` | `qt6-base-dev` | GUI toolkit |
| `vulkan-headers`, `vulkan-loader` | `libvulkan-dev` | Vulkan loader and headers |
| `vulkan-validation-layers` | `vulkan-validationlayers` | Optional, dev only |
| `glslang` | `glslang-tools` | `glslangValidator` |

miniaudio has no MSYS2 package, so it is fetched at build time exactly as it is
on Linux when `libminiaudio-dev` is absent. Nothing is vendored in-tree.

Optional dev tools: `mingw-w64-clang-x86_64-clang-tools-extra` (clang-tidy,
clang-format) and `mingw-w64-clang-x86_64-vulkan-tools` (`vulkaninfo`).

## Build notes

* The presets **auto-detect Clang** here — the `clang++-21` pin is Linux-only —
  so `cmake --preset release` and `--preset debug` work unchanged.
* **ASan/UBSan are not enabled on Windows** (there is no MinGW LeakSanitizer
  runtime), so the `debug` preset builds as a plain Debug build. Sanitizer
  coverage is a Linux-only property of this project.

## Running outside the CLANG64 shell

The build links against Qt, libc++ and friends from `/clang64/bin`, which is why
the exe only starts from that shell. To make the build directory self-contained —
so it runs from Explorer, or on a machine with no MSYS2 at all:

```bash
tools/windeploy.sh build/release/app/md_app.exe
```

It runs `windeployqt6` for Qt's own DLLs and plugins, then walks the dependency
graph to a fixpoint copying everything else that resolves inside the CLANG64
prefix — which `windeployqt` does not do here. DLLs land next to the exe, in
place.

`vulkan-1.dll` is deliberately **not** bundled: the Vulkan loader ships with the
OS and GPU driver and has to match them.

## The installer

`cpack --config build/release/CPackConfig.cmake -B build/release -G "NSIS;ZIP"`
builds both shipped Windows artifacts: an NSIS installer and a portable ZIP of
the same tree. CI does exactly this, and the release attaches both.

The installer offers **two** components. The game is required; the *Training
console* is offered unticked, because someone who came for an arcade game must
be able to decline an interpreter without reading a manual.

That component is the one thing here that is not self-contained, and the reason
is the ABI. The game is built with MSYS2/CLANG64, but the console is exec'd by
whatever `python` is on your PATH — a python.org CPython, because that is where
the PySide6 and torch wheels are, and a mingw-built extension cannot be loaded
by it. So the console's `_md_native` is built separately, with MSVC against a
python.org CPython, and it is that module the installer ships. It is a stable-ABI
(`abi3`) `.pyd`, so it works on 3.12 and later rather than on one exact minor
version.

What that means if you install it: have a **python.org CPython 3.12+** on PATH
and `pip install PySide6` into it. `md-console.cmd` checks before it execs and
says so if either is missing, rather than opening the Microsoft Store — which is
what a bare `python` does on a machine that has none.

## Training on Windows

The Clang presets build under MSYS2/MinGW, and **torch publishes no MinGW
wheel** — there is no distribution for that platform tag at all. Only the
*extension module* has to share an ABI with the interpreter importing it, and the
simulation needs neither Qt nor Vulkan, so build the headless half natively and
leave the game on MSYS2:

1. Install **VS Build Tools** (C++ workload) and a **python.org CPython**.
2. From a Developer Command Prompt:

   ```bash
   poe bindings -- win-native --python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
   ```

3. `pip install torch` into that interpreter, then `poe train`.

Both module ABIs can sit beside the package at once — each interpreter loads its
own — so the MSYS2 tooling and the training environment coexist. The rest of
training is platform-independent: see [TRAINING.md](TRAINING.md).

The **training console** (`poe ui`) belongs to that same native interpreter —
install the project with its console extra there (`pip install -e ".[console]"`),
because PySide6's wheels are MSVC-built like torch's. That extra also includes
the NVIDIA telemetry binding; AMD SMI is Linux-only.

`poe` itself is usually the MSYS2 one, so `poe train` and `poe ui` would run the
interpreter that has neither package. Both go through `tools/launch.py`, which
looks for one that does — `$MD_PYTHON` first, then the running interpreter, then
whatever `py -0p` knows about — and puts this checkout's `python/` on its import
path. When nothing on the machine can run it, it names the interpreters it tried
and the `pip install` that would fix one, rather than failing with an ImportError
from inside a module. Set `MD_PYTHON` to skip the search.

Double-clicking a recording starts the MSYS2-built game from a
non-MSYS2 process, so the console puts `<msys2>\clang64\bin` back on `PATH` for
the child — otherwise it dies looking for `libc++.dll` with no window to say so.
Set `MSYS2_ROOT` if MSYS2 is not at `C:\msys64`.

### Reaching the console from an installed game

**TRAIN AI in the game's menu is the way in.** The installer and the portable
ZIP both put the console's payload — `md\ui\` — beside `md_app.exe`, and the
game looks there: with a usable `python` on `PATH` the entry appears, and
choosing it runs `python -m md.ui` with that directory on the import path.

What the game deliberately does *not* use is the `md-console.cmd` sitting in
the same folder. **Smart App Control blocks unsigned scripts** on a stock
Windows 11, so where that policy is on, the `.cmd` cannot be run at all —
neither by the game nor by anyone double-clicking it. It stays for machines
without the policy; nothing depends on it.

Until 2026-07-27 there was no way in at all. The lookup asked for
`md-console.exe` — the wrong extension for a file called `md-console.cmd` — and
searched `PATH` plus two Unix directories, never the one the installer actually
writes to. Every Windows install resolved to nothing and the menu offered no
training, with the console unreachable in the same folder as the binary that
could not find it.

**Nothing is written into the install directory.** The managed PyTorch runtime,
the runs and the models all live under `%LOCALAPPDATA%\MissileDefense`, so an
install in `C:\Program Files` never needs write access of its own.

## Screenshots

`poe shot` and `poe rec` work here — the backend is PowerShell plus
`Graphics.CopyFromScreen` for stills and ffmpeg's `gdigrab` for video, so
nothing extra is installed. `poe shot -- --launch` starts the game, waits for
its window, raises it and closes it again; `--title` aims the same command at
the training console instead.

Two things to know, because both give you a *plausible but wrong* picture rather
than an error. **Capture in windowed mode** — a fullscreen Vulkan swapchain
bypasses the compositor, so a screen grab of one is the desktop behind it, and
`poe shot` warns when the window fills the screen. And the pixels come from the
screen, so **a notification that arrives mid-grab is in the file**. Look at the
image before you believe it.

> **Determinism.** `-ffp-contract=off` is what makes replays bit-identical, and
> `cl.exe` has no exact equivalent (`clang-cl` passes it through). The MSVC build
> has been checked against the golden trajectory checksum and matches, but if you
> change compilers, run `ctest --preset win-native -L e2e` before trusting a
> recording it produced.

## GPU

Vendor notes for training are in [TRAINING.md](TRAINING.md#gpu) — the ROCm and
DirectML caveats there are Windows-specific and worth reading before you buy into
a plan that needs a GPU.
