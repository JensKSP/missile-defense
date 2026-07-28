# Windows

The game builds on Windows with the same Clang + CMake + Ninja toolchain as
Linux, through [MSYS2](https://www.msys2.org/)'s **CLANG64** environment — no
Visual Studio needed. Training is the one thing that needs a second, native
toolchain; see [Training on Windows](#two-shells-one-job-each) for why.

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
./build/release/app/missile-defense.exe
```

**Run it from the CLANG64 shell**, so the Qt and MSYS2 runtime DLLs resolve on
`PATH`. Launched from Explorer or PowerShell it will usually fail to load them —
see [Running outside the CLANG64 shell](#running-outside-the-clang64-shell) if
you want a bundle that does not need the shell.

If `git` is not on the path in that shell, `pacman -S git` installs it (that one
is an MSYS package, not a `mingw-w64-clang-x86_64-` one).

From here the [main README](../README.md#quick-start) applies unchanged — play
it, then `./build/release/app/missile-defense.exe --watch` to hand the crosshair to the
scripted agent.

**That is the game, and it is all MSYS2 is for.** If you also want the trainer,
the tests or the task runner, those live on a native CPython in PowerShell and
are a separate five minutes — see [Two shells, one job
each](#two-shells-one-job-each). You do not need MSYS2 for them, and they do not
need MSYS2 to be installed.

## What the packages are

The set maps onto the Debian dependencies in the
[main README](../README.md#requirements):

| MSYS2 package | Debian equivalent | Purpose |
|---|---|---|
| `clang`, `lld` | `g++` | C++23 compiler and linker — CLANG64 links with lld because that is its toolchain's default; the Linux build passes no `-fuse-ld` and takes the system linker |
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

* The presets take **the toolchain of the shell you are in** — the `clang++`
  preference in [CMakeLists.txt](../CMakeLists.txt) applies to Linux development
  builds only — so from CLANG64, `cmake --preset release` and `--preset debug`
  work unchanged.
* **ASan/UBSan are not enabled on Windows** (there is no MinGW LeakSanitizer
  runtime), so the `debug` preset builds as a plain Debug build. Sanitizer
  coverage is a Linux-only property of this project.

## Running outside the CLANG64 shell

The build links against Qt, libc++ and friends from `/clang64/bin`, which is why
the exe only starts from that shell. To make the build directory self-contained —
so it runs from Explorer, or on a machine with no MSYS2 at all:

```bash
tools/windeploy.sh build/release/app/missile-defense.exe
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
trainer* is offered unticked, because someone who came for an arcade game must
be able to decline an interpreter without reading a manual.

That component is the one thing here that is not self-contained, and the reason
is the ABI. The game is built with MSYS2/CLANG64, but the trainer is exec'd by
whatever `python` is on your PATH — a python.org CPython, because that is where
the PySide6 and torch wheels are, and a mingw-built extension cannot be loaded
by it. So the trainer's `_md_native` is built separately, with MSVC against a
python.org CPython, and it is that module the installer ships. It is a stable-ABI
(`abi3`) `.pyd`, so it works on 3.12 and later rather than on one exact minor
version.

What that means if you install it: have a **python.org CPython 3.12+** on PATH.
You do not need to `pip install` anything yourself — TRAIN AI does it, from the
wheel that came with the game, into the interpreter it found. It reports which
one it picked before it starts, because a silent choice on a machine with several
Pythons is impossible to correct afterwards.

## Two shells, one job each

Windows needs two toolchains, and the tidiest way to hold that in your head is
that each owns exactly one half:

| | Shell | Builds |
|---|---|---|
| **The game** | MSYS2 **CLANG64** | Qt, Vulkan, the app — `cmake --preset release` |
| **Everything Python** | **PowerShell**, native CPython | the venv, `poe`, the tests, `_md_native` |

The reason is the ABI, and it is not negotiable: **torch and PySide6 publish no
MinGW wheels** — there is no distribution for that platform tag at all — so the
trainer and the training loop can only run on an MSVC-built CPython. An extension
module has to share an ABI with the interpreter importing it, and the simulation
needs neither Qt nor Vulkan, so the headless half is built natively and the game
stays on MSYS2.

## The Python half, natively

From **PowerShell**, with a [python.org](https://www.python.org/downloads/)
CPython 3.12+ and **VS Build Tools** (C++ workload) installed:

```powershell
python -m tools.bootstrap
.venv\Scripts\Activate.ps1
poe pytest          # the fast suite
poe bindings        # rebuild _md_native after a C++ change
poe ui              # the trainer
poe train           # needs torch; bootstrap installs the CPU wheel
```

**No Developer Command Prompt needed.** `tools/bootstrap.py` finds MSVC through
`VCINSTALLDIR`, then `cl.exe` on `PATH`, then vswhere, and scikit-build-core
drives CMake itself — so a plain PowerShell is enough. MinGW deliberately does
not count as a toolchain here: an MSYS2 clang on `PATH` is the *wrong* compiler
for this job, not a substitute for the right one, and a binding it produced fails
to load in the interpreter that has torch.

Without MSVC, bootstrap says so and sets the rest up anyway — the trainer starts,
browses runs and plays replays; it refuses to *start* a run and says why.

`poe` is the venv's, so `poe train` and `poe ui` run the interpreter that has
torch and PySide6, and `MD_PYTHON` is not something you need to set. Both still
go through `tools/launch.py`, which looks for an interpreter that has them —
`$MD_PYTHON` first, then the running interpreter, then whatever `py -0p` knows
about — and puts this checkout's `python/` on its import path. That search is the
fallback for an installed trainer rather than the everyday path. When nothing on
the machine can run it, it names the interpreters it tried and the `pip install`
that would fix one, rather than failing with an ImportError from inside a module.

Both module ABIs can sit beside the package at once — each interpreter loads its
own — so the MSYS2 tooling and the training environment coexist. The rest of
training is platform-independent: see [TRAINING.md](TRAINING.md).

> **Building the bindings against a different interpreter.** `poe bindings`
> targets the one running it. To aim it elsewhere:
> `poe bindings -- win-native --python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"`.

Double-clicking a recording starts the MSYS2-built game from a
non-MSYS2 process, so the trainer puts `<msys2>\clang64\bin` back on `PATH` for
the child — otherwise it dies looking for `libc++.dll` with no window to say so.
Set `MSYS2_ROOT` if MSYS2 is not at `C:\msys64`.

### Reaching the trainer from an installed game

**TRAIN AI in the game's menu is the way in, and it is also how the trainer gets
installed.** The installer and the portable ZIP carry the trainer as a *wheel*
beside `missile-defense.exe` — not as a Python payload the game has to find.
Choosing TRAIN AI on a machine that has no trainer yet:

1. finds every interpreter on `PATH` and asks each its version;
2. offers to install into the newest one that is 3.12 or later;
3. on ENTER, opens a terminal running
   `pip install --user "<wheel>[trainer]"` so you can watch it;
4. writes the interpreter it used into `trainer.conf` in the game's data
   directory, so the next TRAIN AI starts the trainer instead.

**Nothing here is a script file**, and that is deliberate: Smart App Control
blocks unsigned scripts outright on a stock Windows 11. The installer used to
write a `missile-defense-trainer.cmd` that the policy simply refused to run — and
that the game had to route around by exec'ing the interpreter directly. There is
no `.cmd` any more; the game spawns `cmd.exe` with the pip command as arguments,
and `cmd.exe` is a system binary.

Python 3.12 and not 3.11: the wheel is `cp312-abi3`, and pip refuses it below
that. `HOW-TO-TRAIN.html`, which the game opens from the same screen, says so
with a link.

**No console window, anywhere along that path.** Three separate things arrange
that, because a console appears for three separate reasons:

* `missile-defense-trainer` is a `[project.gui-scripts]` entry, so pip builds
  the *pythonw* launcher for it rather than the console-subsystem one;
* the game starts the trainer through `pythonw.exe` beside the interpreter it
  resolved (`windowless_interpreter`, app/trainer.hpp — the Python side of the
  same lookup makes the same swap);
* everything the trainer itself starts — a run, a pip install into the managed
  runtime — is spawned with `CREATE_NO_WINDOW`
  (`missile_defense.runs.spawn`), since their output is already on a pipe and
  into the progress pane.

The cost of the first is that the trainer has no `stderr` when it is started
that way: `print` reaches nobody, silently. That is the one place this matters —
the entry point that explains a missing PySide6 — and it puts the sentence in a
message box instead (`missile_defense.ui.__main__.announce`).

**Nothing is written into the install directory.** The managed PyTorch runtime,
the runs and the models all live under `%LOCALAPPDATA%\MissileDefense`, so an
install in `C:\Program Files` never needs write access of its own.

## Screenshots

`poe shot` and `poe rec` work here — the backend is PowerShell plus
`Graphics.CopyFromScreen` for stills and ffmpeg's `gdigrab` for video, so
nothing extra is installed. `poe shot -- --launch` starts the game, waits for
its window, raises it and closes it again; `--title` aims the same command at
the trainer instead.

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
