# Windows

**Windows is MSVC, throughout.** The Qt/Vulkan game, the `_md_native` extension,
the tests and the trainer all build with one compiler in one tree.

It used to be two. The game was built with [MSYS2](https://www.msys2.org/)'s
CLANG64 environment and the extension with MSVC, because **torch and PySide6
publish no MinGW wheels** — there is no distribution for that platform tag at
all — and an extension module has to share an ABI with the interpreter importing
it. That meant two toolchains, two Pythons, two ABIs, a separate build preset
for each, and a `PATH` fixup so a MinGW-built game could find its runtime when
the trainer launched it. One compiler removes all of it.

Everything that is not platform-specific — how to play, what the scripted AI is,
the project layout — is in the [main README](../README.md). This page is only the
Windows delta.

## Quick start

```powershell
# 1 — the toolchain
winget install -e --id Microsoft.VisualStudio.2022.BuildTools `
  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"
winget install -e --id Kitware.CMake
winget install -e --id Ninja-build.Ninja
winget install -e --id KhronosGroup.VulkanSDK

# 2 — Qt. Not on winget, so via aqtinstall, which is what CI uses too.
#     qtbase only: the app links Qt6::Gui and nothing else.
pip install aqtinstall
python -m aqt install-qt windows desktop 6.9.3 win64_msvc2022_64 --outputdir C:\Qt
$env:QT_ROOT_DIR = "C:/Qt/6.9.3/msvc2022_64"

# 3 — build
git clone https://github.com/JensKSP/missile-defense.git
cd missile-defense
cmake --preset release
cmake --build --preset release

# 4 — play
.\build\release\app\md_app.exe
```

`QT_ROOT_DIR` is what the presets read as `CMAKE_PREFIX_PATH`, so no path is
baked into any of them (it is empty, and harmless, off Windows). Set it permanently with
`setx QT_ROOT_DIR C:/Qt/6.9.3/msvc2022_64`, or pass `-DCMAKE_PREFIX_PATH=…`.

**Add `%QT_ROOT_DIR%\bin` to `PATH` to run from the build tree**, or the exe
will not find `Qt6Gui.dll`. An *installed* copy needs nothing: `windeployqt`
puts Qt beside the binary — see [The installer](#the-installer).

From here the [main README](../README.md#quick-start) applies unchanged — play
it, then `.\build\release\app\md_app.exe --watch` to hand the crosshair to the
scripted agent.

## What the pieces are

The set maps onto the Debian dependencies in the
[main README](../README.md#requirements):

| Windows | Debian equivalent | Purpose |
|---|---|---|
| VS Build Tools 2022 (VCTools) | `g++` | C++23 compiler and linker |
| `Kitware.CMake`, `Ninja-build.Ninja` | `cmake ninja-build` | Build system |
| Qt 6 `qtbase`, MSVC 2022 x64 | `qt6-base-dev` | GUI toolkit |
| `KhronosGroup.VulkanSDK` | `libvulkan-dev`, `glslang-tools`, `vulkan-validationlayers` | **Four in one:** headers, loader import library, `glslangValidator` and the validation layers |

miniaudio and Catch2 have no Windows package, so CMake fetches both — exactly as
it does on Linux when `libminiaudio-dev` is absent. Nothing is vendored in-tree.

The Vulkan SDK is the reason this list is short: on MSYS2 those were four
separate `pacman` packages.

## Build notes

* **The presets are the same ones every platform uses** — `debug`, `release`,
  `profile`, `coverage`. What differs between toolchains lives in
  [CMakeLists.txt](../CMakeLists.txt), which already branches on MSVC, rather
  than in a parallel set of Windows-only preset names that every tool and `poe`
  task would then have to know about.
* One tree builds **both halves**: `MD_BUILD_APP` and `MD_BUILD_BINDINGS` are
  both on. With one ABI there is no reason to separate them.
* **ASan/UBSan are not enabled on Windows**, so `debug` degrades to a plain
  Debug build there with the Vulkan validation layer on. Sanitizer coverage is a
  Linux-only property of this project.
* **Warnings are not errors here.** The Linux gate holds the zero-warning line
  on a pinned compiler; MSVC's warning set differs enough that gating on it
  would mean tuning the tree for a second compiler with no extra safety.

> **Determinism.** `-ffp-contract=off` is what makes replays bit-identical, and
> `cl.exe` has no exact equivalent. The MSVC build has been checked against the
> golden trajectory checksum and matches — `ctest --preset release -L e2e` is
> what says so.

## The installer

`cpack --config build/release/CPackConfig.cmake -B build/release -G "NSIS;ZIP"`
builds both shipped Windows artifacts: an NSIS installer and a portable ZIP of
the same tree. CI does exactly this, and the release attaches both.

The install step runs **`windeployqt --compiler-runtime`**, which copies Qt's
DLLs and plugins and the C runtime beside the exe, so an installed copy starts
from Explorer with nothing on `PATH`. That used to be a 50-line shell script:
`windeployqt` handled Qt, and everything else — libc++, ICU, HarfBuzz, FreeType,
glib, PCRE2, zlib — had to be found by walking the dependency graph to a fixpoint
inside the CLANG64 prefix, because a MinGW build drags its toolchain's runtime
with it. An MSVC build has no such tail.

`vulkan-1.dll` is deliberately **not** bundled: the loader ships with the OS and
the GPU driver and has to match them.

The installer ships the game and, beside it, the **trainer's wheel** — built by
the `wheels` job with cibuildwheel against a real CPython and import-tested
there, so the file that ships is the file that was tested. It is a stable-ABI
(`abi3`) `.pyd` inside, so it works on 3.12 and later rather than on one exact
minor version.

What that means if you install it: have a **python.org CPython 3.12+** on PATH.
You do not need to `pip install` anything yourself — TRAIN AI does it, from the
wheel that came with the game, into the interpreter it found. It reports which
one it picked before it starts, because a silent choice on a machine with several
Pythons is impossible to correct afterwards.

## The Python half

Same shell, same compiler. From **PowerShell**, with a
[python.org](https://www.python.org/downloads/) CPython 3.12+:

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
drives CMake itself. `poe bindings` goes further: CMake's Ninja generator needs
`cl.exe` on `PATH`, so [tools/build_bindings.py](../tools/build_bindings.py)
runs `vcvars64.bat`, captures what it sets and adopts it — entering the
developer environment instead of telling you to open a different window.

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

The rest of training is platform-independent: see [TRAINING.md](TRAINING.md).

> **Building the bindings against a different interpreter.** `poe bindings`
> targets the one running it. To aim it elsewhere:
> `poe bindings -- windows --python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"`.

> **One ABI now.** The trainer used to prepend `<msys2>\clang64\bin` to `PATH`
> before launching the game, because a recording double-clicked in the trainer
> started a MinGW-built exe from a native process and it died looking for
> `libc++.dll` with no window to say so. `launch_environ` imposes nothing on any
> platform now.

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

## GPU

Vendor notes for training are in [TRAINING.md](TRAINING.md#gpu) — the ROCm and
DirectML caveats there are Windows-specific and worth reading before you buy into
a plan that needs a GPU.
