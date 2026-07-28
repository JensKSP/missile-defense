# Packaging

What ships, where it lands, and where a run's files go. The Debian package is the
reference — it is the one with rules about all three — but the file locations
below are what the game and the trainer do on every platform.

## What ships today

**Three binary packages from one source package.** The split described further
down has landed; `debian/*.install` is the authority for which file goes where.

| Package | Path | What |
|---|---|---|
| `missile-defense` | `/usr/games/missile-defense` | the game (renamed from `md_app`) |
| | `/usr/share/missile-defense/models/` | the bundled learned policy (`.mdp`) |
| | `/usr/share/applications/missile-defense.desktop` | menu entry |
| | `/usr/share/icons/hicolor/…` | icons the desktop entry resolves |
| | `/usr/share/man/man6/missile-defense.6.gz` | man page (section 6: games) |
| | `/usr/share/doc/missile-defense/…` | licence and third-party notices |
| `missile-defense-trainer` | `/usr/bin/missile-defense-trainer`, `/usr/bin/missile-defense-train` | the trainer window and the training command |
| | `/usr/lib/python3*/dist-packages/missile_defense/` | the Python package and its native extension |
| | `/usr/share/applications/missile-defense-trainer.desktop` | menu entry for the trainer |
| | `/usr/share/icons/hicolor/…/missile-defense-trainer.png` | the trainer's own emblem, which that entry resolves |

The game package still contains **no Python at all** — that is the boundary the
split exists to make into a packaging fact rather than a rule people remember,
and `test_packages.py` holds it.

There is one runtime data directory, and only one: `/usr/share/missile-defense/models/`,
holding the bundled `.mdp` policy the game plays natively with no interpreter in
the process. The SPIR-V shaders are still compiled into the executable at build
time (`app/CMakeLists.txt`), so nothing about *rendering* is loaded from disk.
The model is optional — a build without one ships without one, and the game omits
MODELS from its WATCH AI menu rather than offering an empty list.

Two build paths exist and that is deliberate: `debian/` (debhelper) is the
authority for Linux, and CPack produces the Windows NSIS installer and ZIP, plus
the macOS disk image. `poe deb` runs CPack's DEB generator, which is a
convenience for testing a local package — it cannot express the multi-binary
split, so `debian/` is where that lives.

On macOS, `poe dmg` builds a drag-to-Applications image containing `Missile Defense.app`
with the Qt frameworks *and* MoltenVK inside it, so it depends on nothing but
macOS. It is ad-hoc signed by default, which runs but is not distributable;
[MACOS.md](MACOS.md#signing-it-for-other-people) covers the Developer ID and
notarisation path.

### Every artifact is built on every push

[.github/workflows/ci.yml](../.github/workflows/ci.yml) builds all four on the
platform each is meant for, and uploads them:

| Job | Platform | Produces |
|---|---|---|
| `gate` | GitHub's current Ubuntu LTS runner | the quality gate, plus the CPack `.deb` |
| `debian` | Ubuntu 26.04 (primary), Debian trixie, Ubuntu 24.04 (compatibility) | the **debhelper** `.deb`s from `debian/`, lintian-checked |
| `windows` | Windows / MSYS2 CLANG64 | NSIS installer + portable ZIP |
| `macos` | macOS on Apple silicon | the `.dmg` |

`gate` and `debian` both produce Debian packages on purpose. They are different
code paths — CPack's DEB generator against `cmake --install`, versus debhelper's
`dh_auto_*` with `hardening=+all` and no `-Werror` — and only the second is what
Debian would build. One can break while the other still works.

The packaging steps are in CI because their failure mode is invisible locally: a
DLL that resolves only because MSYS2 is on the machine's PATH, a Qt framework
that loads only because Homebrew installed it. Building on a runner that has
none of a developer's incidental state is the check.

## Where a run's files go

Everything a run writes — `metrics.csv`, `evals.csv`, `config.json`,
`model.json`, `train.log`, the `.mdr` recordings, `checkpoints/` — goes in **one
directory**, chosen by this rule
(`python/missile_defense/runs/paths.py`, mirrored in `app/game_window.cpp`):

1. an explicit `--out-dir`, or the trainer's run picker;
2. `$MD_RUNS_DIR`;
3. `./runs`, **if that directory already exists**;
4. otherwise the per-user data directory:

| Platform | Default |
|---|---|
| Linux | `$XDG_DATA_HOME/MissileDefense/runs` (`~/.local/share/…`) |
| Windows | `%LOCALAPPDATA%\MissileDefense\runs` |
| macOS | `~/Library/Application Support/MissileDefense/runs` |

Rule 3 is what keeps a checkout behaving exactly as it always has, without a
build-time switch: the same binary does the obvious thing in a source tree and in
`/usr/games` without being told which one it is in.

Two things live *beside* the runs rather than in one, because both must outlive
the run that produced them: promoted models in `models/`, and the trainer's saved
training presets in `presets.json` (`$MD_MODELS_DIR` and `$MD_PRESETS_FILE`
override). The presets file is a small, indented JSON list meant to be opened in
an editor and copied between machines; a missing or damaged one costs you the
saved names and nothing else.

**Why data and not state or cache.** `~/.local/state` is for things you would
shrug at losing and `~/.cache` for things that regenerate. A checkpoint is the
output of hours of compute and regenerates only by spending them again, so it
belongs with the user's own files.

**Why `MissileDefense` and not `missile-defense`.** The game already keeps its
high scores there — it is `QGuiApplication::setApplicationName` in
`app/main.cpp`, and Qt derives the path from it. Runs join them rather than
creating a second directory and a migration for the first. A test asserts the
Python constant and the C++ call still agree.

**Local, not roaming, on Windows.** High scores are a few hundred bytes and use
`AppDataLocation`; a run directory is checkpoints, and syncing those onto a
domain profile would be a surprise measured in gigabytes.

Nothing is written outside the user's own directories, so there is no
`/var/games`, no setgid binary, and no shared score file to corrupt — high scores
are per-user, which is both the modern convention and one fewer attack surface.

**Recordings are not archival.** A `.mdr` embeds the simulation `Config` it was
recorded with and is checked against the build that reads it (see M3 in
[ROADMAP.md](ROADMAP.md)), so a recording from one release may refuse to load in
the next. That is a deliberate promise not made — and the reason no sample
recordings ship in the package, where they would break on the next upload.

## The split, when the Python side is packaged

**Built.** `debian/control` produces the two binaries below from one source,
and `python/tests/e2e/test_packages.py` asserts from a staged install tree that
they really are two products: the game-only tree has no `.py` file in it and its
menu has no **TRAIN AI** entry, and the full tree resolves the launcher and
grows one. The division is by *dependency weight*, not by tidiness:

| Package | Arch | Contents | Depends |
|---|---|---|---|
| `missile-defense` | any | the game, as today | Qt 6, Vulkan loader |
| `missile-defense-trainer` | any | the whole `missile_defense` package, `_md_native*.so`, and the two entry points | `${python3:Depends}`, `python3-numpy`, PySide6, `python3-venv`; **Suggests** torch, pynvml |

Two packages, not three. There was a `python3-md` holding the Python half so it
could be installed without the trainer; it is gone. The audience for a headless
environment with no Qt is served by the wheel — the same code, and the
distribution channel that audience already uses — and a third binary package was
a third name to explain for a split nobody was choosing.

What survives the merge is the boundary that matters: the game's dependency chain
still contains no Python and no LGPLv3 Qt binding, and CI asserts it from the
built `.deb` rather than from `debian/control`.

Two costs, stated rather than discovered later. The trainer is `Architecture:
any` now, because it carries a compiled extension. And Debian Python Policy would
rather a module under `dist-packages` came from a package named `python3-*`;
this one is not, which is the price of having two names instead of three.

The trainer and the training loop stay in one package for now. Their heavy dependencies
are disjoint (PySide6 versus torch), which argues for splitting them, but both are
`Architecture: all` and neither imports the other, so the cost of being wrong is a
few unused `.py` files. Split when the trainer grows a real one.

File locations for that split: `/usr/lib/python3/dist-packages/missile_defense/**` for the
package and its extension, `/usr/bin/missile-defense-train` and `/usr/bin/missile-defense-trainer` for the
tools — `/usr/games` is for the game — and their man pages in section 1 rather
than the game's section 6.

The extension is built with nanobind's `STABLE_ABI`, so it is an `abi3` object
that survives a Python minor-version bump. That is not only tidiness; it is what
makes the torch story below work.

## The same split on Windows and macOS

Debian expresses "two products" as two packages. The other two platforms have no
package manager to express it with, so each says it in its own idiom — and every
difference below comes from one fact: **only Debian owns the interpreter.**

| | Who installs the trainer | What ships beside the game | How it is found afterwards |
|---|---|---|---|
| Debian | `apt`, as `missile-defense-trainer` | — | `/usr/bin/missile-defense-trainer` on `PATH` |
| Windows | the game, on request, with pip | `missile_defense-*.whl`, `HOW-TO-TRAIN.html` | the interpreter it installed into, recorded in `trainer.conf` |
| macOS | the game, on request, with pip | the same two, inside `Contents/` | the same record |

**This used to be three different mechanisms and is now two.** Windows shipped
the Python package beside the exe with a `.cmd` launcher; macOS shipped a second
`.app` with a shell wrapper. Both existed to answer the same question — where did
the payload go — and both were guesses, because the interpreter that has to
import it belongs to the user. The `.cmd` had a second problem no amount of care
would fix: Smart App Control blocks scripts outright on a stock Windows 11.

So neither ships a payload. The game carries the wheel, TRAIN AI runs
`pip install --user "<wheel>[trainer]"` in a terminal window the user can watch,
and it writes down the interpreter it used. Nothing has to be found by searching
afterwards — see `app/trainer.hpp` for why searching could not be made to work,
and `app/install.hpp` for the four answers TRAIN AI can give.

There is **one** CPack component now, `game`, declared in the top-level
`CMakeLists.txt` and tagged on every `install()` rule. The tagging is load-bearing
beyond the installer: `cmake --install --component game` is how the packaging
tests stage the exact game-only product out of a build tree that also built the
bindings. Before the tags existed, every such rule went into `Unspecified` and a
"game-only" staging quietly carried `_md_native` with it.

The wheel itself is not built here. `MD_TRAINER_WHEEL` names the artifact the
`wheels` CI job already produced with cibuildwheel and import-tested, so what
ships beside the game is that same file rather than a second one nobody has run.
It is attached to the GitHub release; there is no PyPI publish step yet, so
`pip install missile-defense` does not resolve and every install path goes
through the file. Unset — which is every developer build — the game simply has
nothing to install and says so.

macOS gets `CPACK_MONOLITHIC_INSTALL` even so. A disk image is a window with
icons in it, not an installer with checkboxes, so splitting it into two images
would be the wrong shape for the platform; the components exist there to build
the *layout*, and the user's choice is which icon they drag.

**The trainer needs the native binding.** The managed runtime (`missile_defense.runs.runtime`)
installs torch and nothing else — `md` and `_md_native` come from the payload —
so a Windows or macOS build that packages the trainer without
`MD_BUILD_BINDINGS=ON` produces a trainer that starts, browses and replays, and
cannot train. `bindings/CMakeLists.txt` skips itself silently when Python or
nanobind is missing, so the top-level file raises a CMake warning when the
payload is being installed without it, and both CI jobs assert the extension is
inside the staged tree.

## PyTorch

**`Suggests`, never `Depends`.** Three reasons, heaviest first:

1. Debian's `python3-torch` is CPU-only and lags upstream. Anyone doing real runs
   installs the vendor wheels (CUDA, or ROCm), so a hard dependency would drag in
   a multi-gigabyte second copy that never gets used.
2. Debian 12 and later mark the system interpreter *externally managed* (PEP 668),
   so `pip install torch` there fails by design. The documented path has to be a
   virtualenv — and the one that works alongside a packaged trainer is:

   ```bash
   python3 -m venv --system-site-packages ~/.venvs/md
   ~/.venvs/md/bin/pip install torch
   ~/.venvs/md/bin/missile-defense-train
   ```

   `--system-site-packages` is the whole trick: the venv sees the distro-installed
   `_md_native.abi3.so` while pip owns torch. The stable ABI is what keeps that
   working when the distro's Python moves.
3. It keeps the trainer package cheap for someone who only wants to watch and
   replay runs, which needs no torch at all.

The price is that a missing torch must be *explained* rather than raised, and
both commands now do. The trainer checks with `importlib.util.find_spec` and
disables Start with a reason (`missile_defense.runs.runner.can_train`); `missile-defense-train` goes through
`missile_defense.training.cli`, which checks before importing anything heavy and names the `pip
install` that would fix it. CI installs the wheel into a venv with neither
package and asserts both messages, because the failure being avoided is one that
only appears where the optional half is absent — which is never a developer's
machine.

## What is pip-installable today

`pip install .` works, and produces an `md` that can `import _md_native`:

| Piece | Where |
|---|---|
| build backend | `scikit-build-core` — compiles the extension through this same CMake tree, with `MD_BUILD_APP=OFF`, so a NumPy array does not cost a Vulkan SDK |
| the package | `wheel.packages = ["python/missile_defense"]` |
| the extension | `install(TARGETS _md_native …)` in `bindings/CMakeLists.txt`, into `${MD_PYTHON_INSTALL_DIR}` — `md` by default, an absolute `dist-packages` path for a distribution build |
| commands | `missile-defense-train` → `missile_defense.training.cli:train` (`[project.scripts]`), `missile-defense-trainer` → `missile_defense.ui.__main__:main` (`[project.gui-scripts]`) |
| extras | `[train]` = torch, `[trainer]` = PySide6 + psutil + nvidia-ml-py + amdsmi (Linux); neither is ever required |

`STABLE_ABI` is what makes the installed object worth keeping: it is `abi3`, so
it survives the distribution's Python moving a minor version instead of having
to be rebuilt in lockstep with it.

That was the blocker under everything below — there was no build backend at all,
so nothing here was installable and `pybuild` had nothing to invoke.

### `STABLE_ABI` is a request, and it was being refused

nanobind honours it only when three things hold: CPython, ≥ 3.12, and a
`Python::SABIModule` target. That last one exists only if `find_package(Python)`
asked for `Development.SABIModule` — and for a long time this one did not, so
nanobind quietly built a version-tagged module instead and said nothing. Every
build worked on the machine that built it, and the paragraph above was false.

It is checked now rather than assumed, in three places, because the failure is
silent by nature:

* `bindings/CMakeLists.txt` reads the target's own `SUFFIX` after the fact and
  compares it to nanobind's `NB_SUFFIX_S`. Not a match on the string `abi3`:
  the limited-API suffix is `.abi3.so` on Linux and macOS but a plain `.pyd` on
  Windows, so the literal would fail every correct Windows build.
* The top-level file turns that into a **fatal error** when
  `MD_INSTALL_PYTHON_PACKAGE` is on. A developer build may be tied to one
  interpreter; a build that is being packaged may not.
* Each packaging job asserts the shipped filename — the trainer `.deb`'s
  contents, the staged NSIS component, the macOS trainer bundle — and `test_packaging.py`
  asserts both the declaration and the built artifact.

### Windows ships an extension from a second build

`MD_PREBUILT_PYTHON_MODULE` names an `_md_native.<suffix>` built elsewhere, to
install in place of this build's own. It exists for one situation, and Windows
is the only place that has it.

The game is built in MSYS2/CLANG64 — that is where Qt and Vulkan are — so the
extension built beside it is a mingw object against MSYS2's interpreter and
libc++. The installed trainer does not run there: `missile-defense-trainer.cmd` execs
whatever `python` is on PATH, which is a python.org CPython for anyone who
followed docs/WINDOWS.md, because that is where the PySide6 and torch wheels
are. Loading one into the other fails inside an import with a DLL error.

So the Windows job builds `_md_native` twice: once with the game, and once with
MSVC against a python.org CPython (the `win-native` preset, which exists for
this reason), and names the second one here. The stable ABI makes the
substitution total — both are called `_md_native.pyd`, so nothing downstream
has to know which build it got, and the installed trainer works on 3.12 and
later rather than on one exact minor version.

## Checklist for the day this is published

* `debian/control`: the two binary packages above, `dh-python`/`pybuild` for the
  trainer, `${python3:Depends}` and `${shlibs:Depends}` on the extension.
* `debian/rules`: build with `-DMD_BUILD_BINDINGS=ON` and
  `-DMD_PYTHON_INSTALL_DIR=/usr/lib/python3/dist-packages/missile_defense`, then
  `dh_auto_install` the `python` component. ✅ the install rule exists; what is
  left is the `debian/` side of it.
* A `README.Debian` for `missile-defense-trainer` carrying the venv recipe above.
  ✅ `debian/missile-defense-trainer.README.Debian` — spelled per-package, because
  a bare `debian/README.Debian` is installed into the *first* binary package,
  which is the game and has no Python in it. `missile-defense-train` prints its installed path.
* `debian/copyright` extended for the Python sources and for miniaudio, which is
  fetched at build time when `libminiaudio-dev` is absent.
* Desktop entry for the trainer under `Categories=Development;Science;` — it is a
  tool, not a game, and does not belong in the games menu.
