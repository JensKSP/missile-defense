# Packaging

What ships, where it lands, and where a run's files go. The Debian package is the
reference — it is the one with rules about all three — but the file locations
below are what the game and the trainer do on every platform.

## What ships today

One binary package, the game only:

| Path | What |
|---|---|
| `/usr/games/missile-defense` | the game (renamed from `md_app`) |
| `/usr/share/applications/missile-defense.desktop` | menu entry |
| `/usr/share/icons/hicolor/…` | icons the desktop entry resolves |
| `/usr/share/man/man6/missile-defense.6.gz` | man page (section 6: games) |
| `/usr/share/doc/missile-defense/…` | licence and third-party notices |

There is no runtime data directory at all: the SPIR-V shaders are compiled into
the executable at build time (`app/CMakeLists.txt`), so nothing is loaded from
disk and nothing can go missing.

Two build paths exist and that is deliberate: `debian/` (debhelper) is the
authority for Linux, and CPack produces the Windows NSIS installer and ZIP, plus
the macOS disk image. `poe deb` runs CPack's DEB generator, which is a
convenience for testing a local package — it cannot express the multi-binary
split below, so when that lands, `debian/` is where it lands.

On macOS, `poe dmg` builds a drag-to-Applications image containing `md_app.app`
with the Qt frameworks *and* MoltenVK inside it, so it depends on nothing but
macOS. It is ad-hoc signed by default, which runs but is not distributable;
[MACOS.md](MACOS.md#signing-it-for-other-people) covers the Developer ID and
notarisation path.

### Every artifact is built on every push

[.github/workflows/ci.yml](../.github/workflows/ci.yml) builds all four on the
platform each is meant for, and uploads them:

| Job | Platform | Produces |
|---|---|---|
| `gate` | Ubuntu | the quality gate, plus the CPack `.deb` |
| `debian` | Ubuntu | the **debhelper** `.deb` from `debian/`, lintian-checked |
| `windows` | Windows / MSYS2 CLANG64 | NSIS installer + portable ZIP |
| `macos` | macOS on Apple silicon | the `.dmg` |

`gate` and `debian` both produce a Debian package on purpose. They are different
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
(`python/md/paths.py`, mirrored in `app/game_window.cpp`):

1. an explicit `--out-dir`, or the console's run picker;
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

Not built yet; this is the shape it should take. The division is by *dependency
weight*, not by tidiness:

| Package | Arch | Contents | Depends |
|---|---|---|---|
| `missile-defense` | any | the game, as today | Qt 6, Vulkan loader |
| `python3-md` | any | `md.env`, `md.eval`, `md.control`, `md.paths`, `_md_native*.so` | `${python3:Depends}`, `python3-numpy` |
| `missile-defense-training` | all | `md.train`, `md.ppo`, `md.ui`, the `md-train`/`md-console` entry points | `python3-md`; **Suggests** torch, PySide6, psutil, pynvml |

`python3-md` is the piece with reuse value on its own: a deterministic, vectorised
RL environment that imports without a game installed. Splitting it also makes two
boundaries into packaging facts rather than rules people remember — the game's
dependencies never include Python, and the console's LGPLv3 Qt Charts never
appear in the game's chain.

The console and the trainer stay in one package for now. Their heavy dependencies
are disjoint (PySide6 versus torch), which argues for splitting them, but both are
`Architecture: all` and neither imports the other, so the cost of being wrong is a
few unused `.py` files. Split when the console grows a real one.

File locations for that split: `/usr/lib/python3/dist-packages/md/**` for the
package and its extension, `/usr/bin/md-train` and `/usr/bin/md-console` for the
tools — `/usr/games` is for the game — and their man pages in section 1 rather
than the game's section 6.

The extension is built with nanobind's `STABLE_ABI`, so it is an `abi3` object
that survives a Python minor-version bump. That is not only tidiness; it is what
makes the torch story below work.

## PyTorch

**`Suggests`, never `Depends`.** Three reasons, heaviest first:

1. Debian's `python3-torch` is CPU-only and lags upstream. Anyone doing real runs
   installs the vendor wheels (CUDA, or ROCm), so a hard dependency would drag in
   a multi-gigabyte second copy that never gets used.
2. Debian 12 and later mark the system interpreter *externally managed* (PEP 668),
   so `pip install torch` there fails by design. The documented path has to be a
   virtualenv — and the one that works alongside a packaged `python3-md` is:

   ```bash
   python3 -m venv --system-site-packages ~/.venvs/md
   ~/.venvs/md/bin/pip install torch
   ~/.venvs/md/bin/md-train
   ```

   `--system-site-packages` is the whole trick: the venv sees the distro-installed
   `_md_native.abi3.so` while pip owns torch. The stable ABI is what keeps that
   working when the distro's Python moves.
3. It keeps `apt install python3-md` cheap for someone who only wants the
   environment to run their own agent against.

The price is that a missing torch must be *explained* rather than raised, and
both commands now do. The console checks with `importlib.util.find_spec` and
disables Start with a reason (`md.ui.runner.can_train`); `md-train` goes through
`md.cli`, which checks before importing anything heavy and names the `pip
install` that would fix it. CI installs the wheel into a venv with neither
package and asserts both messages, because the failure being avoided is one that
only appears where the optional half is absent — which is never a developer's
machine.

## What is pip-installable today

`pip install .` works, and produces an `md` that can `import _md_native`:

| Piece | Where |
|---|---|
| build backend | `scikit-build-core` — compiles the extension through this same CMake tree, with `MD_BUILD_APP=OFF`, so a NumPy array does not cost a Vulkan SDK |
| the package | `wheel.packages = ["python/md"]` |
| the extension | `install(TARGETS _md_native …)` in `bindings/CMakeLists.txt`, into `${MD_PYTHON_INSTALL_DIR}` — `md` by default, an absolute `dist-packages` path for a distribution build |
| commands | `md-train` → `md.cli:train`, `md-console` → `md.ui.__main__:main` |
| extras | `[train]` = torch, `[console]` = PySide6 + psutil + nvidia-ml-py + amdsmi (Linux); neither is ever required |

`STABLE_ABI` is what makes the installed object worth keeping: it is `abi3`, so
it survives the distribution's Python moving a minor version instead of having
to be rebuilt in lockstep with it.

That was the blocker under everything below — there was no build backend at all,
so nothing here was installable and `pybuild` had nothing to invoke.

## Checklist for the day this is published

* `debian/control`: the three binary packages above, `dh-python`/`pybuild` for the
  Python one, `${python3:Depends}` and `${shlibs:Depends}` on the extension.
* `debian/rules`: build with `-DMD_BUILD_BINDINGS=ON` and
  `-DMD_PYTHON_INSTALL_DIR=/usr/lib/python3/dist-packages/md`, then
  `dh_auto_install` the `python` component. ✅ the install rule exists; what is
  left is the `debian/` side of it.
* A `README.Debian` for `missile-defense-training` carrying the venv recipe above.
* `debian/copyright` extended for the Python sources and for miniaudio, which is
  fetched at build time when `libminiaudio-dev` is absent.
* Desktop entry for the console under `Categories=Development;Science;` — it is a
  tool, not a game, and does not belong in the games menu.
