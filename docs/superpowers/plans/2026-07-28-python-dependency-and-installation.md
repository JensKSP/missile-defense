# Python as a Dependency: Naming, Layering and Installation

> **Status (2026-07-28):** Built, and every checkbox closed. What is *not*
> proven is listed under "Unverifiable here": three claims need a real Windows
> or macOS machine, and no amount of local testing will change that.

**Goal:** Make the two products — the game and the trainer — installable and
findable on every supported platform without shipping a Python interpreter, and
without leaving a menu entry that does nothing.

**Design:** worked out in conversation on 2026-07-28; the reasoning that is not
obvious from the outcome is recorded under *Why* below.

---

## The shape

The user installs Python themselves. We ship the trainer as a wheel **beside the
game**, and `TRAIN AI` installs it from that local copy. Debian is unaffected:
there the trainer is an apt package and the distribution owns the interpreter.

| | Windows | macOS | Debian |
|---|---|---|---|
| Artifact | NSIS installer / ZIP, **game only** | DMG with **one** `.app` | two binary packages |
| Wheel lives | install directory | `Contents/Resources` | — (apt) |
| Get the trainer | `TRAIN AI` → confirm → terminal | `TRAIN AI` → confirm → terminal | `apt install missile-defense-trainer` |
| Start the trainer | `TRAIN AI` | `TRAIN AI` | `TRAIN AI` or menu entry |
| Python floor | 3.12 (the wheel is `cp312-abi3`) | 3.12 | from apt |

## Decisions

1. **Message location:** a short notice in the game's own font, `ENTER` opens a
   bundled `HOW-TO-TRAIN.html` in the system's default application
   (`QDesktopServices::openUrl`, already in Qt6::Gui).
2. **Confirm before installing:** yes — it downloads ~150 MB of PySide6.
3. **pip target:** `pip install --user`. PEP 668 is effectively a Linux-distro
   concern and Linux goes through apt.
4. **Windows installer:** the trainer component is dropped entirely.
5. **Wheel source:** the artifact from the existing `wheels` CI job — the same
   wheel that goes to PyPI, already import-tested.
6. **Interpreter choice:** newest ≥ 3.12; the helper prints which it picked
   before installing.
7. **Version drift:** `TRAIN AI` compares the installed trainer against the
   bundled wheel and offers to reinstall.
8. **Instructions page:** one page, three platform sections, shipped everywhere.

### Two decisions that are consequences, not choices

* **No `.cmd` / `.command` helper file.** `app/trainer.cpp` already records that
  Smart App Control blocks scripts outright on a stock Windows 11 — which is why
  the current Windows lookup deliberately runs the interpreter instead of the
  launcher. A helper script would walk into the same wall. The game spawns
  `cmd.exe /k <python> -m pip install --user "<wheel>[trainer]"`; `cmd.exe` is a
  system binary and everything else is an argument, so there is no script file to
  block. The terminal window is what gives progress, scrollback and a copyable
  error for free.
* **Python 3.12, not 3.11.** `requires-python` stays `>=3.11` for anyone
  building from source, but the shipped wheel is `cp312-abi3` and pip will refuse
  it on 3.11. The message has to say 3.12.

## Lookup order (replaces the current one)

```
1. MD_TRAINER                        explicit override
2. the recorded interpreter          written when we install; runs -m missile_defense.ui
3. missile-defense-trainer on PATH   Debian
4. a checkout                        developers
```

Step 2 records **the interpreter**, not the script. That answers three problems
at once: pip's scripts directory needs no guessing, a Finder-launched app on
macOS has almost no `PATH` (launchd gives it `/usr/bin:/bin:/usr/sbin:/sbin`,
and `/opt/homebrew/bin` and `~/Library/Python/3.x/bin` are in neither that nor
the two directories `trainer.cpp` searches), and `.exe` vs `.cmd` stops
mattering. Running `-m missile_defense.ui` is what the checkout case already
does.

## Steps

### Step 4 of the rename series (prerequisite)

- [x] Debian: two binary packages. `python3-md` disappears; its contents move
      into `missile-defense-trainer`, which becomes `Architecture: any`.
      Accepted cost: no headless `apt install` of the library alone, and
      `dist-packages` shipped from a package not named `python3-*` (lintian
      friction, deliberate).

### 1 — Rework the lookup

- [x] Delete `Origin::Payload`, `Lookup::payload_root`, `payload_module` and the
      probe that fills it (`app/trainer.cpp`, `app/trainer.hpp`).
- [x] Add the recorded-interpreter source, read from `AppLocalDataLocation`.
- [x] Mirror the order in `missile_defense.runs.runner.trainer_executable` — the
      two must not disagree, which is what `app/trainer.hpp` calls the boundary
      between the two products.
- [x] Tests in `app/tests/unit/test_trainer.cpp` and `test_ui_runner.py`.

### 2 — The install flow

- [x] `app/install.{cpp,hpp}`: find interpreters, pick the newest ≥ 3.12, spawn
      the terminal, record the interpreter on success.
- [x] Notice screens for the three outcomes (no trainer / no Python / update
      available), in the game's own font. **The font has no lowercase** — so a
      command or URL must never be rendered there.
- [x] `packaging/HOW-TO-TRAIN.html`, installed on all platforms.
- [x] Version comparison against the bundled wheel. `Offer::Update` — and the
      record is written by the *install*, chained after pip with `&&`, because
      the game spawns it detached and never learns whether it succeeded. An
      optimistic record would name an interpreter that exists and cannot import
      the trainer.

### 3 — Packaging

- [x] Ship the wheel: install dir (Windows), `Contents/Resources` (macOS).
- [x] Delete `packaging/launcher.cmd.in`, `packaging/trainer-bundle-launcher.in`,
      `packaging/trainer.Info.plist.in`, the second `.app`, the `md\` payload
      install rules and the `python` CPack component.
- [x] CI: the packaging jobs `needs: wheels`, download the artifact for their
      platform and pass `-DMD_TRAINER_WHEEL`. It serialises those two jobs behind
      the wheel build, which is the price of shipping the file that was actually
      tested rather than a second one built here.
- [x] Removed with it: 56 lines of Windows CI that installed a second CPython and
      built `_md_native` again with MSVC, to be substituted through
      `MD_PREBUILT_PYTHON_MODULE`. The wheel *is* that module, built by
      cibuildwheel and import-tested there.

### 4 — Documentation and Debian polish

- [x] `Suggests: missile-defense-trainer` in the game's Debian stanza — today
      nothing points a game-only user at the trainer.
- [x] `docs/PACKAGING.md` and `docs/WINDOWS.md` rewritten for the new
      mechanism. `docs/MACOS.md` needed nothing — it never described the trainer.
      `README.md` gained the developer-dependency section instead.

## Unverifiable here

Three claims in this plan cannot be tested on the development machine and need a
real run before they count as done:

1. The Smart App Control workaround via `cmd.exe`.
2. Launching Terminal from an unsigned `.app` on macOS.
3. Parsing `py -0p` for interpreter selection.

## Deferred, deliberately

* `_md_native` still carries the `md` abbreviation. nanobind couples the module
  name to the filename and to the stable-ABI suffix, three CI jobs assert
  `_md_native.abi3.so` literally, and the Windows path through
  `MD_PREBUILT_PYTHON_MODULE` cannot be exercised where no extension builds.
  Its own commit, where CI can prove it.
* The PyPI `description` still promises *"game + RL environment"* for a wheel
  that contains no game.
* The trainer has no icon of its own; both `.desktop` files point at
  `Icon=missile-defense`.
* A Windows Start-menu entry for the trainer — moot once the component is
  dropped, but the underlying `CPACK_PACKAGE_EXECUTABLES` gap is worth a look.
