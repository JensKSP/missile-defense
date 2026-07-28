# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What the console looks like to the desktop it opens on.

An icon is cosmetic right up to the point where two windows of the same product
are indistinguishable in a taskbar, or a menu entry draws a blank square. All
three facts here are load-bearing for one of those, and each has a way of going
missing quietly: a file left out of the wheel, an id inherited from whatever
started the process, a desktop entry naming an icon nobody installed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import missile_defense
from missile_defense.ui import branding

ROOT = Path(missile_defense.__file__).parents[2]


def test_the_icon_travels_with_the_package() -> None:
    """Not in `packaging/`, which exists only in a checkout.

    On Windows and macOS the console is a wheel installed into somebody's own
    interpreter: there is no install prefix beside it and no data directory it
    owns, so an icon anywhere but inside the package is an icon it cannot find.
    """
    assert branding.ICON.is_file(), f"no icon at {branding.ICON} — run `python -m tools.make_icon`"
    assert branding.ICON.parent == Path(branding.__file__).parent


def test_the_wheel_does_not_exclude_what_the_icon_is() -> None:
    """`wheel.exclude` is applied after the CMake install and deletes by pattern.

    A `*.png` swept up there would produce a package whose console starts with a
    blank icon and no error — the same class of bug as the `*.pyd` pattern that
    once deleted the extension this project had just built.
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    excluded = config["tool"]["scikit-build"]["wheel"]["exclude"]
    assert not [pattern for pattern in excluded if pattern.endswith(".png")]


def test_the_desktop_entry_names_the_icon_that_is_installed_with_it() -> None:
    """The Linux half: `Icon=` is a theme lookup, not a path.

    It resolves to `hicolor/<size>/apps/<name>.png`, and those files are
    installed by the `python` component beside this entry — so the name in the
    entry, the name of the committed PNGs and what CMake installs have to be one
    name. Naming the *game's* icon would work and be wrong: two products, one
    emblem, and no way to tell which window is which.
    """
    entry = (ROOT / "packaging" / "missile-defense-trainer.desktop").read_text(encoding="utf-8")
    icon = next(line for line in entry.splitlines() if line.startswith("Icon="))
    name = icon.removeprefix("Icon=").strip()
    assert name == branding.DESKTOP_ENTRY
    for size in (16, 32, 48, 64, 128, 256):
        png = ROOT / "packaging" / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{name}.png"
        assert png.is_file(), f"{png} is missing — run `python -m tools.make_icon`"


def test_the_taskbar_identity_is_claimed_only_where_it_exists() -> None:
    """AppUserModelID is a Windows concept; every other desktop uses the entry.

    Injected platform rather than the real one, because both branches have to
    hold on the machine running the test — and the branch that matters cannot be
    reached at all on the machine most of this is written on.
    """
    assert branding.claim_taskbar_identity(platform="linux") is False
    assert branding.claim_taskbar_identity(platform="darwin") is False


def test_the_console_does_not_borrow_the_game_s_identity() -> None:
    """Sharing the id would file the console's window under the game's button.

    Which is worse than the python icon it replaces: a person who just started
    the trainer from the game's menu would find no new taskbar button at all.
    """
    assert branding.APP_ID.startswith("de.koehler-speyer.missile-defense")
    assert branding.APP_ID != "de.koehler-speyer.missile-defense"
