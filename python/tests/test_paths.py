# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for where a run's files live.

The trainer, the trainer and the game all resolve this, so it has exactly one
implementation on the Python side and a mirror of it in the app. What is pinned
here is the order — a rule nobody can remember the order of is a rule that will
be got wrong.
"""

from __future__ import annotations

from pathlib import Path

from missile_defense.runs.paths import APP_NAME, RUNS_ENV, data_home, runs_dir


def test_an_explicit_directory_always_wins(tmp_path: Path) -> None:
    chosen = runs_dir(
        Path("somewhere/else"),
        environ={RUNS_ENV: str(tmp_path)},
        cwd=tmp_path,
    )
    assert chosen == Path("somewhere/else")


def test_the_environment_beats_the_working_directory(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    assert runs_dir(environ={RUNS_ENV: "/scratch/md"}, cwd=tmp_path) == Path("/scratch/md")


def test_a_checkout_keeps_using_its_own_runs_directory(tmp_path: Path) -> None:
    # Presence, not a build-time switch: the same binary does the obvious thing
    # in a checkout and in /usr/games without being told which it is.
    (tmp_path / "runs").mkdir()
    assert runs_dir(environ={}, cwd=tmp_path) == tmp_path / "runs"


def test_without_one_it_falls_back_to_the_users_data_directory(tmp_path: Path) -> None:
    env = {"XDG_DATA_HOME": str(tmp_path / "share")}
    chosen = runs_dir(environ=env, cwd=tmp_path, platform="linux")
    assert chosen == tmp_path / "share" / APP_NAME / "runs"
    assert not chosen.exists()  # resolving a path must not create one


def test_the_data_directory_follows_each_platforms_convention(tmp_path: Path) -> None:
    assert data_home({"XDG_DATA_HOME": "/x"}, platform="linux") == Path("/x") / APP_NAME
    # Local rather than roaming: a run directory is checkpoints, and syncing
    # those onto a domain profile would be a surprise measured in gigabytes.
    assert (
        data_home({"LOCALAPPDATA": "C:/u/AppData/Local"}, platform="win32")
        == Path("C:/u/AppData/Local") / APP_NAME
    )
    assert data_home({}, platform="darwin").parts[-3:] == (
        "Library",
        "Application Support",
        APP_NAME,
    )


def test_xdg_falls_back_to_the_specified_default(tmp_path: Path) -> None:
    # An unset XDG_DATA_HOME means ~/.local/share, per the spec — not an error
    # and not the working directory.
    home = data_home({}, platform="linux")
    assert home.parts[-3:] == (".local", "share", APP_NAME)


def test_the_app_name_matches_the_one_the_game_registers() -> None:
    # app/main.cpp sets this as the Qt application name, which is what puts the
    # high scores in the same directory. If one moves, both move.
    source = (Path(__file__).parents[2] / "app" / "main.cpp").read_text(encoding="utf-8")
    assert f'setApplicationName("{APP_NAME}")' in source


def test_the_game_does_not_derive_its_data_directory_from_the_organisation_name() -> None:
    """The bug this test exists for, and the reason the one above did not catch it.

    `main.cpp` sets an organisation name *and* an application name, both
    "MissileDefense", and Qt builds `AppDataLocation` and `AppLocalDataLocation`
    out of both — so the game resolved `<data>/MissileDefense/MissileDefense`
    while `data_home` appends the name once. Two promoted models sat in the
    trainer's `models/` and WATCH AI listed none of them.

    Asserting the application name is registered was never enough: it was, and
    the paths still disagreed. What has to hold is that the *data* directory
    comes from `GenericDataLocation`, which Qt builds from neither name.
    """
    app = Path(__file__).parents[2] / "app"
    header = (app / "data_dir.hpp").read_text(encoding="utf-8")
    assert "GenericDataLocation" in header
    assert f'app_data_name = "{APP_NAME}"' in header

    # Every other source has to go through it. Qualified, because that is how the
    # constants are spelled where they do damage — prose may still name them, and
    # the migration in the header has to.
    for source in sorted(app.glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        for location in ("AppDataLocation", "AppLocalDataLocation"):
            assert f"QStandardPaths::{location}" not in text, (
                f"{source.name} resolves a path with QStandardPaths::{location}, which Qt "
                f"builds from the organisation name — use md::app_data_dir() instead"
            )
