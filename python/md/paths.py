# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Where a run's files live. No Qt, no torch — the trainer and the console agree
on this, and so does the game.

In a checkout everything lands in ``./runs`` and always has. Installed from a
package that is wrong twice over: ``/usr/games`` is not writable, and the working
directory of a program started from a desktop entry is ``$HOME`` or ``/``, so
"beside the binary" and "beside the shell" both mean nowhere useful. The game's
REPLAYS browser looked in the working directory for exactly this reason and found
nothing for every packaged user.

So, in order:

1. an explicit ``--out-dir`` (or the console's picker) — always wins;
2. ``$MD_RUNS_DIR`` — one env var for a scratch disk or a shared box;
3. ``./runs`` **if it already exists** — a checkout keeps behaving as it did,
   with no flag and no migration;
4. the per-user data directory — ``~/.local/share/MissileDefense/runs`` and its
   equivalents, which is what an installed copy uses.

Rule 3 is the interesting one: presence rather than a build-time switch, so the
same binary does the obvious thing in a checkout and in ``/usr/games`` without
knowing which it is.

**Data, not state or cache.** Checkpoints are the expensive output of hours of
compute, so they belong where a user expects their own files to be kept, not in
``~/.local/state`` (things you would shrug at losing) or ``~/.cache`` (things
that regenerate). See ``docs/PACKAGING.md``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from ._protocol import MODELS_DIR, RUNS_DIR

#: Mirrors ``QGuiApplication::setApplicationName`` in ``app/main.cpp``. The game
#: already keeps its high scores under this name; runs join them rather than
#: inventing a second directory and a migration for the first.
APP_NAME = "MissileDefense"

#: The directory a run writes into, under whichever root wins below.
#: From `protocol.toml`; the game looks in the same place.
RUNS_NAME = RUNS_DIR

#: One override for "put the runs somewhere else entirely".
RUNS_ENV = "MD_RUNS_DIR"

#: Where promoted models live: one directory per model, beside the runs rather
#: than inside one. A promoted model has to *outlive* the run that produced it —
#: that is the whole point of promoting rather than pointing at a checkpoint —
#: and a run directory is the thing cleanup and archiving delete.
MODELS_NAME = MODELS_DIR

#: One override, mirroring the runs one. A shared box wants a shared league.
MODELS_ENV = "MD_MODELS_DIR"

#: Named sets of training options, saved from the console's Start dialog.
PRESETS_NAME = "presets.json"

#: One override, so a test — or a second machine sharing a home directory — can
#: point the console at another file.
PRESETS_ENV = "MD_PRESETS_FILE"

#: Where the console installs a training runtime it manages itself.
RUNTIME_NAME = "runtime"

#: One override, for a scratch disk — a torch install is several gigabytes.
RUNTIME_ENV = "MD_RUNTIME_DIR"


def data_home(environ: Mapping[str, str] | None = None, *, platform: str = sys.platform) -> Path:
    """The per-user data directory for this application.

    The same place Qt's ``AppLocalDataLocation`` resolves to, because the game
    uses that and the two must not disagree about where a recording is. *Local*
    rather than roaming on Windows: a run directory is checkpoints, and syncing
    those onto a domain profile would be a surprise measured in gigabytes.
    """
    env = os.environ if environ is None else environ
    if platform == "win32":
        root = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif platform == "darwin":
        root = str(Path.home() / "Library" / "Application Support")
    else:
        root = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(root) / APP_NAME


def runs_dir(
    explicit: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    platform: str = sys.platform,
) -> Path:
    """Where this process should read and write runs. Creates nothing."""
    if explicit is not None:
        return explicit
    env = os.environ if environ is None else environ
    override = env.get(RUNS_ENV)
    if override:
        return Path(override)
    local = (Path.cwd() if cwd is None else cwd) / RUNS_NAME
    if local.is_dir():
        return local
    return data_home(env, platform=platform) / RUNS_NAME


def models_dir(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    platform: str = sys.platform,
) -> Path:
    """Where promoted models live. Creates nothing.

    Beside :func:`runs_dir` and by the same rules, so a checkout keeps its
    league next to its runs and an installed copy keeps both under the per-user
    data directory. Deliberately *not* inside a run: promotion exists so a model
    survives the run being cleaned up, archived or deleted, and storing it in
    the thing it must outlive would defeat that on the first tidy-up.
    """
    env = os.environ if environ is None else environ
    override = env.get(MODELS_ENV)
    if override:
        return Path(override)
    runs = runs_dir(environ=env, cwd=cwd, platform=platform)
    # Siblings: `runs/` and `models/`. A checkout that has one gets the other
    # in the obvious place, with no second rule to remember.
    return runs.parent / MODELS_NAME


def presets_file(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    platform: str = sys.platform,
) -> Path:
    """The user's saved training-option sets. Creates nothing.

    Beside the runs and the models, by the same rules: a checkout keeps its
    presets next to the runs it starts with them, and an installed copy keeps
    all three under the per-user data directory. A file rather than a directory
    because the whole point is that it is small, hand-editable and easy to copy
    to another machine — a preset is a dozen numbers, not an artifact.
    """
    env = os.environ if environ is None else environ
    override = env.get(PRESETS_ENV)
    if override:
        return Path(override)
    runs = runs_dir(environ=env, cwd=cwd, platform=platform)
    return runs.parent / PRESETS_NAME


def runtime_dir(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str = sys.platform,
) -> Path:
    """Where a console-managed training runtime is installed. Creates nothing.

    Deliberately *not* the checkout's rule-3 treatment that :func:`runs_dir` has.
    A run directory belongs beside the source you started it from; a multi-gigabyte
    torch install does not, and one per clone would be a surprise measured in tens
    of gigabytes. So it always lands in the per-user data directory, where a
    second checkout finds the same one already installed.

    Data rather than cache for the same reason runs are: pip would have to
    re-download it, and on a metered connection that is not a shrug.
    """
    env = os.environ if environ is None else environ
    override = env.get(RUNTIME_ENV)
    if override:
        return Path(override)
    return data_home(env, platform=platform) / RUNTIME_NAME
