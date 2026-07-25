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

#: Mirrors ``QGuiApplication::setApplicationName`` in ``app/main.cpp``. The game
#: already keeps its high scores under this name; runs join them rather than
#: inventing a second directory and a migration for the first.
APP_NAME = "MissileDefense"

#: The directory a run writes into, under whichever root wins below.
RUNS_NAME = "runs"

#: One override for "put the runs somewhere else entirely".
RUNS_ENV = "MD_RUNS_DIR"


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
