# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""``python -m missile_defense.training`` — one training run, or one experiment.

Deliberately the same door as the installed ``missile-defense-train`` command:
both land in :func:`missile_defense.training.cli.train`, so a run started from a
terminal, from the trainer window and from the multi-seed runner's children all
go through the same torch check and the same argument handling. A second
entry point that skipped the check would report a missing optional dependency as
a traceback, which is the thing that check exists to prevent.
"""

from __future__ import annotations

from .cli import train

if __name__ == "__main__":
    raise SystemExit(train())
