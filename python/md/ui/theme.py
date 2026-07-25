# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The console's palette, lifted from the game. Dark by default; no Qt in here.

The game already has colours — a deep navy night sky, cool blue cities, amber
HUD numerals, red-orange warheads — and they are chosen in ``app/renderer.cpp``
as float triples. The same values are written here as hex so the console reads as
the same product seen from the other side, rather than a generic tool pointed at
it. When one of them changes over there, change it here too; there is no shared
source for a palette that lives in a shader on one side and a stylesheet on the
other.

Each curve then gets a colour with a *reason*: the score is the game's HUD amber,
the baseline it is chasing is interceptor blue, and the diagnostics take the
remaining threat colours so they never compete with the two that matter.
"""

from __future__ import annotations

# ---- from the game ----------------------------------------------------------
#: The clear colour of the night sky (renderer.cpp: 0.05, 0.06, 0.12).
NIGHT = "#0d0f1f"
#: City blue (0.25, 0.62, 0.95) and the interceptor's cooler trail (0.6, 0.85, 1.0).
CITY = "#409ef2"
INTERCEPTOR = "#99d9ff"
#: Warhead red-orange (0.95, 0.35, 0.30), MIRV purple, smart-bomb green.
THREAT = "#f25a4d"
MIRV = "#994dd9"
SMART = "#4dcc66"
#: The HUD's wave counter (0.95, 0.75, 0.30) and its dimmed prompt text.
AMBER = "#f2bf4d"
MUTED = "#99a6b3"

# ---- console chrome ---------------------------------------------------------
PANEL = "#141a2c"
EDGE = "#232b45"
GRID = "#1e2540"
TEXT = "#e8edf7"

#: A run either is or is not still writing. Green means the file just moved.
AHEAD = SMART
BEHIND = THREAT

# ---- the curves -------------------------------------------------------------
SCORE = AMBER
BASELINE = INTERCEPTOR
RETURN = CITY
ENTROPY = SMART
VALUE = MIRV

#: Numbers are read by shape as much as by value, so they get a fixed pitch and
#: never reflow as digits change.
MONO = "Consolas, 'DejaVu Sans Mono', 'Liberation Mono', monospace"
SANS = "'Segoe UI', 'Inter', 'DejaVu Sans', sans-serif"


def stylesheet() -> str:
    """The whole console's QSS. One place, so nothing is styled twice."""
    return f"""
    QWidget {{
        background: {NIGHT};
        color: {TEXT};
        font-family: {SANS};
        font-size: 13px;
    }}
    QLabel[role="title"] {{
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 2px;
    }}
    QLabel[role="caption"] {{
        color: {MUTED};
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    QLabel[role="value"] {{
        font-family: {MONO};
        font-size: 26px;
        font-weight: 600;
    }}
    QLabel[role="note"] {{
        color: {MUTED};
        font-size: 11px;
    }}
    QLabel[role="placeholder"] {{
        color: {MUTED};
        font-size: 12px;
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 6px;
        padding: 14px;
    }}
    QFrame[role="tile"], QFrame[role="panel"] {{
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 6px;
    }}
    QListWidget {{
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 6px;
        outline: none;
        padding: 2px;
    }}
    QListWidget::item {{
        padding: 7px 9px;
        border-radius: 4px;
        color: {TEXT};
    }}
    QListWidget::item:selected {{
        background: {EDGE};
        color: {TEXT};
    }}
    QListWidget::item:hover {{
        background: {GRID};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {EDGE};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QStatusBar {{
        color: {MUTED};
        border-top: 1px solid {EDGE};
        font-size: 11px;
    }}
    QSplitter::handle {{ background: transparent; width: 10px; }}
    QToolTip {{
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {EDGE};
        padding: 4px;
    }}
    """
