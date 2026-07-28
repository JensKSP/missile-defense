# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The trainer's palette, lifted from the game. Dark by default; no Qt in here.

The game already has colours — a deep navy night sky, cool blue cities, amber
HUD numerals, red-orange warheads — and they are chosen in ``app/renderer.cpp``
as float triples. The same values are written here as hex so the trainer reads as
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

# ---- trainer chrome ---------------------------------------------------------
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
    """The whole trainer's QSS. One place, so nothing is styled twice."""
    return f"""
    QWidget {{
        background: {NIGHT};
        color: {TEXT};
        font-family: {SANS};
        font-size: 13px;
    }}
    /* Labels sit on panels and tiles, so they take their parent's colour. Every
       role below that wants a background of its own says so explicitly. */
    QLabel {{
        background: transparent;
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
    /* The reward equation, in the config panel. Monospaced because it is an
       equation and the terms should line up under each other, and full-strength
       text because it is the headline of that panel rather than a footnote to
       it — the surrounding prose is what carries `role="note"`. */
    QLabel[role="formula"] {{
        font-family: {MONO};
        font-size: 12px;
    }}
    /* The statistics in a chart's corner: a footnote to the curve, so it is the
       smallest text in the window and never competes with the line itself. */
    QLabel[role="stat"] {{
        color: {MUTED};
        font-family: {MONO};
        font-size: 10px;
    }}
    /* The value under the pointer. A chip that has to stay readable over the
       curve it is floating on, so unlike the footnote it carries a background. */
    QLabel[role="readout"] {{
        color: {TEXT};
        font-family: {MONO};
        font-size: 11px;
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 4px;
        padding: 3px 6px;
    }}
    /* A note that is telling you something is about to go wrong. Amber rather
       than red: the estimate behind it is a model, and the button it sits above
       is still the right one to press if you know better than the model. */
    QLabel[role="warning"] {{
        color: {AMBER};
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
    QLabel[role="preview"] {{
        font-family: {MONO};
        font-size: 11px;
        color: {MUTED};
        background: {NIGHT};
        border: 1px solid {EDGE};
        border-radius: 6px;
        padding: 8px;
    }}
    QPushButton, QToolButton {{
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 6px;
        padding: 6px 14px;
        color: {TEXT};
    }}
    QPushButton:hover, QToolButton:hover {{ background: {GRID}; }}
    QPushButton:disabled {{ color: {MUTED}; border-color: {GRID}; }}
    /* Keyboard focus, visibly. A stylesheet that sets `border` on a control
       replaces the platform's focus rectangle with nothing, so every widget
       styled above became invisible to Tab — which is not a cosmetic problem:
       it makes the whole trainer unusable without a mouse. One rule, on every
       focusable thing, in the accent that means "this is where you are". */
    QPushButton:focus, QToolButton:focus, QComboBox:focus, QLineEdit:focus,
    QCheckBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTabBar::tab:focus,
    QAbstractItemView:focus, QListWidget:focus, QTableWidget:focus {{
        border: 1px solid {INTERCEPTOR};
        outline: none;
    }}
    /* The selected row of a table that does *not* have focus stays visible but
       recedes: two panels both showing a bright selection is two claims about
       where the keyboard is, and only one of them can be true. */
    QTableWidget::item:selected:!active, QListWidget::item:selected:!active {{
        background: {GRID};
        color: {TEXT};
    }}
    /* Buttons that share a row with a caption rather than the control bar —
       they belong to the panel under them, so they must not out-shout it. */
    QPushButton[role="compact"] {{
        padding: 2px 9px;
        font-size: 11px;
    }}
    /* The version stamp beside the title. It is a button because it opens
       About, and it is styled as a caption because it must not read as one:
       the control bar is three affordances and this is not a fourth. */
    QPushButton[role="version"] {{
        background: transparent;
        border: none;
        padding: 0px 4px;
        color: {MUTED};
        font-size: 11px;
        letter-spacing: 1px;
    }}
    QPushButton[role="version"]:hover {{ color: {TEXT}; }}
    /* The one button that changes meaning — Start, then Pause, then Resume. */
    QPushButton[role="primary"] {{
        background: {CITY};
        border: 1px solid {CITY};
        color: {NIGHT};
        font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{ background: {INTERCEPTOR}; border-color: {INTERCEPTOR}; }}
    QPushButton[role="primary"]:disabled {{
        background: {PANEL};
        border-color: {GRID};
        color: {MUTED};
    }}
    QComboBox {{
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 6px;
        padding: 5px 10px;
        color: {TEXT};
    }}
    QComboBox:hover {{ background: {GRID}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {PANEL};
        border: 1px solid {EDGE};
        selection-background-color: {EDGE};
        color: {TEXT};
        padding: 4px;
    }}
    QLineEdit, QSpinBox {{
        background: {NIGHT};
        border: 1px solid {EDGE};
        border-radius: 4px;
        padding: 4px 6px;
        color: {TEXT};
        selection-background-color: {EDGE};
        font-family: {MONO};
    }}
    /* The one control that is dragged rather than pressed (the eval interval).
       Built from the same parts as the meters below — dark groove, lit fill — so
       it reads as part of this window rather than as a stray widget from
       another program, and so "how far along" is legible at a glance. */
    QSlider::groove:horizontal {{
        background: {NIGHT};
        border: 1px solid {EDGE};
        border-radius: 3px;
        height: 4px;
    }}
    QSlider::sub-page:horizontal {{
        background: {CITY};
        border: 1px solid {CITY};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {TEXT};
        border: 1px solid {TEXT};
        width: 9px;
        margin: -5px 0;
        border-radius: 5px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {INTERCEPTOR};
        border-color: {INTERCEPTOR};
    }}
    QSlider::sub-page:horizontal:disabled {{ background: {GRID}; border-color: {GRID}; }}
    QSlider::handle:horizontal:disabled {{ background: {GRID}; border-color: {EDGE}; }}
    QSlider::tick-mark {{ color: {EDGE}; }}
    /* Meters: a groove and a fill, no text — the number lives beside it. The
       fill colour is set per-widget, because it means something (see meters.py). */
    QProgressBar {{
        background: {NIGHT};
        border: 1px solid {EDGE};
        border-radius: 3px;
    }}
    QProgressBar::chunk {{
        background: {CITY};
        border-radius: 2px;
    }}
    QPlainTextEdit {{
        background: {NIGHT};
        border: 1px solid {EDGE};
        border-radius: 6px;
        color: {MUTED};
        font-family: {MONO};
        font-size: 11px;
    }}
    QListWidget {{
        background: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 6px;
        outline: none;
        padding: 2px;
    }}
    /* Tight rows: this list is the column's scarce space, and the run adds an
       episode every few minutes — how many fit is worth more than the air. */
    QListWidget::item {{
        padding: 5px 9px;
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
    /* The two views of one run (training curves, statistics).
       Quiet, but *findable*: the first attempt used caption type on a
       transparent strip and it read as a heading rather than a control — the
       human's first reaction to it was "where is the statistics tab?", which is
       the whole review this needed. So the unselected tab now carries the same
       panel fill as every other pressable thing in the window, and the selected
       one is lit from below in the accent. Still flat, still no raised borders;
       it just no longer disappears into the background it sits on. */
    QTabWidget::pane {{ border: none; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{
        background: {PANEL};
        color: {MUTED};
        border: 1px solid {EDGE};
        border-bottom: 2px solid {EDGE};
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 7px 18px 5px 18px;
        margin-right: 4px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    QTabBar::tab:selected {{
        color: {TEXT};
        background: {GRID};
        border-color: {EDGE};
        border-bottom: 2px solid {CITY};
    }}
    QTabBar::tab:hover:!selected {{ color: {TEXT}; background: {GRID}; }}
    QSplitter::handle {{ background: transparent; width: 10px; }}
    QToolTip {{
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {EDGE};
        padding: 4px;
    }}
    """
