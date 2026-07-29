# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The machine's own row: CPU, memory, and the GPU when something can read it.

It lives under the recordings list rather than in the tile row, because that
column already had the space and the curve must not lose any. Four thin bars
answer four questions at a glance — is the CPU saturated, is memory about to run
out, is the accelerator doing anything at all, and is its memory about to run out
— and nothing here needs to be read precisely, so the numbers stay small and the
bars do the talking.

**VRAM gets a bar of its own, under the GPU's load.** It was a fragment of the
caption line underneath for a long time, which is the wrong weight for it: GPU
*utilisation* is a curiosity — it tells you the card is busy, which you knew,
because you started a run — while GPU *memory* is the number that ends a run
eight hours in. It is also the one you check before starting a second one
(docs/TRAINING.md, how much GPU memory a run needs), and a figure you have to
read out of a sentence is a figure you do not check.

A bar changes colour as it fills. That is the only thing on this strip worth
looking up from the curve for: amber means the machine is working hard, red
means it is the thing limiting the run.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from . import theme
from .system import NO_PSUTIL, GpuSample, Sample, SystemMonitor

#: Where a meter stops being blue. Below the first it is simply busy.
BUSY = 70.0
FULL = 90.0

#: Bytes in a gibibyte — memory is read in GB and nothing else here is.
GB = 1024**3


class Meter(QWidget):
    """A labelled bar with a value beside it."""

    def __init__(self, caption: str) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._caption = QLabel(caption)
        self._caption.setProperty("role", "caption")
        self._caption.setFixedWidth(34)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._value = QLabel("—")
        self._value.setProperty("role", "note")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value.setMinimumWidth(110)
        row.addWidget(self._caption)
        row.addWidget(self._bar, stretch=1)
        row.addWidget(self._value)
        self._colour = ""

    def set(self, percent: float | None, text: str) -> None:
        self._value.setText(text)
        self._bar.setVisible(percent is not None)
        if percent is None:
            return
        self._bar.setValue(int(max(0.0, min(100.0, percent))))
        colour = theme.THREAT if percent >= FULL else theme.AMBER if percent >= BUSY else theme.CITY
        if colour != self._colour:  # restyling every second would repaint the world
            self._bar.setStyleSheet(f"QProgressBar::chunk {{ background: {colour}; }}")
            self._colour = colour


class SystemPanel(QWidget):
    """CPU, memory and GPU, or a line saying why one of them is missing."""

    def __init__(self, monitor: SystemMonitor | None = None) -> None:
        super().__init__()
        self._monitor = monitor if monitor is not None else SystemMonitor()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        caption = QLabel("system")
        caption.setProperty("role", "caption")
        layout.addWidget(caption)

        self._cpu = Meter("cpu")
        self._memory = Meter("ram")
        self._gpu = Meter("gpu")
        # Directly under the load it belongs to, and captioned as the thing a
        # trainer actually says ("vram"), not as "gpu ram".
        self._vram = Meter("vram")
        for meter in (self._cpu, self._memory, self._gpu, self._vram):
            layout.addWidget(meter)

        # One line for whichever of the two is not available. It names the
        # package to install, because "no GPU" and "no way to ask" are different
        # states and only one of them is worth doing something about.
        self._note = QLabel()
        self._note.setProperty("role", "note")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self.refresh()

    def _say(self, text: str) -> None:
        """Show `text` in the note, wrappable even where it holds a path.

        A backslash is no break opportunity to Qt's line breaking, so a
        Windows interpreter path is one unbreakable token — and the widest
        token is the minimum width a word-wrapped QLabel still demands. This
        note names `sys.executable`, which put a floor of the path's full
        length under the whole console window: it could not be made narrower
        than ~700px, on displays that had no ~700px to give (reported
        2026-07-29). A zero-width space after each backslash is invisible,
        changes nothing anyone reads or retypes, and lets the wrap do its job.
        """
        self._note.setText(text.replace("\\", "\\" + "\u200b"))

    def refresh(self) -> None:
        sample = self._monitor.sample()
        if sample is None:
            for meter in (self._cpu, self._memory, self._gpu, self._vram):
                meter.set(None, "—")
            self._say(NO_PSUTIL)
            return
        self._show(sample)

    def _show(self, sample: Sample) -> None:
        self._cpu.set(sample.cpu, f"{sample.cpu:.0f}%")
        used, total = sample.memory_used / GB, sample.memory_total / GB
        share = (100.0 * sample.memory_used / sample.memory_total) if sample.memory_total else 0.0
        self._memory.set(share, f"{used:.1f} / {total:.0f} GB")

        gpu = sample.gpu
        if gpu is None:
            self._gpu.set(None, "—")
            self._vram.set(None, "—")
            self._say(self._monitor.gpu_note)
            return
        busy = "—" if gpu.utilisation is None else f"{gpu.utilisation:.0f}%"
        self._gpu.set(gpu.utilisation, busy)
        self._vram.set(*_vram_reading(gpu))
        self._say(_gpu_detail(gpu))


def _vram_reading(gpu: GpuSample) -> tuple[float | None, str]:
    """The VRAM bar's fill and its label, in the same shape as the RAM row.

    A card that reports a total but no usage gets its size and no bar: "32 GB,
    and nobody will say how much of it is left" is a truer thing to show than a
    bar sitting at zero, which reads as an empty card.
    """
    if not gpu.memory_total:
        return None, "—"
    total = gpu.memory_total / GB
    if gpu.memory_used is None:
        return None, f"— / {total:.0f} GB"
    used = gpu.memory_used / GB
    return 100.0 * gpu.memory_used / gpu.memory_total, f"{used:.1f} / {total:.0f} GB"


def _gpu_detail(gpu: GpuSample) -> str:
    """The card and its temperature. Its memory is a meter of its own now."""
    parts = [gpu.name]
    if gpu.temperature is not None:
        parts.append(f"{gpu.temperature:.0f} °C")
    return " · ".join(parts)
