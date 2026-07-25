# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# pyright: reportMissingImports=false
"""The machine's own row: CPU, memory, and the GPU when something can read it.

It lives under the recordings list rather than in the tile row, because that
column already had the space and the curve must not lose any. Three thin bars
answer three questions at a glance — is the CPU saturated, is memory about to
run out, is the accelerator doing anything at all — and nothing here needs to be
read precisely, so the numbers stay small and the bars do the talking.

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
        for meter in (self._cpu, self._memory, self._gpu):
            layout.addWidget(meter)

        # One line for whichever of the two is not available. It names the
        # package to install, because "no GPU" and "no way to ask" are different
        # states and only one of them is worth doing something about.
        self._note = QLabel()
        self._note.setProperty("role", "note")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self.refresh()

    def refresh(self) -> None:
        sample = self._monitor.sample()
        if sample is None:
            for meter in (self._cpu, self._memory, self._gpu):
                meter.set(None, "—")
            self._note.setText(NO_PSUTIL)
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
            self._note.setText(self._monitor.gpu_note)
            return
        busy = "—" if gpu.utilisation is None else f"{gpu.utilisation:.0f}%"
        self._gpu.set(gpu.utilisation, busy)
        self._note.setText(_gpu_detail(gpu))


def _gpu_detail(gpu: GpuSample) -> str:
    """The card, its VRAM and its temperature — whichever of those it reports."""
    parts = [gpu.name]
    if gpu.memory_total:
        parts.append(f"{(gpu.memory_used or 0) / GB:.1f} / {gpu.memory_total / GB:.0f} GB")
    if gpu.temperature is not None:
        parts.append(f"{gpu.temperature:.0f} °C")
    return " · ".join(parts)
