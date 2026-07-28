# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What the machine is doing. No Qt in here, so the sampling is testable.

CPU and memory come from psutil. The GPU does not: there is no cross-vendor
Python API for it, so rather than picking one vendor or dropping the feature,
this talks to a small protocol and discovers whichever backend happens to be
installed (docs/ROADMAP.md, M8). Adding a vendor is one file in ``probes/``.

Everything here is *optional by construction*. A missing psutil, a missing
probe, a probe whose driver disappears mid-run — each is a normal state that the
panel explains, never an error that reaches the user as a traceback. The trainer
is a window onto a run; it must not be the thing that breaks.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

#: One module per vendor, tried in order; each exposes ``probe() -> GpuProbe | None``.
BACKENDS = ("md.ui.probes.nvidia", "md.ui.probes.amd")

#: What to say when none of them import. Name the exact interpreter because the
#: trainer and the training loop can deliberately run in a different Python from the
#: shell that launched them (tools.launch).
NO_PROBE = (
    f"no GPU probe in {sys.executable} — "
    f"{sys.executable} -m pip install nvidia-ml-py (NVIDIA) or amdsmi (ROCm)"
)
NO_PSUTIL = "psutil is not installed — pip install psutil"


@dataclass(frozen=True)
class GpuSample:
    """One reading from a GPU. Every field but the name may be unavailable.

    Per-field rather than all-or-nothing on purpose: vendor APIs differ in what
    they expose and in what they raise for, and a utilisation figure is still
    worth showing when the temperature call is the one that failed.
    """

    name: str
    utilisation: float | None = None  #: per cent
    memory_used: int | None = None  #: bytes
    memory_total: int | None = None  #: bytes
    temperature: float | None = None  #: degrees Celsius


class GpuProbe(Protocol):
    """A vendor backend: a name, and a reading on demand."""

    name: str

    def sample(self) -> GpuSample: ...


@dataclass(frozen=True)
class Sample:
    """One reading of the whole machine."""

    cpu: float  #: per cent across all cores
    memory_used: int  #: bytes
    memory_total: int  #: bytes
    gpu: GpuSample | None


def find_gpu_probe(backends: Sequence[str] = BACKENDS) -> GpuProbe | None:
    """The first vendor backend that both imports and finds a card."""
    return discover_gpu_probe(backends)[0]


def discover_gpu_probe(backends: Sequence[str] = BACKENDS) -> tuple[GpuProbe | None, str]:
    """The first working backend, or the most useful reason none worked."""
    notes: list[str] = []
    for name in backends:
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover — the probe modules are in-tree
            continue
        # Backends may explain the important middle state where their Python
        # package imports but its driver or hardware does not. Older/new vendor
        # modules need only expose probe(); adding diagnostics is optional.
        probe_with_note = getattr(module, "probe_with_note", None)
        if callable(probe_with_note):
            probe, note = cast(tuple[GpuProbe | None, str], probe_with_note())
        else:
            probe, note = cast(GpuProbe | None, module.probe()), ""
        if probe is not None:
            return probe, ""
        if note:
            notes.append(note)
    return None, " · ".join(notes) if notes else NO_PROBE


def _import_psutil() -> Any | None:
    try:
        return importlib.import_module("psutil")
    except ImportError:
        return None


class SystemMonitor:
    """Samples the machine, and degrades to saying why it cannot.

    The dependencies are constructor arguments so a test can hand in a fake
    psutil and a fake probe — including one that fails — without either being
    installed.
    """

    def __init__(
        self,
        *,
        psutil_module: Any | None = None,
        probe: GpuProbe | None = None,
        discover: bool = True,
    ) -> None:
        # `discover` governs both halves. Without that, passing no psutil could
        # not be *said* — it fell through to importing the real one — so the
        # "psutil is missing" behaviour was only ever exercised on a machine that
        # happened not to have it, and silently stopped being tested on one that did.
        self._psutil = (
            psutil_module if psutil_module is not None else (_import_psutil() if discover else None)
        )
        self._probe: GpuProbe | None
        if probe is not None:
            self._probe, self._gpu_note = probe, ""
        elif discover:
            self._probe, self._gpu_note = discover_gpu_probe()
        else:
            self._probe, self._gpu_note = None, NO_PROBE

    @property
    def available(self) -> bool:
        """Whether CPU and memory can be read at all."""
        return self._psutil is not None

    @property
    def gpu_name(self) -> str | None:
        return None if self._probe is None else self._probe.name

    @property
    def gpu_note(self) -> str:
        """Why there is no GPU reading — empty when there is one."""
        return self._gpu_note

    def sample(self) -> Sample | None:
        """A reading, or ``None`` when psutil is not installed."""
        if self._psutil is None:
            return None
        memory = self._psutil.virtual_memory()
        return Sample(
            # interval=None measures since the previous call rather than
            # sleeping, which is the only version of this a UI timer can use.
            cpu=float(self._psutil.cpu_percent(interval=None)),
            memory_used=int(memory.total - memory.available),
            memory_total=int(memory.total),
            gpu=self._gpu(),
        )

    def _gpu(self) -> GpuSample | None:
        if self._probe is None:
            return None
        try:
            return self._probe.sample()
        except Exception as error:  # noqa: BLE001 — any vendor error, one outcome
            # A driver can be reset, or a card removed, mid-run. Drop the probe
            # rather than raising once a second forever, and say what happened.
            self._gpu_note = f"{self._probe.name} stopped responding ({error})"
            self._probe = None
            return None
