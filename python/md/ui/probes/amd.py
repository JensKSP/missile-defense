# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""AMD, through ROCm's own bindings — ``amdsmi`` first, ``pyrsmi`` as a fallback.

Both ship with ROCm rather than as ordinary wheels, which makes this the
Linux-and-ROCm path in practice: on Windows an AMD card has no supported Python
telemetry API at all, so the panel says so instead of pretending. That is the
whole reason the probe is pluggable — a vendor gap is a missing file here, not a
hole in the console.

**Not verified against real hardware.** There is no AMD/ROCm machine in this
project's loop, so this is written from the published API and every field is
read defensively: a call that does not exist or returns a different shape costs
that one field, not the reading and certainly not the window.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from ..system import GpuSample
from . import maybe

#: ROCm reports VRAM in megabytes; the panel works in bytes throughout.
MB = 1024 * 1024


def probe() -> _AmdSmi | _Rsmi | None:
    """A probe for the first AMD card, or ``None`` if there is not one."""
    return _amdsmi() or _pyrsmi()


def _amdsmi() -> _AmdSmi | None:
    try:
        amdsmi = importlib.import_module("amdsmi")
    except ImportError:
        return None
    try:
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return None
        board = maybe(lambda: amdsmi.amdsmi_get_gpu_board_info(handles[0]))
    except Exception:  # noqa: BLE001 — installed package, no driver or no card
        return None
    name = "AMD GPU"
    if isinstance(board, dict):
        info = cast(dict[str, Any], board)
        name = str(info.get("product_name") or info.get("marketing_name") or name)
    return _AmdSmi(amdsmi, handles[0], name)


def _pyrsmi() -> _Rsmi | None:
    try:
        rocml = importlib.import_module("pyrsmi.rocml")
    except ImportError:
        return None
    try:
        rocml.smi_initialize()
        if rocml.smi_get_device_count() < 1:
            return None
    except Exception:  # noqa: BLE001
        return None
    name = maybe(lambda: str(rocml.smi_get_device_name(0))) or "AMD GPU"
    return _Rsmi(rocml, name)


class _AmdSmi:
    def __init__(self, amdsmi: Any, handle: Any, name: str) -> None:
        self._amdsmi = amdsmi
        self._handle = handle
        self.name = name

    def sample(self) -> GpuSample:
        activity = maybe(lambda: self._amdsmi.amdsmi_get_gpu_activity(self._handle))
        vram = maybe(lambda: self._amdsmi.amdsmi_get_gpu_vram_usage(self._handle))
        return GpuSample(
            name=self.name,
            utilisation=_field(activity, "gfx_activity"),
            memory_used=_bytes(_field(vram, "vram_used")),
            memory_total=_bytes(_field(vram, "vram_total")),
            temperature=maybe(self._temperature),
        )

    def _temperature(self) -> float:
        kind = self._amdsmi.AmdSmiTemperatureType.EDGE
        metric = self._amdsmi.AmdSmiTemperatureMetric.CURRENT
        return float(self._amdsmi.amdsmi_get_temp_metric(self._handle, kind, metric))


class _Rsmi:
    """The older ROCm SMI wrapper. Utilisation and memory only."""

    def __init__(self, rocml: Any, name: str) -> None:
        self._rocml = rocml
        self.name = name

    def sample(self) -> GpuSample:
        return GpuSample(
            name=self.name,
            utilisation=maybe(lambda: float(self._rocml.smi_get_device_utilization(0))),
            memory_used=maybe(lambda: int(self._rocml.smi_get_device_memory_used(0))),
            memory_total=maybe(lambda: int(self._rocml.smi_get_device_memory_total(0))),
        )


def _field(reading: Any, key: str) -> float | None:
    """One entry of a dict the vendor may have returned in another shape."""
    if not isinstance(reading, dict):
        return None
    values = cast(dict[str, Any], reading)
    if key not in values:
        return None
    return maybe(lambda: float(values[key]))


def _bytes(megabytes: float | None) -> int | None:
    return None if megabytes is None else int(megabytes * MB)
