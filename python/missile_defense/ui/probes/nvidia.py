# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""NVIDIA, through NVML (``pip install nvidia-ml-py``).

The first card only. A second one would need a picker in the panel, and the
question this answers — "is the accelerator doing anything?" — is asked about
the one training is on.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..system import GpuSample
from . import maybe


def probe() -> _Nvml | None:
    """A probe for the first NVIDIA card, or ``None`` if there is not one."""
    found, _ = probe_with_note()
    return found


def probe_with_note() -> tuple[_Nvml | None, str]:
    """A probe and, when NVML is installed but unusable, the reason why.

    A missing binding is ordinary discovery: another vendor may work, so it
    contributes no diagnosis. Once the binding imports, however, hiding an NVML
    error behind "not installed" sends the reader toward the one thing they
    already did. Keep that distinction for the system panel.
    """
    try:
        nvml = importlib.import_module("pynvml")
    except ImportError:
        return None, ""
    except Exception as error:  # noqa: BLE001 — wrapper present, native library absent
        return None, f"NVIDIA telemetry unavailable ({error})"
    try:
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        name = nvml.nvmlDeviceGetName(handle)
    except Exception as error:  # noqa: BLE001 — installed package, no driver or no card
        return None, f"NVIDIA telemetry unavailable ({error})"
    return _Nvml(nvml, handle, _text(name)), ""


class _Nvml:
    def __init__(self, nvml: Any, handle: Any, name: str) -> None:
        self._nvml = nvml
        self._handle = handle
        self.name = name

    def sample(self) -> GpuSample:
        memory = maybe(lambda: self._nvml.nvmlDeviceGetMemoryInfo(self._handle))
        rates = maybe(lambda: self._nvml.nvmlDeviceGetUtilizationRates(self._handle))
        return GpuSample(
            name=self.name,
            utilisation=None if rates is None else float(rates.gpu),
            memory_used=None if memory is None else int(memory.used),
            memory_total=None if memory is None else int(memory.total),
            temperature=maybe(self._temperature),
        )

    def _temperature(self) -> float:
        sensor = self._nvml.NVML_TEMPERATURE_GPU
        return float(self._nvml.nvmlDeviceGetTemperature(self._handle, sensor))


def _text(name: Any) -> str:
    """NVML returned bytes for years and str more recently. Accept both."""
    return name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
