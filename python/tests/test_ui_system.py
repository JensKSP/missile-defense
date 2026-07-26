# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the machine monitor and its pluggable GPU probes.

None of the vendor packages are installed here — that is the normal state, and
half of what these tests pin. The other half is the mapping from each vendor's
API onto `GpuSample`, which is checked by handing the probe a stand-in module:
the shapes come from the published APIs, so what is tested is our translation of
them, not the vendor's.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from md.ui.probes import amd, maybe, nvidia
from md.ui.system import NO_PROBE, GpuSample, SystemMonitor, discover_gpu_probe, find_gpu_probe


class FakeMemory:
    total = 32 * 1024**3
    available = 20 * 1024**3


class FakePsutil:
    def __init__(self) -> None:
        self.calls = 0

    def cpu_percent(self, interval: float | None = None) -> float:
        self.calls += 1
        assert interval is None, "a UI timer cannot afford a blocking sample"
        return 42.5

    def virtual_memory(self) -> FakeMemory:
        return FakeMemory()


class FakeProbe:
    name = "Fake GPU"

    def sample(self) -> GpuSample:
        return GpuSample(name=self.name, utilisation=61.0)


class BrokenProbe:
    name = "Flaky GPU"

    def sample(self) -> GpuSample:
        raise RuntimeError("driver reset")


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


# ---- the monitor ------------------------------------------------------------


def test_without_psutil_there_is_no_sample_and_a_reason() -> None:
    monitor = SystemMonitor(psutil_module=None, discover=False)
    assert not monitor.available
    assert monitor.sample() is None


def test_cpu_and_memory_come_back_as_bytes_and_per_cent() -> None:
    monitor = SystemMonitor(psutil_module=FakePsutil(), discover=False)
    sample = monitor.sample()
    assert sample is not None
    assert sample.cpu == 42.5
    # Used is total minus *available*, not total minus free: cache is reclaimable
    # and counting it as used reads as a machine about to swap when it is not.
    assert sample.memory_used == 12 * 1024**3
    assert sample.memory_total == 32 * 1024**3
    assert sample.gpu is None


def test_no_probe_says_what_to_install() -> None:
    monitor = SystemMonitor(psutil_module=FakePsutil(), discover=False)
    assert monitor.gpu_name is None
    assert "nvidia-ml-py" in monitor.gpu_note and "amdsmi" in monitor.gpu_note
    assert sys.executable in monitor.gpu_note
    assert monitor.gpu_note == NO_PROBE


def test_a_probe_that_works_is_reported_with_the_sample() -> None:
    monitor = SystemMonitor(psutil_module=FakePsutil(), probe=FakeProbe())
    sample = monitor.sample()
    assert sample is not None and sample.gpu is not None
    assert sample.gpu.utilisation == 61.0
    assert monitor.gpu_name == "Fake GPU"
    assert monitor.gpu_note == ""


def test_a_probe_that_breaks_mid_run_is_dropped_not_raised() -> None:
    # A driver can be reset or a card removed while the window is open. Raising
    # once a second forever is the one outcome that must not happen.
    monitor = SystemMonitor(psutil_module=FakePsutil(), probe=BrokenProbe())
    first = monitor.sample()
    assert first is not None and first.gpu is None
    assert "stopped responding" in monitor.gpu_note
    assert monitor.sample() is not None  # and it keeps working without the GPU


def test_discovery_returns_nothing_when_no_vendor_package_is_installed() -> None:
    # The state on this machine, and on most: neither ROCm nor CUDA present.
    assert find_gpu_probe() is None


def test_an_unimportable_backend_is_skipped_not_fatal() -> None:
    assert find_gpu_probe(("md.ui.probes.does_not_exist",)) is None


# ---- the vendor backends ----------------------------------------------------


def test_nvidia_maps_nvml_onto_the_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = types.SimpleNamespace(used=3 * 1024**3, total=24 * 1024**3)
    fake = _module(
        "pynvml",
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: "handle",
        nvmlDeviceGetName=lambda handle: b"NVIDIA RTX 4090",  # NVML returns bytes
        nvmlDeviceGetUtilizationRates=lambda handle: types.SimpleNamespace(gpu=73),
        nvmlDeviceGetMemoryInfo=lambda handle: memory,
        nvmlDeviceGetTemperature=lambda handle, sensor: 61,
        NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    probe = nvidia.probe()
    assert probe is not None
    sample = probe.sample()
    assert sample.name == "NVIDIA RTX 4090"
    assert sample.utilisation == 73.0
    assert (sample.memory_used, sample.memory_total) == (3 * 1024**3, 24 * 1024**3)
    assert sample.temperature == 61.0


def test_nvidia_with_the_package_but_no_card_is_no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # pynvml installed next to an AMD card, say: the import succeeds and the
    # driver call is what fails, so a soft import alone would not notice.
    def explode() -> None:
        raise RuntimeError("NVML Shared Library Not Found")

    monkeypatch.setitem(sys.modules, "pynvml", _module("pynvml", nvmlInit=explode))
    assert nvidia.probe() is None


def test_nvidia_with_the_package_but_no_driver_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode() -> None:
        raise RuntimeError("NVML Shared Library Not Found")

    monkeypatch.setitem(sys.modules, "pynvml", _module("pynvml", nvmlInit=explode))
    probe, note = discover_gpu_probe(("md.ui.probes.nvidia",))

    assert probe is None
    assert note == "NVIDIA telemetry unavailable (NVML Shared Library Not Found)"
    assert "install" not in note


def test_a_field_the_driver_refuses_costs_only_that_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_sensor(handle: object, sensor: object) -> float:
        raise RuntimeError("this card has no temperature sensor")

    fake = _module(
        "pynvml",
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: "handle",
        nvmlDeviceGetName=lambda handle: "NVIDIA T4",
        nvmlDeviceGetUtilizationRates=lambda handle: types.SimpleNamespace(gpu=12),
        nvmlDeviceGetMemoryInfo=lambda handle: types.SimpleNamespace(used=1, total=2),
        nvmlDeviceGetTemperature=no_sensor,
        NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    probe = nvidia.probe()
    assert probe is not None
    sample = probe.sample()
    assert sample.temperature is None
    assert sample.utilisation == 12.0  # the rest of the reading survives


def test_amd_maps_amdsmi_onto_the_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _module(
        "amdsmi",
        amdsmi_init=lambda: None,
        amdsmi_get_processor_handles=lambda: ["handle"],
        amdsmi_get_gpu_board_info=lambda handle: {"product_name": "AMD RX 7900 XTX"},
        amdsmi_get_gpu_activity=lambda handle: {"gfx_activity": 88},
        # ROCm reports VRAM in megabytes; the panel works in bytes.
        amdsmi_get_gpu_vram_usage=lambda handle: {"vram_used": 2048, "vram_total": 24576},
        amdsmi_get_temp_metric=lambda handle, kind, metric: 54,
        AmdSmiTemperatureType=types.SimpleNamespace(EDGE=0),
        AmdSmiTemperatureMetric=types.SimpleNamespace(CURRENT=0),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake)

    probe = amd.probe()
    assert probe is not None
    sample = probe.sample()
    assert sample.name == "AMD RX 7900 XTX"
    assert sample.utilisation == 88.0
    assert sample.memory_used == 2048 * 1024 * 1024
    assert sample.memory_total == 24576 * 1024 * 1024
    assert sample.temperature == 54.0


def test_amd_falls_back_to_pyrsmi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "amdsmi", _module("amdsmi"))  # present but unusable
    rocml = _module(
        "pyrsmi.rocml",
        smi_initialize=lambda: None,
        smi_get_device_count=lambda: 1,
        smi_get_device_name=lambda index: "AMD MI210",
        smi_get_device_utilization=lambda index: 37,
        smi_get_device_memory_used=lambda index: 1024,
        smi_get_device_memory_total=lambda index: 4096,
    )
    monkeypatch.setitem(sys.modules, "pyrsmi", _module("pyrsmi", rocml=rocml))
    monkeypatch.setitem(sys.modules, "pyrsmi.rocml", rocml)

    probe = amd.probe()
    assert probe is not None
    sample = probe.sample()
    assert (sample.name, sample.utilisation) == ("AMD MI210", 37.0)
    assert sample.temperature is None  # this wrapper does not report one


def test_amd_with_the_package_but_no_driver_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode() -> None:
        raise RuntimeError("libamd_smi.so not found")

    monkeypatch.setitem(sys.modules, "amdsmi", _module("amdsmi", amdsmi_init=explode))
    # Prevent a real fallback package on the test machine from hiding the error.
    monkeypatch.setitem(sys.modules, "pyrsmi", _module("pyrsmi"))
    probe, note = discover_gpu_probe(("md.ui.probes.amd",))

    assert probe is None
    assert note == "AMD telemetry unavailable (libamd_smi.so not found)"
    assert "install" not in note


def test_a_vendor_returning_an_unexpected_shape_is_survivable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reason every field is read defensively: these APIs are versioned, and
    # a dict that became a namespace must cost a field, not the window.
    fake = _module(
        "amdsmi",
        amdsmi_init=lambda: None,
        amdsmi_get_processor_handles=lambda: ["handle"],
        amdsmi_get_gpu_board_info=lambda handle: object(),
        amdsmi_get_gpu_activity=lambda handle: types.SimpleNamespace(gfx_activity=50),
        amdsmi_get_gpu_vram_usage=lambda handle: None,
        amdsmi_get_temp_metric=lambda handle, kind, metric: "warm",
        AmdSmiTemperatureType=types.SimpleNamespace(EDGE=0),
        AmdSmiTemperatureMetric=types.SimpleNamespace(CURRENT=0),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake)

    probe = amd.probe()
    assert probe is not None
    sample = probe.sample()
    assert sample.name == "AMD GPU"  # the board info was not a dict
    assert sample.utilisation is None
    assert sample.memory_total is None
    assert sample.temperature is None


def test_maybe_returns_none_instead_of_raising() -> None:
    assert maybe(lambda: 1 / 0) is None
    assert maybe(lambda: 7) == 7
