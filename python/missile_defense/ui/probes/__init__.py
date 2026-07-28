# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""GPU backends, one file per vendor.

Each module exposes ``probe() -> GpuProbe | None`` and may also expose
``probe_with_note() -> tuple[GpuProbe | None, str]``. The latter distinguishes
an absent package from one whose native driver or hardware is unavailable — an
NVIDIA binding can be installed on a machine whose driver is not loaded, and a
soft import alone would report the wrong fix.

Nothing here is imported eagerly: :func:`missile_defense.ui.system.find_gpu_probe` walks the
list, and a vendor whose package is missing costs an ImportError that is caught.
Adding a vendor is therefore one new file and one entry in ``BACKENDS`` — never a
change to the panel.
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

__all__ = ["maybe"]


def maybe(read: Callable[[], T]) -> T | None:
    """A reading the driver refuses is *missing*, not fatal to the sample.

    Vendor libraries raise their own exception hierarchies for a field that a
    particular card does not report — a laptop GPU with no temperature sensor
    exposed, say. Losing that field is not a reason to lose the utilisation
    figure next to it, and it is certainly not a reason to break the window.
    """
    try:
        return read()
    except Exception:  # noqa: BLE001 — every vendor error means the same here
        return None
