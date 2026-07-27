# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The renderer against `VK_LAYER_KHRONOS_validation`, with nothing excused.

`harness.validation_errors` has no allow-list. It had one entry for most of this
project's life — `VUID-vkAcquireNextImageKHR-semaphore-01779` — justified by the
observation that `QVulkanWindow` owns the swapchain and therefore the semaphore.
The observation was right; the conclusion that it could not be fixed here was
not. Qt hardcodes two sets of frame resources and then takes whatever swapchain
the driver's `minImageCount` demands, which is three everywhere tested, so it
reuses an acquire semaphore whose wait has not retired. An application cannot
reorder Qt's acquire, but it can decline to run ahead of it, and a
`vkQueueWaitIdle` in `Renderer::submit` does exactly that.

That leaves a workaround whose necessity is invisible from the code it sits in,
which is how workarounds outlive their cause. So this file asserts both halves:

* the game raises nothing, across every scenario that renders;
* a bare `QVulkanWindow` *still does*, so the workaround is still earning its
  frame of latency — and the day it does not, this fails and says to delete it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from .harness import (
    PROJECT_ROOT,
    _display_wrapper,
    app_environ,
    assert_clean,
    needs_app,
    needs_display,
    run_app,
    validation_errors,
)

pytestmark = [pytest.mark.e2e, needs_app, needs_display]


def _baseline_binary() -> Path | None:
    """The bare-QVulkanWindow witness, if this build produced one.

    Built only under `MD_VULKAN_VALIDATION`, which is the same condition that
    makes the game's own messages observable — so when it is missing, there is
    nothing for this file to check either.
    """
    for build in ("build/debug/app", "build/release/app"):
        candidate = PROJECT_ROOT / build / "md_vulkan_baseline"
        if candidate.exists():
            return candidate
    return None


# The scenarios the 1.0 plan requires the runtime check to cover. Written as
# flags rather than as a list of screens because the flags are the contract the
# game actually exposes; a screen can be renamed without this going stale.
RENDERING_SCENARIOS = pytest.mark.parametrize(
    ("args", "what"),
    [
        ((), "the menu"),
        (("--play",), "gameplay"),
        (("--watch-scripted", "low"), "watching the scripted agent"),
        (("--watch-scripted", "high"), "watching the expert scripted agent"),
    ],
    ids=["menu", "play", "watch-low", "watch-high"],
)


@RENDERING_SCENARIOS
def test_no_scenario_raises_a_validation_message(
    args: tuple[str, ...], what: str, tmp_path: Path
) -> None:
    run = run_app(*args, frames=240, sandbox=tmp_path)
    assert_clean(run)
    assert not validation_errors(run), f"{what} raised Vulkan validation messages"


def test_the_bundled_model_renders_clean(tmp_path: Path) -> None:
    """The learned-policy path, which is the one a first-time user is shown.

    Skipped rather than failed when no model is bundled: `models/pretrained.mdp`
    is optional by design (see `app/CMakeLists.txt`), and a source tree without
    one is a legitimate state.
    """
    model = PROJECT_ROOT / "models" / "pretrained.mdp"
    if not model.exists():
        pytest.skip("no bundled model in this tree — nothing to render")
    run = run_app("--watch-model", str(model), frames=240, sandbox=tmp_path)
    assert_clean(run)
    assert not validation_errors(run)


def test_the_qt_workaround_is_still_necessary(tmp_path: Path) -> None:
    """Fails when Qt is fixed, which is the only time it should.

    This is the evidence that `VUID-vkAcquireNextImageKHR-semaphore-01779` is
    upstream: the witness contains no line of this project, renders nothing, and
    still trips it. If that ever stops being true, the `vkQueueWaitIdle` in
    `Renderer::submit` is buying nothing and costs a frame of CPU/GPU overlap.
    """
    baseline = _baseline_binary()
    if baseline is None:
        pytest.skip("md_vulkan_baseline not built — configure with -DMD_VULKAN_VALIDATION=ON")

    # Through the same virtual X server the game gets: the witness needs a real
    # surface for a real swapchain, which is the whole point of it.
    wrapper = _display_wrapper()
    assert wrapper is not None, "no way to render this invisibly"
    result = subprocess.run(
        [*wrapper, str(baseline)],
        capture_output=True,
        text=True,
        timeout=180.0,
        env=app_environ(tmp_path),
        check=False,
    )
    reported = next(
        (json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith("{")),
        None,
    )
    assert reported is not None, f"no report from the witness:\n{result.stdout}\n{result.stderr}"

    assert reported["vuid_01779_reports"] > 0, (
        "a bare QVulkanWindow no longer trips "
        "VUID-vkAcquireNextImageKHR-semaphore-01779 — Qt appears fixed. Delete the "
        "vkQueueWaitIdle workaround in Renderer::submit and this test with it."
    )
    # The mechanism, not just the symptom: fewer semaphore sets than swapchain
    # images is *why* Qt reuses a busy one. Asserting it means a future failure
    # says which of the two facts changed.
    assert reported["distinct_semaphores"] <= reported["concurrent_frames"]
    assert reported["concurrent_frames"] < reported["swapchain_images"]
