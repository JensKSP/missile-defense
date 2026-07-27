# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Vulkan correctness gates: the shaders statically, the renderer at runtime.

Two checks, deliberately split by what they cost:

``shaders``
    Compiles every GLSL source the app embeds and runs `spirv-val` against the
    *declared* Vulkan target. Needs no GPU, no display and no build — a second,
    so it belongs in `poe check`.

``runtime``
    Starts the debug game under `VK_LAYER_KHRONOS_validation` across the
    scenarios that render, with **synchronization validation** on, and fails on
    any message. Needs a display and minutes, so it belongs in `poe check-all`.

Both fail loudly rather than silently passing when their tools are missing. A
check that quietly does nothing is worse than no check, because it reports green
— that is exactly how the zero-validation-error gate here sat inert for weeks
while grepping a stream that `xvfb-run` had already merged away.

**On best practices.** `--best-practices` runs that layer too, but its findings
are *reported and do not fail*, and that is a considered line rather than a
loophole. What it currently says is that `VK_EXT_debug_utils` is a debugging
extension (it is, and it is enabled precisely because we are debugging — the
check complaining about the checker), and that some buffers are small enough to
be worth sub-allocating (true, and a performance opinion about a renderer that
allocates a handful of buffers for instanced quads). Neither is a correctness
defect. Errors and synchronization hazards gate; advice is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import _util

#: What `app/main.cpp` passes to `QVulkanInstance::setApiVersion`. Kept in step
#: with it deliberately: validating SPIR-V against a *newer* environment than the
#: instance requests is how you ship a shader the driver then refuses, which is
#: the shape of the `pCode-08740` bug this project already had once.
TARGET_ENV = "vulkan1.0"

#: The GLSL the app compiles into itself (`md_embed_shader` in app/CMakeLists).
#: Discovered from the directory rather than listed, so a new shader is covered
#: the moment it exists instead of the day someone remembers this file.
SHADER_DIR = _util.PROJECT_ROOT / "app" / "shaders"
SHADER_GLOBS = ("*.vert", "*.frag", "*.comp", "*.geom", "*.tesc", "*.tese")

#: Scenarios that put the renderer through a different path. `--frames` keeps
#: each bounded; the point is coverage of code paths, not of playing time.
SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("the menu", ()),
    ("gameplay", ("--play",)),
    ("watching the scripted agent", ("--watch-scripted", "low")),
    ("watching the expert scripted agent", ("--watch-scripted", "high")),
)

#: How a layer message is recognised. `VUID-` covers spec violations, `SYNC-` the
#: synchronization layer, `UNASSIGNED-` the checks that have no VUID assigned,
#: and `BestPractices-` the advisory layer — collected so it can be *printed*,
#: then separated out below so it cannot fail the build.
MESSAGE_MARKERS = ("VUID-", "SYNC-", "UNASSIGNED-", "BestPractices-")


def _debug_app() -> Path:
    """The debug binary, which is the one built with the validation layer on."""
    candidate = _util.PROJECT_ROOT / "build" / "debug" / "app" / f"md_app{_util.EXE}"
    if not candidate.exists():
        raise SystemExit(
            f"error: {candidate} is not built — `cmake --build --preset debug`.\n"
            "       The release build does not enable the validation layer, so it "
            "cannot answer this question."
        )
    return candidate


def check_shaders() -> list[str]:
    """Compile every shader and validate the SPIR-V. Returns failure messages."""
    glslang = _util.tool("glslangValidator")
    spirv_val = _util.tool_optional("spirv-val")
    if spirv_val is None:
        raise SystemExit(
            "error: spirv-val is not on PATH — install it (Debian/Ubuntu: "
            "`sudo apt install spirv-tools`; macOS: `brew install spirv-tools`).\n"
            "       Skipping it would report success for an unchecked shader."
        )

    sources = sorted(path for glob in SHADER_GLOBS for path in SHADER_DIR.glob(glob))
    if not sources:
        raise SystemExit(f"error: no shader sources under {SHADER_DIR}")

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="md-spirv-") as work:
        for source in sources:
            binary = Path(work) / f"{source.name}.spv"
            compiled = _util.run(
                [glslang, "-V", "--target-env", TARGET_ENV, "-o", str(binary), str(source)],
                check=False,
                capture=True,
                quiet=True,
            )
            if compiled.returncode != 0:
                failures.append(
                    f"{source.name}: did not compile\n{compiled.stdout}{compiled.stderr}"
                )
                continue
            validated = _util.run(
                [spirv_val, "--target-env", TARGET_ENV, str(binary)],
                check=False,
                capture=True,
                quiet=True,
            )
            if validated.returncode != 0:
                failures.append(
                    f"{source.name}: invalid SPIR-V for {TARGET_ENV}\n"
                    f"{validated.stdout}{validated.stderr}"
                )
            else:
                print(f"  ok  {source.relative_to(_util.PROJECT_ROOT)} ({TARGET_ENV})")
    return failures


def _display_wrapper() -> list[str]:
    """How to give the game a surface, or exit saying this machine cannot.

    A real X server, not Qt's ``offscreen`` platform: that plugin has no Vulkan
    support at all, so there would be no swapchain to validate.
    """
    if sys.platform != "linux" or os.environ.get("MD_VULKAN_CHECK_VISIBLE"):
        return []
    xvfb = shutil.which("xvfb-run")
    if xvfb is None:
        raise SystemExit(
            "error: this check needs a display and xvfb-run is not installed "
            "(`sudo apt install xvfb`).\n"
            "       Set MD_VULKAN_CHECK_VISIBLE=1 to run it on your own screen instead."
        )
    return [xvfb, "-a", "--server-args=-screen 0 1280x720x24"]


def _layer_settings(work: Path, *, best_practices: bool) -> Path:
    """A layer settings file turning on the checks the plain layer leaves off.

    ``duplicate_message_limit`` matters more than it looks: the layer prints ten
    of any repeated message and then goes quiet, so a count taken without this is
    a count of *ten*, not of what happened.
    """
    enables = ["VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"]
    if best_practices:
        enables.append("VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT")
    settings = work / "vk_layer_settings.txt"
    settings.write_text(
        "khronos_validation.enables = " + ",".join(enables) + "\n"
        "khronos_validation.duplicate_message_limit = 100000\n",
        encoding="utf-8",
    )
    return settings


def _witness_binary() -> Path | None:
    """The bare-QVulkanWindow witness, if this build produced one."""
    candidate = _util.PROJECT_ROOT / "build" / "debug" / "app" / f"md_vulkan_baseline{_util.EXE}"
    return candidate if candidate.exists() else None


def _require_a_live_layer(wrapper: list[str], env: dict[str, str]) -> None:
    """Refuse to run at all unless the validation layer is actually loaded.

    `QVulkanInstance::setLayers` on a layer the loader does not have is **silently
    ignored**. Every message this gate looks for then never appears, every
    scenario passes, and the result is a green tick that means "nothing was
    checked". CI ran exactly that way — no `vulkan-validationlayers` package —
    and both this gate and the e2e suite's `assert_clean` were inert in it.

    So the witness runs first as a canary: it reports whether the loader offers
    the layer, and a clean run is only trusted after that says yes.
    """
    witness = _witness_binary()
    if witness is None:
        raise SystemExit(
            "error: md_vulkan_baseline is not built, so this check cannot confirm "
            "the validation layer is live.\n"
            "       Configure with -DMD_VULKAN_VALIDATION=ON and rebuild. Without "
            "it a clean result would be indistinguishable from no result."
        )
    result = subprocess.run(
        [*wrapper, str(witness)],
        capture_output=True,
        text=True,
        timeout=300.0,
        env=env,
        check=False,
    )
    report = next(
        (json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith("{")),
        None,
    )
    if report is None:
        raise SystemExit(
            "error: the validation-layer canary produced no report:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if not report.get("validation_layer_available"):
        raise SystemExit(
            "error: VK_LAYER_KHRONOS_validation is not available to the loader, so "
            "nothing would be validated and this gate would pass without checking "
            "anything.\n"
            "       Install it (Debian/Ubuntu: `sudo apt install vulkan-validationlayers`; "
            "elsewhere: the Vulkan SDK)."
        )
    print(f"  layer live, canary sees {report['swapchain_images']} swapchain images")


def check_runtime(*, best_practices: bool) -> list[str]:
    """Run every rendering scenario under the layer. Returns failure messages."""
    app = _debug_app()
    wrapper = _display_wrapper()
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="md-vk-") as work_str:
        work = Path(work_str)
        env = dict(os.environ)
        env["VK_LAYER_SETTINGS_PATH"] = str(_layer_settings(work, best_practices=best_practices))
        if sys.platform == "linux":
            env["QT_QPA_PLATFORM"] = "xcb"
        # Keep the run out of the developer's real files, exactly as the e2e
        # harness does — a check should not be able to eat a high-score table.
        for name, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data")):
            (work / sub).mkdir(parents=True, exist_ok=True)
            env[name] = str(work / sub)
        env["MD_RUNS_DIR"] = str(work / "runs")
        # The leak checker fires on the X and GPU driver libraries, which is a
        # different question from the one being asked here.
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"

        _require_a_live_layer(wrapper, env)

        for what, args in SCENARIOS:
            command = [*wrapper, str(app), *args, "--frames", "240", "--silent", "--report"]
            print(f"  .. {what}")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=300.0, env=env, check=False
            )
            # Both streams: `xvfb-run` runs its command as `"$@" 2>&1`, so under
            # it the layer's messages arrive on stdout and stderr is empty. This
            # suite once grepped stderr alone and reported every run clean.
            output = result.stdout + result.stderr
            messages = [
                line for line in output.splitlines() if any(m in line for m in MESSAGE_MARKERS)
            ]
            gating = [line for line in messages if "BestPractices-" not in line]
            advice = [line for line in messages if "BestPractices-" in line]
            for line in advice:
                print(f"     advice: {line.strip()}")
            if gating:
                failures.append(
                    f"{what}: {len(gating)} validation message(s)\n" + "\n".join(gating)
                )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    # `__doc__` is None under `python -OO`, which strips docstrings.
    parser = argparse.ArgumentParser(
        prog="python -m tools.vulkan_check",
        description=(__doc__ or "Vulkan correctness gates.").splitlines()[0],
    )
    parser.add_argument(
        "what",
        nargs="?",
        default="all",
        choices=("shaders", "runtime", "all"),
        help="which gate to run (default: all)",
    )
    parser.add_argument(
        "--best-practices",
        action="store_true",
        help="also run the best-practices layer, reporting its advice without failing",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    if args.what in ("shaders", "all"):
        print("spirv-val:")
        failures += check_shaders()
    if args.what in ("runtime", "all"):
        print("validation layer:")
        failures += check_runtime(best_practices=args.best_practices)

    if failures:
        print("\nVulkan check failed:\n", file=sys.stderr)
        for failure in failures:
            print(failure + "\n", file=sys.stderr)
        return 1
    print("Vulkan check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
