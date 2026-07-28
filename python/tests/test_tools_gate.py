# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The parallel gate runs the same stages the serial one lists.

There are two records of what the gate is: :data:`tools.gate.STAGES`, which the
driver walks, and the `check-serial` task in pyproject.toml, which is what you
fall back to when concurrent output is too interleaved to read. Two lists of the
same thing drift, and the drift is silent in the worst direction — a stage
dropped from `STAGES` makes `poe check` faster *and* green while no longer
checking anything.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import gate  # noqa: E402 — after the path insert


def _tasks() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    tasks = data["tool"]["poe"]["tasks"]
    assert isinstance(tasks, dict)
    return tasks


def test_the_parallel_gate_and_the_serial_one_run_the_same_stages() -> None:
    serial = _tasks()["check-serial"]
    assert isinstance(serial, dict)
    listed = set(serial["sequence"])
    driven = {stage.task for stage in gate.STAGES}
    assert driven == listed, (
        "tools/gate.py and the check-serial task disagree about the gate; "
        f"only in the driver: {sorted(driven - listed)}, "
        f"only in the sequence: {sorted(listed - driven)}"
    )


def test_every_stage_names_a_task_that_exists() -> None:
    tasks = _tasks()
    missing = [stage.task for stage in gate.STAGES if stage.task not in tasks]
    assert not missing, f"no such poe task: {missing}"


def test_stages_sharing_a_build_tree_declare_it() -> None:
    # Two `cmake --build` runs over one directory is a race that produces a
    # corrupt tree rather than a diagnostic, so the lock is the whole safety
    # argument for running these at the same time. `tidy` and `test` both build
    # Debug; if either loses its resource the gate becomes flaky in a way that
    # looks like a compiler bug.
    by_task = {stage.task: stage for stage in gate.STAGES}
    assert by_task["tidy"].resource == by_task["test"].resource == "debug"
    assert by_task["test-release"].resource == "release"
    assert by_task["coverage"].resource == "coverage"


def test_the_slowest_stage_starts_first() -> None:
    # `tidy` is the critical path — around twenty seconds where nothing else
    # reaches ten — so starting it late adds its whole duration to the run.
    heaviest = max(gate.STAGES, key=lambda stage: stage.weight)
    assert heaviest.task == "tidy"


@pytest.mark.parametrize("stage", gate.STAGES, ids=lambda s: s.task)
def test_no_stage_writes_over_another_stages_log(stage: gate.Stage) -> None:
    same_name = [other for other in gate.STAGES if other.task == stage.task]
    assert len(same_name) == 1, f"{stage.task} appears twice, so one log overwrites the other"
