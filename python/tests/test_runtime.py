# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for the managed training runtime — without installing one.

``md.runtime`` splits planning from effects for exactly this reason: what to
install is a pure function of the machine and can be asserted outright, and the
install itself takes its subprocess runner as an argument, so every state the
dialog has to render is reachable here without a network, a venv, or torch.

The states under test are the ones the UI must distinguish: nothing installed,
a recommendation, a half-finished directory, a healthy runtime, a broken one,
a cancelled install, and a removed one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from md import runtime
from md.runtime import (
    ALLOWED_INDEX_HOSTS,
    Runtime,
    RuntimePlan,
    SystemInfo,
    UnsafeIndex,
    recommend,
)

PY = (3, 13)


def _system(platform: str = "linux", *, version: tuple[int, int] = PY) -> SystemInfo:
    return SystemInfo(platform=platform, python=Path("/usr/bin/python3"), python_version=version)


class FakeProbe:
    """A vendor detector that answers without a driver being installed."""

    def __init__(self, backend: str, present: bool) -> None:
        self.backend = backend
        self._present = present

    def present(self) -> bool:
        return self._present


class FakeRunner:
    """Stands in for running venv, pip and the health probe.

    It *acts* on the filesystem the way the real commands would — a venv step
    creates the interpreter, so the code under test sees the same directory
    layout it would in production — and it records every command, which is what
    the index-url assertions read.
    """

    def __init__(
        self,
        *,
        fail_at: str | None = None,
        health: dict[str, object] | None = None,
        on_command: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.commands: list[list[str]] = []
        self._fail_at = fail_at
        # What HEALTH_SCRIPT actually prints, `native` included — the binding
        # import is the half of the check that catches an ABI mismatch.
        self._health = (
            health
            if health is not None
            else {"python": "3.13.0", "native": True, "torch": "2.13.0", "device": "cuda"}
        )
        self._on_command = on_command

    def __call__(self, command: list[str], emit: Callable[[str], None]) -> int:
        self.commands.append(list(command))
        if self._on_command is not None:
            self._on_command(command)
        if self._fail_at is not None and self._fail_at in " ".join(command):
            emit(f"pretend failure in {self._fail_at}")
            return 1
        if "venv" in command:
            self._make_venv(Path(command[-1]))
            emit("created virtual environment")
        elif "pip" in command:
            emit("Successfully installed torch")
        elif runtime.HEALTH_SCRIPT in command:
            emit(json.dumps(self._health))
        return 0

    @staticmethod
    def _make_venv(target: Path) -> None:
        interpreter = runtime.venv_python(target, platform="linux")
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.write_text("#!/bin/sh\n")
        interpreter.chmod(0o755)


def _install(root: Path, runner: FakeRunner, **kwargs: object) -> runtime.RuntimeStatus:
    store = Runtime(root, runner=runner, platform="linux")
    plan = recommend(_system(), [FakeProbe("cuda", True)], root=root)
    return store.install(plan, **kwargs)  # type: ignore[arg-type]


# ---- planning ----------------------------------------------------------------


def test_an_nvidia_machine_is_offered_the_cuda_build(tmp_path: Path) -> None:
    plan = recommend(_system(), [FakeProbe("cuda", True)], root=tmp_path)
    assert plan.backend == "cuda"
    assert "download.pytorch.org" in plan.index_url
    assert plan.packages == ("torch",)


def test_a_machine_with_no_accelerator_is_offered_the_cpu_build(tmp_path: Path) -> None:
    plan = recommend(_system(), [FakeProbe("cuda", False)], root=tmp_path)
    assert plan.backend == "cpu"


def test_rocm_is_not_offered_on_windows_where_it_does_not_exist(tmp_path: Path) -> None:
    # amdsmi/pyrsmi are ROCm, so effectively Linux (docs/ROADMAP.md, M8 phase 4).
    # A probe that answers yes must still not produce a plan that cannot install.
    plan = recommend(_system("win32"), [FakeProbe("rocm", True)], root=tmp_path)
    assert plan.backend == "cpu"


def test_rocm_is_offered_on_linux(tmp_path: Path) -> None:
    plan = recommend(_system("linux"), [FakeProbe("rocm", True)], root=tmp_path)
    assert plan.backend == "rocm"


def test_the_first_probe_that_answers_wins(tmp_path: Path) -> None:
    probes: Sequence[FakeProbe] = [FakeProbe("cuda", True), FakeProbe("rocm", True)]
    assert recommend(_system(), probes, root=tmp_path).backend == "cuda"


def test_the_target_directory_names_what_is_in_it(tmp_path: Path) -> None:
    plan = recommend(_system(), [FakeProbe("cuda", True)], root=tmp_path)
    assert plan.target.name.startswith("cuda-py3.13")
    assert plan.target.parent == tmp_path


def test_a_second_plan_does_not_reuse_a_directory_that_exists(tmp_path: Path) -> None:
    first = recommend(_system(), [FakeProbe("cuda", True)], root=tmp_path)
    first.target.mkdir(parents=True)
    second = recommend(_system(), [FakeProbe("cuda", True)], root=tmp_path)
    assert second.target != first.target


@pytest.mark.parametrize(
    "index",
    [
        "https://evil.example.com/simple",
        "http://pypi.org/simple",  # https or nothing: this fetches executable code
        "file:///tmp/wheels",
    ],
)
def test_only_allow_listed_indexes_can_be_installed_from(tmp_path: Path, index: str) -> None:
    plan = RuntimePlan(
        backend="cpu",
        python=Path("/usr/bin/python3"),
        target=tmp_path / "cpu-py3.13-1",
        packages=("torch",),
        index_url=index,
    )
    with pytest.raises(UnsafeIndex):
        Runtime(tmp_path, runner=FakeRunner(), platform="linux").install(plan)


def test_the_allow_list_is_the_two_hosts_the_docs_name() -> None:
    assert set(ALLOWED_INDEX_HOSTS) == {"pypi.org", "download.pytorch.org"}


# ---- states ------------------------------------------------------------------


def test_a_machine_with_nothing_installed_reports_absent(tmp_path: Path) -> None:
    status = Runtime(tmp_path, runner=FakeRunner(), platform="linux").status()
    assert status.state == "absent"
    assert not status.ready
    assert status.python is None


def test_a_successful_install_is_ready_and_names_its_interpreter(tmp_path: Path) -> None:
    runner = FakeRunner()
    status = _install(tmp_path, runner)
    assert status.state == "ready"
    assert status.ready
    assert status.backend == "cuda"
    assert status.torch_version == "2.13.0"
    assert status.device == "cuda"
    assert status.python is not None and status.python.exists()


def test_readiness_survives_a_new_store_object(tmp_path: Path) -> None:
    # The state lives on disk, not in the object: the console is restarted far
    # more often than a runtime is installed.
    _install(tmp_path, FakeRunner())
    assert Runtime(tmp_path, runner=FakeRunner(), platform="linux").status().ready


def test_the_installed_runtime_is_the_one_the_plan_named(tmp_path: Path) -> None:
    runner = FakeRunner()
    store = Runtime(tmp_path, runner=runner, platform="linux")
    plan = recommend(_system(), [FakeProbe("cuda", True)], root=tmp_path)
    status = store.install(plan)
    assert status.python == runtime.venv_python(plan.target, platform="linux")


def test_the_index_url_reaches_pip(tmp_path: Path) -> None:
    runner = FakeRunner()
    _install(tmp_path, runner)
    pip = next(c for c in runner.commands if "pip" in c)
    assert "--index-url" in pip
    assert pip[pip.index("--index-url") + 1] == runtime.CUDA.index_url
    assert "torch" in pip


def test_a_failed_install_leaves_nothing_behind_and_says_why(tmp_path: Path) -> None:
    status = _install(tmp_path, FakeRunner(fail_at="pip"))
    assert status.state == "absent"
    assert "pip" in status.detail or "install" in status.detail
    assert list(tmp_path.glob("cuda-*")) == []


def test_a_runtime_that_cannot_import_torch_is_broken_not_ready(tmp_path: Path) -> None:
    # The health check is the whole point of installing transactionally: pip can
    # succeed and the result still not run — a wheel with no kernel for this card
    # is the documented case (docs/NVIDIA.md).
    status = _install(tmp_path, FakeRunner(fail_at=runtime.HEALTH_SCRIPT))
    assert status.state == "absent"
    assert list(tmp_path.glob("cuda-*")) == []


def test_a_runtime_that_stops_working_later_reports_broken_and_repairable(
    tmp_path: Path,
) -> None:
    _install(tmp_path, FakeRunner())
    # The interpreter is gone — a removed venv, a deleted directory, an OS upgrade.
    installed = next(tmp_path.glob("cuda-*"))
    runtime.venv_python(installed, platform="linux").unlink()

    status = Runtime(tmp_path, runner=FakeRunner(), platform="linux").status()
    assert status.state == "broken"
    assert status.repairable
    assert status.removable
    assert not status.ready


def test_a_tampered_manifest_is_broken_rather_than_trusted(tmp_path: Path) -> None:
    _install(tmp_path, FakeRunner())
    installed = next(tmp_path.glob("cuda-*"))
    manifest = installed / runtime.MANIFEST_NAME
    payload = json.loads(manifest.read_text())
    payload["backend"] = "rocm"  # edited without recomputing the checksum
    manifest.write_text(json.dumps(payload))

    status = Runtime(tmp_path, runner=FakeRunner(), platform="linux").status()
    assert status.state == "broken"


def test_a_half_finished_directory_is_reported_and_cleaned_by_repair(tmp_path: Path) -> None:
    orphan = tmp_path / "cuda-py3.13-1"
    (orphan / "bin").mkdir(parents=True)

    store = Runtime(tmp_path, runner=FakeRunner(), platform="linux")
    assert store.status().state == "incomplete"
    store.repair()
    assert not orphan.exists()
    assert store.status().state == "absent"


# ---- cancellation and removal ------------------------------------------------


def test_cancelling_removes_only_the_incomplete_directory(tmp_path: Path) -> None:
    cancelled = {"now": False}
    runner = FakeRunner(on_command=lambda command: cancelled.__setitem__("now", "pip" in command))
    store = Runtime(tmp_path, runner=runner, platform="linux")
    plan = recommend(_system(), [FakeProbe("cuda", True)], root=tmp_path)

    status = store.install(plan, cancel=lambda: cancelled["now"])

    assert status.state == "absent"
    assert not plan.target.exists()
    assert "cancel" in status.detail.lower()


def test_cancelling_a_second_install_leaves_the_working_one_alone(tmp_path: Path) -> None:
    _install(tmp_path, FakeRunner())
    working = next(tmp_path.glob("cuda-*"))

    store = Runtime(tmp_path, runner=FakeRunner(), platform="linux")
    plan = recommend(_system(), [FakeProbe("rocm", True)], root=tmp_path)
    store.install(plan, cancel=lambda: True)

    assert not plan.target.exists()
    status = store.status()
    assert status.ready
    assert status.python == runtime.venv_python(working, platform="linux")


def test_removing_takes_the_whole_runtime_and_returns_to_absent(tmp_path: Path) -> None:
    _install(tmp_path, FakeRunner())
    store = Runtime(tmp_path, runner=FakeRunner(), platform="linux")
    assert store.status().removable

    store.remove()

    assert store.status().state == "absent"
    assert list(tmp_path.glob("cuda-*")) == []


def test_removing_when_there_is_nothing_installed_is_not_an_error(tmp_path: Path) -> None:
    store = Runtime(tmp_path, runner=FakeRunner(), platform="linux")
    store.remove()
    assert store.status().state == "absent"


def test_repairing_a_broken_runtime_reinstalls_it(tmp_path: Path) -> None:
    _install(tmp_path, FakeRunner())
    runtime.venv_python(next(tmp_path.glob("cuda-*")), platform="linux").unlink()

    store = Runtime(tmp_path, runner=FakeRunner(), platform="linux")
    assert store.status().state == "broken"
    store.repair()
    assert store.status().state == "absent"  # cleaned; the dialog then offers Install


# ---- progress ----------------------------------------------------------------


def test_the_install_reports_what_it_is_doing(tmp_path: Path) -> None:
    lines: list[str] = []
    _install(tmp_path, FakeRunner(), on_output=lines.append)
    assert any("Successfully installed torch" in line for line in lines)
    assert any("virtual environment" in line for line in lines)


def test_progress_is_reported_for_a_failure_too(tmp_path: Path) -> None:
    lines: list[str] = []
    _install(tmp_path, FakeRunner(fail_at="pip"), on_output=lines.append)
    assert any("pretend failure" in line for line in lines)
