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
    ABSENT,
    ALLOWED_INDEX_HOSTS,
    BROKEN,
    READY,
    Runtime,
    RuntimePlan,
    SystemInfo,
    UnsafeIndex,
    recommend,
)

# Bound here rather than reached through the module, because the autouse fixture
# below replaces `runtime._missing_binding` for every other test in the file.
# This name keeps pointing at the real one, which is what its own tests need.
from md.runtime import _missing_binding as probe_binding

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
    # numpy with it, on every backend. torch treats numpy as optional and this
    # project does not: `md.env` types its buffer contract with `numpy.typing`
    # and the trainer imports it on its first line, so a torch-only install is a
    # runtime that passes its check and then cannot run a single update.
    assert plan.packages == ("torch", "numpy")


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


def test_a_failed_check_names_the_import_that_failed() -> None:
    """It used to blame torch whatever went wrong.

    The health script imports the native binding *before* torch on purpose, so
    that an ABI mismatch is the first line of the traceback. Reporting "could
    not import torch" regardless threw that away and sent people to look at a
    torch install that was perfectly fine — which is exactly what happens in a
    checkout whose binding has not been built yet.
    """
    missing_binding = [
        "Successfully installed torch-2.13.0+cu130",
        "Traceback (most recent call last):",
        '  File "<string>", line 2, in <module>',
        "    import md._md_native as native",
        "ModuleNotFoundError: No module named 'md._md_native'",
    ]
    advice = runtime._why_unhealthy(missing_binding)
    assert "poe bindings" in advice
    assert "torch" not in advice.split("`poe bindings`")[0]

    broken_torch = [
        "Traceback (most recent call last):",
        '  File "<string>", line 3, in <module>',
        "    import torch",
        "ImportError: libcudart.so.13: cannot open shared object file",
    ]
    assert "torch" in runtime._why_unhealthy(broken_torch)

    # Something else entirely stays honest rather than picking a module.
    assert runtime._why_unhealthy(["killed by the OOM killer"]) == runtime.UNHEALTHY

    # And the one that shipped: torch imports, the run does not.
    missing_numpy = [
        "Traceback (most recent call last):",
        '  File "<string>", line 4, in <module>',
        "    import numpy",
        "ModuleNotFoundError: No module named 'numpy'",
    ]
    assert "numpy" in runtime._why_unhealthy(missing_numpy)


def test_the_runtime_installs_everything_its_health_check_demands() -> None:
    """The two lists that must not drift, held together here.

    They did drift, and it was invisible from both sides: the check imported
    torch and the binding, the install fetched torch, and numpy — which the
    trainer imports on its first line and torch declares optional — was in
    neither. The result was a runtime that installed cleanly, reported itself
    healthy, and killed every run it was asked to start, in a subprocess whose
    output the console had already scrolled past.

    A check demanding more than the install provides could never pass; an
    install providing less than a run needs is the bug above. So: the same set,
    minus the binding, which is compiled from this checkout and is on no index.
    """
    checked = {module for module, _ in runtime.IMPORT_ADVICE}
    assert "numpy" in checked
    for module in checked:
        assert f"import {module}\n" in runtime.HEALTH_SCRIPT, f"{module} is advised but unchecked"
    assert checked - {"md._md_native"} == set(runtime.PACKAGES)


@pytest.fixture(autouse=True)
def _binding_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here talks to a fake runner, so the real binding is irrelevant.

    `Runtime.install` refuses before downloading when `md._md_native` cannot be
    imported, which is right in production and wrong for a suite that must give
    the same answer on a machine that has never run `poe bindings` — the quality
    gate is exactly such a machine. The one test that cares patches it the other
    way.
    """
    monkeypatch.setattr(runtime, "_missing_binding", lambda _python, _runner: None)


def _ready_store(tmp_path: Path) -> tuple[Runtime, FakeRunner]:
    """A store with one installed, healthy runtime — the state before a doubt.

    Handed back as a *fresh* store over the same directory, because that is what
    a console meets when it opens: the install that filled it happened in another
    process and left nothing behind in memory. A store that has just installed is
    a different situation, and has its own test — it remembers the check it ran.
    """
    root = tmp_path / "store"
    plan = RuntimePlan(
        backend="cpu",
        python=Path("/usr/bin/python3"),
        target=root / "cpu-py3.13-1",
        packages=("torch", "numpy"),
        index_url="https://pypi.org/simple",
    )
    assert Runtime(root, runner=FakeRunner(), platform="linux").install(plan).state == READY
    runner = FakeRunner()
    return Runtime(root, runner=runner, platform="linux"), runner


def _binding_probes(
    *, imports: bool, present: bool
) -> Callable[[list[str], Callable[[str], None]], int]:
    """A runner that answers the two binding probes and refuses anything else."""

    def run(command: list[str], _emit: Callable[[str], None]) -> int:
        script = command[-1]
        if script == runtime.BINDING_PROBE:
            return 0 if imports else 1
        if script == runtime.BINDING_PRESENT_PROBE:
            return 0 if present else 1
        raise AssertionError(f"not a binding probe: {command}")

    return run


def test_a_binding_that_imports_is_no_obstacle() -> None:
    assert (
        probe_binding(Path("/opt/native/python"), _binding_probes(imports=True, present=True))
        is None
    )


def test_a_binding_built_for_another_interpreter_is_caught_before_the_download() -> None:
    """The 1.9 GB one, and the reason the guard imports instead of looking.

    A `.pyd` compiled by MSYS2 clang sits on the path of a python.org CPython
    quite happily and then fails to load, because it wants MinGW's runtime DLLs.
    The old guard asked `find_spec`, which says yes to exactly that file — so it
    passed, pip fetched CUDA torch in full, and the health check failed on the
    import the guard had declined to attempt. Measured on 2026-07-28.
    """
    python = Path("/opt/native/python")
    detail = probe_binding(python, _binding_probes(imports=False, present=True))
    assert detail is not None
    assert detail != runtime.NO_BINDING  # `poe bindings` has already been run
    # Spelled as this platform spells it — the message quotes the path back, and
    # asserting the tidier form only proves the test and the code disagree.
    assert str(python) in detail  # ... just not against this interpreter


def test_a_binding_that_was_never_built_says_so() -> None:
    detail = probe_binding(
        Path("/opt/native/python"), _binding_probes(imports=False, present=False)
    )
    assert detail == runtime.NO_BINDING


def test_the_binding_is_probed_in_the_interpreter_the_runtime_will_be_built_from() -> None:
    # Not in the console's own. They are routinely different on Windows — the
    # console can be running under MSYS2 while the runtime is made from a
    # python.org CPython — and only one of them has to load the extension.
    asked: list[list[str]] = []

    def run(command: list[str], _emit: Callable[[str], None]) -> int:
        asked.append(command)
        return 0

    python = Path("/opt/native/python")
    assert probe_binding(python, run) is None
    assert asked[0][0] == str(python)


def test_an_install_refuses_before_downloading_when_the_binding_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five gigabytes should not be spent to learn something knowable up front.

    The health check imports the native binding, which is compiled from this
    checkout and can never be supplied by pip. Running it only at the end meant
    a checkout whose `.so` was missing downloaded CUDA torch in full, failed,
    and then had the directory — torch included — deleted. The fact was
    available before the first byte.
    """
    monkeypatch.setattr(runtime, "_missing_binding", lambda _python, _runner: runtime.NO_BINDING)
    runner = FakeRunner()
    store = Runtime(tmp_path / "store", runner=runner, platform="linux")
    plan = RuntimePlan(
        backend="cpu",
        python=Path("/usr/bin/python3"),
        target=tmp_path / "store" / "cpu-py3.13-1",
        packages=("torch",),
        index_url="https://pypi.org/simple",
    )
    status = store.install(plan)
    assert runner.commands == [], "the install downloaded something already known to fail"
    assert "poe bindings" in status.detail


def test_verify_asks_the_runtime_to_prove_it_rather_than_trusting_the_manifest(
    tmp_path: Path,
) -> None:
    """`status()` believes a file; `verify()` asks a question.

    Everything `status()` checks stays true of a runtime that no longer works —
    a torch deleted to reclaim disk, a driver downgraded under it, a binding
    rebuilt for another Python. The console would keep offering Start, the
    button would appear to do nothing, and the failure would surface somewhere
    unrelated.
    """
    store, _ = _ready_store(tmp_path)
    assert store.status().state == READY  # the manifest still says so

    store._runner = FakeRunner(fail_at="import torch")  # noqa: SLF001 — the seam under test
    checked = store.verify()
    assert checked.state == BROKEN
    assert checked.python is None, "a runtime that cannot prove itself must not be trainable"


def test_a_verified_runtime_is_not_re_verified_on_every_ask(tmp_path: Path) -> None:
    """It costs a subprocess and an `import torch`; the console polls once a second."""
    store, _ = _ready_store(tmp_path)
    calls: list[list[str]] = []
    store._runner = FakeRunner(on_command=calls.append)  # noqa: SLF001

    assert store.verify().state == READY
    first = len(calls)
    assert first > 0
    assert store.verify().state == READY
    assert len(calls) == first, "verification repeated with nothing having changed"
    assert store.verify(force=True).state == READY
    assert len(calls) > first, "force did not re-check"


def test_an_install_counts_as_the_verification_it_has_just_run(tmp_path: Path) -> None:
    """The health check at the end of an install is a verification, so say so.

    The dialog asks `verify()` to decide what it offers, and it asks immediately
    after an install finishes. Without this the answer is a second health check
    on the UI thread — and the first `torch.cuda.is_available()` of a fresh CUDA
    install takes minutes while the driver builds its caches, which after a
    five-minute download reads as a window that has hung at the finish line.
    """
    root = tmp_path / "store"
    plan = RuntimePlan(
        backend="cpu",
        python=Path("/usr/bin/python3"),
        target=root / "cpu-py3.13-1",
        packages=("torch", "numpy"),
        index_url="https://pypi.org/simple",
    )
    store = Runtime(root, runner=FakeRunner(), platform="linux")
    assert store.install(plan).state == READY

    calls: list[list[str]] = []
    store._runner = FakeRunner(on_command=calls.append)  # noqa: SLF001 — the seam
    assert store.verify().state == READY
    assert calls == [], "the runtime was asked to prove what it had just proved"


def test_removing_a_runtime_forgets_that_it_was_verified(tmp_path: Path) -> None:
    """Otherwise the next install inherits a verdict about a deleted directory."""
    store, _ = _ready_store(tmp_path)
    assert store.verify().state == READY
    store.remove()
    assert store.verify().state == ABSENT


def test_a_failed_verification_is_not_remembered(tmp_path: Path) -> None:
    """The thing it was missing may since have been built.

    Caching a failure would mean a console that has to be restarted after
    `poe bindings` — exactly the kind of stale answer this method exists to stop
    giving.
    """
    store, _ = _ready_store(tmp_path)
    calls: list[list[str]] = []
    store._runner = FakeRunner(fail_at="import torch", on_command=calls.append)  # noqa: SLF001

    assert store.verify().state == BROKEN
    before = len(calls)
    assert store.verify().state == BROKEN
    assert len(calls) > before, "a failure was cached"
