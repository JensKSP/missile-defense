# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The console, built for real against a real run directory.

Offscreen throughout, so nothing appears on anyone's screen: Qt Widgets needs no
graphics device — only the game does, and only because it is Vulkan.

What these add over the existing `test_ui_*` tests is the window itself. Those
cover the Qt-free halves (`sources`, `runner`, `params`) thoroughly and cannot
touch the part where a `QMainWindow` is constructed, wired to a directory, and
ticked. Every panel being fed from files a *different process* wrote is the claim
here, and the failure mode it guards against — a console that starts and shows
empty panels next to a run full of data — is invisible to both halves separately.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest
from md.benchmark import (
    CANONICAL_BASELINE_MEAN_SCORE,
    CANONICAL_FRAME_SKIP,
    CANONICAL_INFERENCE_DEVICE,
    CANONICAL_MAX_TICKS,
    CANONICAL_SEED_OFFSET,
    CANONICAL_SPLIT,
    SEEDS_PER_SPLIT,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
)

from .harness import needs_native, needs_qt, needs_torch

pytestmark = [pytest.mark.e2e, needs_qt]


@pytest.fixture
def console(qt_app: object, trained_run: Path):  # noqa: ANN201 — PySide6 is optional
    """A real console window attached to a finished run."""
    from md.ui.app import Console  # noqa: PLC0415 — optional dependency

    window = Console(trained_run)
    window.resize(1280, 800)
    yield window
    window.close()


@needs_torch
@needs_native
def test_the_console_opens_on_a_run_and_shows_its_curves(console) -> None:  # noqa: ANN001
    console._tick()
    # One update counted per row of metrics.csv. An empty window next to a run
    # full of data is the failure this whole file exists to catch, and it is
    # invisible to the Qt-free tests because they never build the window.
    assert console._updates > 0


@needs_torch
@needs_native
def test_the_console_lists_the_episodes_it_could_play(console) -> None:  # noqa: ANN001
    console._tick()
    assert console._list.count() > 0


@needs_torch
@needs_native
def test_the_console_describes_the_model_that_was_trained(console) -> None:  # noqa: ANN001
    console._tick()
    # ModelPanel keeps what it last painted; a run with a model.json must have
    # made it paint something other than its empty state.
    assert console._model._shown is not None
    assert console._model._headline.text()


@needs_torch
@needs_native
def test_a_run_that_just_wrote_reads_as_live(console) -> None:  # noqa: ANN001
    # Liveness is inferred from the files alone — how long ago metrics.csv was
    # written — which is exactly what lets the console tell the truth about a run
    # it never started. A run that wrote a moment ago is *correctly* live even
    # though its process has exited: the console cannot know that, and guessing
    # "stopped" would take Pause and Stop away from a run that is still going.
    console._tick()
    assert console._stop.isEnabled()
    assert console._primary.text() == "Pause"


@needs_torch
@needs_native
def test_a_run_that_has_gone_quiet_reads_as_idle(
    qt_app: object, trained_run: Path, tmp_path: Path
) -> None:
    # The other side of the same rule, and the reason it is a timeout rather than
    # a flag: nothing writes "this run is over" to the directory, so silence is
    # the only signal there is.
    from md.ui.app import LIVE_AFTER_S, Console  # noqa: PLC0415

    quiet = tmp_path / "quiet"
    quiet.mkdir()
    stale = time.time() - (LIVE_AFTER_S * 2)
    for name in ("metrics.csv", "evals.csv"):
        shutil.copy(trained_run / name, quiet / name)
        os.utime(quiet / name, (stale, stale))

    window = Console(quiet)
    try:
        window._tick()
        assert window._updates > 0, "the rows were still read"
        assert not window._stop.isEnabled()
        assert window._primary.text() in ("Start", "Set up training…")
    finally:
        window.close()


def test_the_console_on_an_empty_directory_explains_itself(qt_app: object, tmp_path: Path) -> None:
    # Empty states are part of the design (docs/ROADMAP.md, M8): a fresh
    # directory must say what is missing and what would fill it, not show a
    # blank panel or a zeroed meter.
    from md.ui.app import Console  # noqa: PLC0415

    window = Console(tmp_path)
    try:
        window._tick()
        message = window.statusBar().currentMessage()
        assert "metrics.csv" in message
        assert window._updates == 0
    finally:
        window.close()


def test_the_console_says_which_build_it_is_and_what_it_runs_on(
    qt_app: object, tmp_path: Path
) -> None:
    # Two claims a released application has to make from inside itself. The
    # version, because a bug report without one is unusable and the console is
    # usually installed from a package rather than a checkout. And the notice,
    # because this MIT program runs on LGPL-3.0 libraries and the user should
    # not have to find a file in a repository to be told so.
    import md  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415

    window = Console(tmp_path)
    try:
        assert window._about.text() == f"v{md.__version__}"
        shown = window._about_text()
        assert md.__version__ in shown
        assert "Jens Köhler" in shown
        assert "MIT" in shown
        assert "PySide6" in shown
        assert "LGPL-3.0-only" in shown
        assert "THIRD_PARTY_LICENSES.md" in shown
    finally:
        window.close()


def test_the_eval_slider_drives_a_run_this_console_never_started(
    qt_app: object, tmp_path: Path
) -> None:
    # The console is a convenience over the files, never the only way in: it
    # reads what a trainer published and writes back the same file a terminal
    # would `echo` into. Nothing about this widget knows which process is
    # training, which is exactly why it works on a run started elsewhere.
    from md.control import Control  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415

    control = Control(tmp_path)
    control.publish_tuning({"eval_every": 50})  # what a starting run does
    window = Console(tmp_path)
    try:
        window._tick()
        assert window._eval_every.isEnabled()
        assert window._eval_shown() == 50

        window._eval_every.setValue(window._eval_stops.index(10))
        assert control.tuned("eval_every", 999) == 10
        assert window._eval_readout.text() == "10 upd"

        # And the other direction: someone edits the file, the handle follows
        # rather than insisting on the last thing it said.
        control.tune("eval_every", 25)
        window._tick()
        assert window._eval_shown() == 25
    finally:
        window.close()


def test_the_eval_slider_shows_an_interval_that_is_not_one_of_its_stops(
    qt_app: object, tmp_path: Path
) -> None:
    # A run started with --eval-every 30 is on 30, and a handle snapped to the
    # nearest stop would be describing it wrongly. The scale gains a stop.
    from md.control import Control  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415

    Control(tmp_path).publish_tuning({"eval_every": 30})
    window = Console(tmp_path)
    try:
        window._tick()
        assert window._eval_shown() == 30
        assert window._eval_readout.text() == "30 upd"
    finally:
        window.close()


def test_the_eval_slider_greys_out_when_no_run_publishes_one(
    qt_app: object, tmp_path: Path
) -> None:
    # A directory with no run in it, or a run started before this existed. A
    # control that happily wrote a file nothing reads would be worse than a dead
    # one, and worse still if it left that file behind for the next run.
    from md.ui.app import EVAL_EVERY_UNPUBLISHED, Console  # noqa: PLC0415

    window = Console(tmp_path)
    try:
        window._tick()
        assert not window._eval_every.isEnabled()
        assert window._eval_every.toolTip() == EVAL_EVERY_UNPUBLISHED
        assert not (tmp_path / "TUNING.json").exists(), "the console invented a tuning file"
    finally:
        window.close()


def test_a_protocol_change_starts_a_new_score_curve_and_controls_the_baseline(
    qt_app: object, tmp_path: Path
) -> None:
    from md.ui.app import Console  # noqa: PLC0415

    path = tmp_path / "evals.csv"
    header = (
        "update,mean_score,seed_split,seed_offset,seed_count,frame_skip,"
        "max_ticks,inference_device\n"
    )
    path.write_text(
        header + f"50,120000,{VALIDATION_SPLIT},{VALIDATION_SEED_OFFSET},"
        f"{SEEDS_PER_SPLIT},4,120000,cpu\n",
        encoding="utf-8",
    )
    window = Console(tmp_path)
    try:
        window._tick()
        assert window._score._count == 1
        assert window._score._baseline is None
        assert "validation" in window._tile_score._note.text()
        assert "baseline" not in window._tile_score._note.text()

        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"100,{CANONICAL_BASELINE_MEAN_SCORE + 100},{CANONICAL_SPLIT},"
                f"{CANONICAL_SEED_OFFSET},{SEEDS_PER_SPLIT},{CANONICAL_FRAME_SKIP},"
                f"{CANONICAL_MAX_TICKS},{CANONICAL_INFERENCE_DEVICE}\n"
            )
        window._tick()

        # No line or peak spans validation -> canonical. Only the newest,
        # internally comparable segment remains on screen.
        assert window._score._count == 1
        assert window._peak_score.update == 100
        assert window._score._baseline == CANONICAL_BASELINE_MEAN_SCORE
        assert "ahead of" in window._tile_score._note.text()
    finally:
        window.close()


def test_comparison_scores_wait_for_the_primary_protocol(qt_app: object, tmp_path: Path) -> None:
    from md.ui.app import Console  # noqa: PLC0415

    primary = tmp_path / "primary"
    comparison = tmp_path / "comparison"
    primary.mkdir()
    comparison.mkdir()
    header = (
        "update,mean_score,seed_split,seed_offset,seed_count,frame_skip,"
        "max_ticks,inference_device\n"
    )
    (comparison / "evals.csv").write_text(
        header + f"50,90000,{VALIDATION_SPLIT},{VALIDATION_SEED_OFFSET},"
        f"{SEEDS_PER_SPLIT},4,120000,cpu\n",
        encoding="utf-8",
    )

    window = Console(primary)
    try:
        window._compare_with(comparison)
        window._tick()
        assert window._score._compare_count == 0

        (primary / "evals.csv").write_text(
            header + f"50,91000,{VALIDATION_SPLIT},{VALIDATION_SEED_OFFSET},"
            f"{SEEDS_PER_SPLIT},4,120000,cpu\n",
            encoding="utf-8",
        )
        window._tick()
        assert window._score._compare_count == 1
    finally:
        window.close()


def test_the_setup_dialog_offers_only_builds_this_platform_has(qt_app: object) -> None:
    # The console's answer to "I have no torch". Built for real; nothing is
    # installed, because the install itself is covered by test_runtime.py against
    # a fake runner and does not need several gigabytes to be exercised again.
    from md import runtime  # noqa: PLC0415
    from md.ui.runtime_dialog import RuntimeDialog  # noqa: PLC0415

    dialog = RuntimeDialog(runtime.Runtime(Path("/nonexistent")), probes=[])
    try:
        offered = {dialog._backend.itemData(i) for i in range(dialog._backend.count())}
        assert "cpu" in offered, "the fallback build must always be installable"
        assert dialog._primary.text() == "Install"
    finally:
        dialog.close()


def test_the_run_a_console_would_start_is_a_command_you_could_type(
    qt_app: object, tmp_path: Path
) -> None:
    # The parameter dialog teaches the CLI rather than replacing it, so the
    # command it shows has to be the command it would run.
    from md.ui.forms import ParameterDialog  # noqa: PLC0415
    from md.ui.params import read_params  # noqa: PLC0415
    from md.ui.runner import PACKAGE_PATH  # noqa: PLC0415

    dialog = ParameterDialog(
        read_params(PACKAGE_PATH / "md"), python="/usr/bin/python3", out_dir=tmp_path
    )
    try:
        command = dialog.command()
        assert command[0] == "/usr/bin/python3"
        assert "md.train" in command
        assert str(tmp_path) in command
    finally:
        dialog.close()


# ---- a stop that nobody is left to obey --------------------------------------


def test_a_leftover_stop_does_not_wedge_the_console(qt_app: object, tmp_path: Path) -> None:  # noqa: ARG001
    """A STOP outlives the run that was asked to stop, and used to be forever.

    The trainer clears it on the way out, so the file only survives when a Stop
    arrives after that — pressing it a second time, or in another window, while
    a finished run still counts as live. The console then reported STOPPING with
    no process behind it *and* disabled Start, which is the button whose whole
    job is to clear the file. The status has to expire on its own.
    """
    from md.control import Control  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415

    run = tmp_path / "test1"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "update,samples,return,entropy,policy_loss,value_loss,clip_fraction,steps_per_second\n"
        "1,100,1.0,0.5,0.1,0.2,0.1,1000\n",
        encoding="utf-8",
    )
    stale = time.time() - 600
    os.utime(run / "metrics.csv", (stale, stale))
    Control(run).request_stop()

    window = Console(run)
    try:
        window._tick()
        assert window._status.text() == "IDLE"
        assert window._primary.isEnabled(), "Start stayed disabled by a stop nobody can obey"
        assert not window._stop.isEnabled()
    finally:
        window.close()


def test_a_stop_while_the_run_is_going_still_reads_as_stopping(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    """The other half: a run that is finishing its update is *not* a leftover."""
    from md.control import Control  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415

    run = tmp_path / "test1"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "update,samples,return,entropy,policy_loss,value_loss,clip_fraction,steps_per_second\n"
        "1,100,1.0,0.5,0.1,0.2,0.1,1000\n",
        encoding="utf-8",
    )  # written just now
    Control(run).request_stop()

    window = Console(run)
    try:
        window._tick()
        assert window._status.text() == "STOPPING"
        assert not window._primary.isEnabled()
    finally:
        window.close()


def test_the_console_clears_the_controls_when_its_own_run_exits(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    """Hiding a stale STOP in the status line is not the same as removing it.

    The file is a request to a process that has gone; leaving it there means the
    next Start has to clear it, and every other reader — a second console, a
    person running `ls` — is told a run is stopping that is not.
    """
    from md.control import Control  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415

    class _Exited:
        """This console's child, already over."""

        finished = True

        def drain(self) -> list[str]:
            return ["update 14 | ..."]

        def exit_code(self) -> int:
            return 0

    run = tmp_path / "test1"
    run.mkdir()
    (run / "metrics.csv").write_text("update,samples,return\n1,100,1.0\n", encoding="utf-8")
    control = Control(run)
    control.request_stop()

    window = Console(run)
    try:
        window._run = _Exited()  # type: ignore[assignment]
        window._tick()
        assert not control.stopping(), "the STOP outlived the process it was addressed to"
        assert window._status.text() == "IDLE"
    finally:
        window.close()


# ---- naming a run, and deleting one -----------------------------------------
# Both are destructive in opposite directions: one decides what a directory will
# be called for the rest of its life, the other removes one for good. The
# confirmations are answered by monkeypatching, never by a synthetic click — a
# deletion a stray Enter could have caused is not evidence of anything.


def _stopped_run(root: Path, name: str, *, updates: int = 3) -> Path:
    """A run directory nothing has written to for a while."""
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "metrics.csv").write_text(
        "update,samples,return,entropy,policy_loss,value_loss,clip_fraction,steps_per_second\n"
        + "".join(f"{i},{i * 100},1.0,0.5,0.1,0.2,0.1,1000\n" for i in range(1, updates + 1)),
        encoding="utf-8",
    )
    (run / "checkpoints" / "policy-final.pt").write_bytes(b"x" * 4096)
    (run / "update-00100.mdr").write_bytes(b"y" * 512)
    stale = time.time() - 600
    for path in (run / "metrics.csv", run):
        os.utime(path, (stale, stale))
    return run


def test_a_deleted_run_is_gone_from_the_disk_and_the_list(
    qt_app: object,  # noqa: ARG001 — the QApplication has to exist
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md.ui.library import LibraryView  # noqa: PLC0415
    from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    root = tmp_path / "runs"
    doomed = _stopped_run(root, "amber-anvil")
    _stopped_run(root, "brisk-harbour")

    view = LibraryView()
    view.attach(root)
    try:
        row = next(i for i, run in enumerate(view.table._runs) if run.run_id == "amber-anvil")
        view.table._table.selectRow(row)

        view.table._delete_selected()
        assert not doomed.exists(), "the run survived a confirmed delete"
        assert (root / "brisk-harbour").is_dir(), "the wrong run was deleted"
        # And the list is the list again without anyone pressing refresh: a row
        # for a directory that is not there is the worst kind of stale.
        assert view.table._table.rowCount() == 1
    finally:
        view.close()


def test_a_declined_delete_removes_nothing(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md.ui.library import LibraryView  # noqa: PLC0415
    from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
    )
    root = tmp_path / "runs"
    kept = _stopped_run(root, "amber-anvil")

    view = LibraryView()
    view.attach(root)
    try:
        view.table.focus_list()
        view.table._delete_selected()
        assert kept.is_dir()
        assert (kept / "checkpoints" / "policy-final.pt").exists()
    finally:
        view.close()


def test_a_live_run_is_not_deleted_at_all(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a directory a trainer still has open loses the run in pieces.

    So the question is not even asked: a confirmation nobody can answer safely
    is worse than a refusal that says which button to press first.
    """
    from md.ui.library import LibraryView  # noqa: PLC0415
    from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415

    asked: list[str] = []
    warned: list[str] = []

    def _question(*args: object, **kwargs: object) -> object:
        asked.append(str(args[1]))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(str(a[2])))
    )

    root = tmp_path / "runs"
    live = _stopped_run(root, "amber-anvil")
    os.utime(live / "metrics.csv", None)  # written to just now

    view = LibraryView()
    view.attach(root)
    try:
        view.table.focus_list()
        run = view.table.selected()
        assert run is not None and run.live, "the fixture is not live"
        view.table._delete_selected()
        assert live.is_dir()
        assert not asked, "a live run was offered for deletion"
        assert warned and "Stop" in warned[0]
    finally:
        view.close()


def test_a_new_run_is_called_what_it_was_named(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one decision the parameter form cannot carry.

    A run is identified by its directory for the rest of its life — every
    `--resume`, every path, every row in the list — so the name is asked for
    before anything else and is what the directory is called.
    """
    from md.ui.app import Console  # noqa: PLC0415
    from PySide6.QtWidgets import QInputDialog  # noqa: PLC0415

    root = tmp_path / "runs"
    _stopped_run(root, "amber-anvil")

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("Entity policy, 3 seeds", True))
    )
    window = Console(root)
    # The parameter dialog is a separate decision and a modal; what is asserted
    # here is where the run would go, not that one starts.
    monkeypatch.setattr(Console, "_primary_pressed", lambda self: None)
    try:
        window._new_run_from_library()
        assert window._run_dir == root / "entity-policy-3-seeds"
        # Nothing on disk yet. A name typed into a dialog somebody then cancels
        # must not leave a directory behind for the next one to trip over.
        assert not window._run_dir.exists()
        # The typed name is kept for the run that is about to start, because the
        # directory could not be called that.
        assert window._pending_name == "Entity policy, 3 seeds"
    finally:
        window.close()


def test_a_cancelled_name_starts_nothing(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md.ui.app import Console  # noqa: PLC0415
    from PySide6.QtWidgets import QInputDialog  # noqa: PLC0415

    root = tmp_path / "runs"
    existing = _stopped_run(root, "amber-anvil")

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("whatever", False)))
    started: list[int] = []
    monkeypatch.setattr(Console, "_primary_pressed", lambda self: started.append(1))
    window = Console(root)
    try:
        window._new_run_from_library()
        assert not started, "cancelling the name still opened the parameter dialog"
        assert window._run_dir == root, "the console moved off the library anyway"
        assert [child.name for child in root.iterdir()] == [existing.name]
    finally:
        window.close()


def test_reset_names_the_directory_it_moves_to(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other way a run directory comes into existence.

    `high-delta-3` is exactly the name nobody can tell from `high-delta-2` a
    fortnight later, so Reset asks the same question the library does — and
    still suggests the numbered name, so Enter is a whole answer.
    """
    from md.ui.app import Console  # noqa: PLC0415
    from PySide6.QtWidgets import QInputDialog  # noqa: PLC0415

    root = tmp_path / "runs"
    first = _stopped_run(root, "high-delta")

    asked: list[str] = []

    def _typed(*args: object, **kwargs: object) -> tuple[str, bool]:
        asked.append(str(kwargs.get("text", "")))
        return ("Wider clusters", True)

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(_typed))
    window = Console(first)
    try:
        window._reset_pressed()
        assert asked == ["high-delta-2"], "the numbered name was not offered"
        assert window._run_dir == root / "wider-clusters"
        assert first.is_dir(), "Reset deleted the run it moved away from"
    finally:
        window.close()


def test_the_name_lands_in_the_library_when_the_run_starts(
    qt_app: object,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The display name is written by the console, and only once there is a run.

    Both halves matter: written, or naming a run does nothing anybody can see;
    and not before, or every cancelled dialog leaves an orphan directory.
    """
    from md import library  # noqa: PLC0415
    from md.ui import app as app_module  # noqa: PLC0415
    from md.ui.app import Console  # noqa: PLC0415
    from PySide6.QtWidgets import QDialog, QInputDialog  # noqa: PLC0415

    class _Accepted:
        """The parameter dialog, already agreed to."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def command(self) -> list[str]:
            return ["/usr/bin/true"]

    class _Started:
        """A trainer that was launched and does nothing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.finished = True

        def poll(self) -> list[str]:
            return []

    root = tmp_path / "runs"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("Entity policy, 3 seeds", True))
    )
    monkeypatch.setattr(app_module, "ParameterDialog", _Accepted)
    monkeypatch.setattr(app_module, "TrainingRun", _Started)
    # This test is about naming, not about torch: without this it would take the
    # "install a runtime" path on any machine that has none, and that is a modal.
    monkeypatch.setattr(app_module, "can_train", lambda: True)

    window = Console(root)
    try:
        window._new_run_from_library()
        run_dir = window._run_dir
        assert run_dir == root / "entity-policy-3-seeds"
        assert library.read_metadata(run_dir).display_name == "Entity policy, 3 seeds"
        # And it is not written twice: a second Start on the same directory is
        # not a second naming, and would undo a rename made in between.
        library.rename(run_dir, "renamed since")
        window._start()
        assert library.read_metadata(run_dir).display_name == "renamed since"
    finally:
        window.close()


@needs_torch
@needs_native
def test_the_console_and_the_trainer_agree_on_what_a_run_directory_is(
    console,  # noqa: ANN001
    trained_run: Path,
) -> None:
    # One directory, two programs, no shared code path — only the file names.
    # This is the contract that silently breaks when either side is refactored.
    written = {path.name for path in trained_run.iterdir()}
    assert {"metrics.csv", "evals.csv", "config.json", "model.json", "train.log"} <= written
    console._tick()
    # And the console agrees about what is *in* them: the parameter count it
    # shows is the one the trainer wrote, not a plausible number of its own.
    card = json.loads((trained_run / "model.json").read_text(encoding="utf-8"))
    assert f"{card['parameters']:,}" in console._model._headline.text()
