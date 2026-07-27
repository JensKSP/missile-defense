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
