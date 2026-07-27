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
    CANONICAL_AIM_TRAIL,
    CANONICAL_BASELINE_MEAN_SCORE,
    CANONICAL_FRAME_SKIP,
    CANONICAL_INFERENCE_DEVICE,
    CANONICAL_LADDER,
    CANONICAL_MAX_TICKS,
    CANONICAL_REACTION_DELAY,
    CANONICAL_SEED_OFFSET,
    CANONICAL_SPLIT,
    SEEDS_PER_SPLIT,
    VALIDATION_LADDER,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
)

from .harness import TINY_RUN, needs_native, needs_qt, needs_torch

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


METRICS_HEADER = (
    "update,samples,return,entropy,policy_loss,value_loss,clip_fraction,steps_per_second\n"
)
#: The handicap columns are part of the protocol: `is_canonical_benchmark`
#: compares them, so a row without them is not comparable with the ladder no
#: matter what else it says.
EVALS_HEADER = (
    "update,mean_score,seed_split,seed_offset,seed_count,frame_skip,max_ticks,"
    "inference_device,aim_trail,reaction_delay\n"
)


def _canonical_eval(update: int, score: float) -> str:
    return (
        f"{update},{score},{CANONICAL_SPLIT},{CANONICAL_SEED_OFFSET},{SEEDS_PER_SPLIT},"
        f"{CANONICAL_FRAME_SKIP},{CANONICAL_MAX_TICKS},{CANONICAL_INFERENCE_DEVICE},"
        f"{CANONICAL_AIM_TRAIL},{CANONICAL_REACTION_DELAY}\n"
    )


def test_a_new_run_starts_the_tiles_at_nothing_notes_included(
    qt_app: object, tmp_path: Path
) -> None:
    # A tile's *note* is the line that describes one particular measurement, so
    # it is the one that goes stale. "—" over "262,144,000 samples" reads as a
    # fresh run that has somehow already seen a quarter of a billion samples.
    from md.ui.app import Console  # noqa: PLC0415

    first = tmp_path / "first"
    first.mkdir()
    (first / "metrics.csv").write_text(
        METRICS_HEADER + "500,262144000,38.2,0.81,-0.003,0.42,0.05,214000.0\n", encoding="utf-8"
    )
    (first / "evals.csv").write_text(
        EVALS_HEADER + _canonical_eval(500, CANONICAL_BASELINE_MEAN_SCORE + 100), encoding="utf-8"
    )
    window = Console(first)
    try:
        window._tick()
        assert window._tile_update._value.text() == "500"
        assert "262,144,000 samples" in window._tile_update._note.text()
        assert "beats HIGH" in window._tile_score._note.text()

        # Reset… aimed somewhere new, which is how a new run begins.
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        window._attach(fresh)
        window._tick()
        assert window._tile_update._value.text() == "—"
        assert "samples" not in window._tile_update._note.text()
        assert window._tile_score._value.text() == "—"
        assert "beats" not in window._tile_score._note.text()

        # And the other way a run restarts: the same directory, written afresh.
        (fresh / "metrics.csv").write_text(
            METRICS_HEADER + "700,367001600,39.1,0.78,-0.002,0.40,0.05,214000.0\n",
            encoding="utf-8",
        )
        window._tick()
        assert "367,001,600 samples" in window._tile_update._note.text()
        (fresh / "metrics.csv").write_text(
            METRICS_HEADER, encoding="utf-8"
        )  # truncated: a new run in the same place
        window._tick()
        assert window._tile_update._value.text() == "—"
        assert "samples" not in window._tile_update._note.text()
    finally:
        window.close()


def test_a_protocol_change_starts_a_new_score_curve_and_controls_the_ladder(
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
        # A validation curve gets the ladder measured on the validation block —
        # not the canonical one, whose rungs are a few hundred points away.
        assert window._score._baselines == [rung.mean_score for rung in VALIDATION_LADDER.rungs]
        assert "validation" in window._tile_score._note.text()
        assert "beats HIGH" in window._tile_score._note.text()

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
        # The ladder switched blocks with the curve: all three canonical rungs,
        # ascending, and the tile says where on them it sits and which they are.
        assert window._score._baselines == [rung.mean_score for rung in CANONICAL_LADDER.rungs]
        assert window._score._baselines[-1] == CANONICAL_BASELINE_MEAN_SCORE
        note = window._tile_score._note.text()
        assert "beats HIGH by 100" in note
        assert "held-out canonical" in note
    finally:
        window.close()


def test_a_protocol_nothing_was_measured_under_gets_no_ladder(
    qt_app: object, tmp_path: Path
) -> None:
    # Frame skip 1 is a different game — the agent reacts four times as often —
    # so no rung on either block applies, and a chart that drew one anyway would
    # be comparing two different measurements.
    from md.ui.app import Console  # noqa: PLC0415

    (tmp_path / "evals.csv").write_text(
        "update,mean_score,seed_split,seed_offset,seed_count,frame_skip,"
        "max_ticks,inference_device\n"
        f"50,70000,{VALIDATION_SPLIT},{VALIDATION_SEED_OFFSET},{SEEDS_PER_SPLIT},"
        f"1,{CANONICAL_MAX_TICKS},cuda\n",
        encoding="utf-8",
    )
    window = Console(tmp_path)
    try:
        window._tick()
        assert window._score._count == 1
        assert window._score._baselines == []
        note = window._tile_score._note.text()
        assert "beats" not in note
        assert "frame skip 1" in note  # and it says what makes it incomparable
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


def test_a_preset_fills_the_form_and_editing_it_stops_claiming_to_be_one(
    qt_app: object, tmp_path: Path
) -> None:
    # The picker is the whole point of naming a set of options: choosing "good"
    # has to produce the run that recipe describes, and the moment a value is
    # edited by hand the form is no longer that preset and must stop saying so.
    from md import presets  # noqa: PLC0415
    from md.ui.forms import ParameterDialog, _read  # noqa: PLC0415
    from md.ui.params import read_params  # noqa: PLC0415
    from md.ui.runner import PACKAGE_PATH  # noqa: PLC0415

    file = tmp_path / "presets.json"
    dialog = ParameterDialog(
        read_params(PACKAGE_PATH / "md"),
        python="/usr/bin/python3",
        out_dir=tmp_path,
        presets_file=file,
    )
    shown = lambda name: _read(dialog._editors[name])  # noqa: E731 — one expression, used often
    try:
        assert dialog._presets.itemText(0) == presets.CUSTOM
        assert [dialog._presets.itemText(i) for i in range(1, 4)] == ["fast", "good", "best"]

        dialog._presets.setCurrentIndex(dialog._presets.findData("good"))
        assert (shown("architecture"), shown("envs"), shown("steps"), shown("updates")) == (
            "entity",
            "1024",
            "256",
            "1000",
        )
        # The command line still only carries the *difference* from the trainer's
        # defaults, which is what makes it readable — three of `good`'s four
        # values are the defaults, so one flag describes the run exactly. The
        # preset pins all four anyway, so the recipe survives a default changing.
        assert " ".join(dialog.command()).count("--") == 2  # --architecture, --out-dir
        assert dialog.values() == {"architecture": "entity"}

        # Switching presets replaces what the last one set rather than layering
        # on top of it: `fast` is an mlp run, and `entity` must not survive it.
        dialog._presets.setCurrentIndex(dialog._presets.findData("fast"))
        assert (shown("architecture"), shown("envs"), shown("steps")) == ("mlp", "4096", "128")
        assert "--architecture" not in " ".join(dialog.command())  # mlp is the default

        # A value typed by hand: still fast's numbers, no longer called "fast".
        dialog._editors["updates"].setValue(37)  # type: ignore[attr-defined]
        assert dialog._presets.currentIndex() == 0
        assert dialog.values()["updates"] == "37"
        assert dialog.values()["envs"] == "4096"
    finally:
        dialog.close()


def test_the_update_tile_says_how_fast_it_is_going_and_how_long_is_left(
    qt_app: object, tmp_path: Path
) -> None:
    # "Is the GPU actually being used?" was unanswerable from this window: the
    # trainer prints steps/s every update and the console kept it to itself,
    # leaving a samples counter as the only sign of life. 42k steps/s is a
    # saturated 5090 on the relational architecture — ten times slower than the
    # flat one, and that difference is what looks like an idle card.
    import json  # noqa: PLC0415

    from md.ui.app import Console  # noqa: PLC0415

    (tmp_path / "config.json").write_text(
        json.dumps({"train": {"updates": 4000, "resume": None}}), encoding="utf-8"
    )
    (tmp_path / "metrics.csv").write_text(
        METRICS_HEADER
        + "1,1048576,nan,1.6399,-0.000364,4.930708,0.0017,38774.6\n"
        + "2,2097152,54.0783,1.8301,-0.000491,12.290598,0.0047,38894.0\n",
        encoding="utf-8",
    )
    window = Console(tmp_path)
    try:
        window._tick()
        note = window._tile_update._note.text()
        assert "2,097,152 samples" in note
        assert "39k steps/s" in note
        # 3,998 updates left at 1,048,576 samples each, at 38,894 steps/s —
        # very nearly the 30 hours `best` was measured at.
        assert "~1 d 5 h left" in note
    finally:
        window.close()


def test_a_run_that_cannot_know_its_horizon_still_says_how_fast_it_is_going(
    qt_app: object, tmp_path: Path
) -> None:
    # A resumed run's `updates` counts additional updates from an iteration that
    # lives in a checkpoint. The rate is still worth showing; a confident "4 h
    # left" on a run with a day to go is not.
    import json  # noqa: PLC0415

    from md.ui.app import Console  # noqa: PLC0415

    (tmp_path / "config.json").write_text(
        json.dumps({"train": {"updates": 4000, "resume": "checkpoints/policy-final.pt"}}),
        encoding="utf-8",
    )
    (tmp_path / "metrics.csv").write_text(
        METRICS_HEADER + "812,212860928,54.0783,1.83,-0.0004,12.29,0.0047,38894.0\n",
        encoding="utf-8",
    )
    window = Console(tmp_path)
    try:
        window._tick()
        note = window._tile_update._note.text()
        assert "39k steps/s" in note
        assert "left" not in note
    finally:
        window.close()


def test_vram_gets_a_meter_of_its_own_under_the_gpu_load(qt_app: object) -> None:
    # GPU utilisation tells you the card is busy, which you knew — you started
    # the run. GPU *memory* is what ends a run eight hours in, and it used to be
    # a fragment of the caption line under the bars.
    from md.ui.meters import GB, SystemPanel  # noqa: PLC0415
    from md.ui.system import GpuSample, Sample  # noqa: PLC0415

    class _Monitor:
        gpu_note = "no GPU"

        def __init__(self, gpu: GpuSample | None) -> None:
            self._gpu = gpu

        def sample(self) -> Sample:
            return Sample(cpu=11.0, memory_used=16 * GB, memory_total=62 * GB, gpu=self._gpu)

    card = GpuSample(
        name="NVIDIA GeForce RTX 5090",
        utilisation=10.0,
        memory_used=int(17.4 * GB),
        memory_total=32 * GB,
        temperature=34.0,
    )
    panel = SystemPanel(_Monitor(card))  # type: ignore[arg-type]
    try:
        panel.refresh()
        assert panel._vram._value.text() == "17.4 / 32 GB"
        assert panel._vram._bar.value() == 54  # 17.4 of 32, as a bar like ram's
        assert panel._gpu._value.text() == "10%"
        # The caption line keeps the card and its temperature and stops
        # duplicating what now has a bar.
        assert panel._note.text() == "NVIDIA GeForce RTX 5090 · 34 °C"
    finally:
        panel.close()

    # A card that reports a size but not its usage: the size, and no bar. A bar
    # at zero would read as an empty card rather than as an unanswered question.
    quiet = SystemPanel(_Monitor(GpuSample(name="card", memory_total=8 * GB)))  # type: ignore[arg-type]
    try:
        quiet.refresh()
        assert quiet._vram._value.text() == "— / 8 GB"
        assert not quiet._vram._bar.isVisible()
    finally:
        quiet.close()

    # And no card at all leaves both GPU rows empty rather than at zero.
    headless = SystemPanel(_Monitor(None))  # type: ignore[arg-type]
    try:
        headless.refresh()
        assert headless._vram._value.text() == "—"
        assert headless._gpu._value.text() == "—"
    finally:
        headless.close()


def test_the_dialog_says_what_a_run_will_cost_the_card_before_it_starts(
    qt_app: object, tmp_path: Path
) -> None:
    # `best` shipped in a state that ran out of memory on the card it was
    # designed for, and nothing on this dialog hinted at it. The estimate is
    # shown whether or not it fits, because 17 GiB is the difference between
    # "start it and go to bed" and "start it and stop using the machine".
    from md import footprint  # noqa: PLC0415
    from md.ui.forms import ParameterDialog  # noqa: PLC0415
    from md.ui.params import read_params  # noqa: PLC0415
    from md.ui.runner import PACKAGE_PATH  # noqa: PLC0415

    def dialog_with(free_gib: float) -> ParameterDialog:
        return ParameterDialog(
            read_params(PACKAGE_PATH / "md"),
            python="/usr/bin/python3",
            out_dir=tmp_path,
            presets_file=tmp_path / "presets.json",
            free_vram_bytes=int(free_gib * footprint.GIB),
        )

    roomy = dialog_with(30)
    try:
        roomy._presets.setCurrentIndex(roomy._presets.findData("best"))
        assert roomy._memory is not None
        text = roomy._memory.text()
        assert "GiB of GPU memory" in text
        assert "30.0 GiB free" in text
        assert roomy._memory.property("role") == "note"
        assert "⚠" not in text
    finally:
        roomy.close()

    cramped = dialog_with(8)
    try:
        cramped._presets.setCurrentIndex(cramped._presets.findData("best"))
        assert cramped._memory is not None
        warning = cramped._memory.text()
        assert warning.startswith("⚠")
        assert "run out" in warning
        assert "--minibatches" in warning, "the warning does not name the cheapest fix"
        assert cramped._memory.property("role") == "warning"

        # And `fast` fits in the same 8 GiB, so the warning is about the
        # configuration rather than a permanent scold on a small card.
        cramped._presets.setCurrentIndex(cramped._presets.findData("fast"))
        assert "⚠" not in cramped._memory.text()
    finally:
        cramped.close()


def test_the_dialog_saves_a_preset_and_refuses_to_overwrite_a_built_in(
    qt_app: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from md import presets  # noqa: PLC0415
    from md.ui import forms as forms_module  # noqa: PLC0415
    from md.ui.forms import ParameterDialog  # noqa: PLC0415
    from md.ui.params import read_params  # noqa: PLC0415
    from md.ui.runner import PACKAGE_PATH  # noqa: PLC0415

    warned: list[str] = []
    monkeypatch.setattr(
        forms_module.QMessageBox,
        "warning",
        staticmethod(lambda _parent, _title, text: warned.append(text)),
    )

    file = tmp_path / "presets.json"
    dialog = ParameterDialog(
        read_params(PACKAGE_PATH / "md"),
        python="/usr/bin/python3",
        out_dir=tmp_path,
        presets_file=file,
    )
    try:
        dialog._editors["envs"].setValue(8192)  # type: ignore[attr-defined]
        dialog._store("overnight", "the long one")
        # Saved, listed, and selected — the name you just gave it is what the
        # picker should be showing, not "custom".
        assert presets.find("overnight", file) is not None
        assert dialog._presets.currentText() == "overnight"
        assert dialog._update_preset.isEnabled()
        assert dialog._delete_preset.isEnabled()

        # A built-in is read-only: selecting one disables both, and saving over
        # its name is refused with a reason rather than silently ignored.
        dialog._presets.setCurrentIndex(dialog._presets.findData("good"))
        assert not dialog._update_preset.isEnabled()
        assert not dialog._delete_preset.isEnabled()
        dialog._store("good", "")
        assert warned and "built-in" in warned[0]
        assert presets.find("good", file) == presets.find("good")
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


# ---- what a run was started with, and continuing it --------------------------


@needs_torch
@needs_native
def test_the_console_shows_what_the_run_was_started_with(console) -> None:  # noqa: ANN001
    """`config.json` has been written since there were runs; nothing read it.

    The button beside Log is the answer to "what was this one trained with?" —
    asked of whichever run turned out to be interesting, usually well after the
    terminal that started it has gone. A dialog rather than a panel, because it
    is read once and closed and must cost the curve no space at all.
    """
    console._tick()
    dialog = console._parameters_dialog()
    settings = {row.name: row for row in dialog.panel.settings}

    assert settings["envs"].value == TINY_RUN["--envs"]
    # And the reasoning written beside the field in the trainer's own source,
    # so the panel teaches rather than listing twenty-six unexplained numbers.
    assert "Environments stepped in parallel" in settings["envs"].help
    # The tiny run is four envs against a default of 1,024: marked, because what
    # a run *changed* is the only readable summary of what it is.
    assert settings["envs"].changed
    dialog.close()


@needs_torch
@needs_native
def test_a_stopped_run_with_checkpoints_offers_to_continue(
    qt_app: object, trained_run: Path, tmp_path: Path
) -> None:
    """Idle in a directory that already has checkpoints is not "start"."""
    from md.ui.app import LIVE_AFTER_S, Console  # noqa: PLC0415
    from md.ui.runner import can_train  # noqa: PLC0415

    if not can_train():
        pytest.skip("no training runtime on this machine, so the button says so instead")
    quiet = tmp_path / "stopped"
    shutil.copytree(trained_run, quiet)
    stale = time.time() - (LIVE_AFTER_S * 2)
    for path in quiet.rglob("*"):
        os.utime(path, (stale, stale))

    window = Console(quiet)
    try:
        window._tick()
        assert window._primary.text() == "Continue"
    finally:
        window.close()


@needs_torch
@needs_native
def test_continuing_restates_the_original_run_rather_than_the_defaults(
    qt_app: object, trained_run: Path
) -> None:
    """The dialog behind *Continue*, filled in from the run it would continue.

    Restated on the command line rather than left to the trainer's inheritance:
    the preview in that dialog is supposed to be a command you could paste into
    a terminal, and `--resume x` alone is a command whose meaning lives in a
    file.
    """
    from md import runconfig  # noqa: PLC0415
    from md.ui import sources as ui_sources  # noqa: PLC0415
    from md.ui.forms import ParameterDialog  # noqa: PLC0415
    from md.ui.params import TRAINER_SOURCES, read_params  # noqa: PLC0415

    checkpoints = ui_sources.list_checkpoints(trained_run)
    assert checkpoints, "the trained run left no checkpoint to continue from"
    dialog = ParameterDialog(
        read_params(TRAINER_SOURCES),
        python="/usr/bin/python3",
        out_dir=trained_run,
        checkpoints=checkpoints,
        initial=runconfig.options(runconfig.read(trained_run)),
        resume=checkpoints[0].path,
    )
    try:
        command = " ".join(dialog.command())
        assert f"--envs {TINY_RUN['--envs']}" in command
        assert f"--max-ticks {TINY_RUN['--max-ticks']}" in command
        assert f"--resume {checkpoints[0].path}" in command
        # And the button says which of the two things it is about to do.
        assert dialog._go.text() == "Continue run"

        # Choosing to start over instead is still one click, and the dialog
        # stops calling itself a continuation the moment it is.
        dialog._resume.setCurrentIndex(0)
        assert "--resume" not in " ".join(dialog.command())
        assert dialog._go.text() == "Start run"
    finally:
        dialog.close()


def test_a_runtime_that_stopped_working_turns_start_back_into_set_up(
    qt_app: object, tmp_path: Path
) -> None:
    """The button must not offer what the machine can no longer do.

    `md.runtime.Runtime.status` reads a manifest and checks that a file exists,
    and both stay true of a runtime whose torch was deleted to reclaim disk or
    whose driver moved under it. The console believed that, showed Start, and
    the press appeared to do nothing — the failure surfacing later and somewhere
    unrelated. A background check now asks the runtime to prove it; this is what
    its "no" has to do to the window.

    Deliberately not on the `console` fixture: that one trains, so it needs
    torch and skips wherever torch is absent — which is the quality gate, and
    every machine a first-time reader has. An empty directory is enough to ask
    what a button says.
    """
    from md.ui.app import Console  # noqa: PLC0415 — optional dependency
    from md.ui.runner import can_train  # noqa: PLC0415

    window = Console(tmp_path)
    try:
        window._runtime_verified(False, "torch will not import")  # noqa: SLF001 — the seam
        assert window._primary.text() == "Set up training…"  # noqa: SLF001
        # …and says so, because a changed button with no reason is a dead end.
        assert "torch will not import" in window.statusBar().currentMessage()

        # A runtime that proves itself puts the offer back — unless nothing is
        # installed at all, which is a different "no" and already had a label.
        window._runtime_verified(True, "NVIDIA (CUDA) — torch 2.13.0 on cuda")  # noqa: SLF001
        expected = "Start" if can_train() else "Set up training…"
        assert window._primary.text() == expected  # noqa: SLF001
    finally:
        window.close()
