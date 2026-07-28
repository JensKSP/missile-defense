# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Discovering runs, naming them, and counting what they cost.

Built against real directories in `tmp_path` rather than mocks, because every
interesting case here *is* a directory shape: a run being written to right now,
a run with a best evaluation whose checkpoint was never kept, a `runs/` holding
eleven experiments instead of one run, and a `LIBRARY.json` somebody edited by
hand. None of those is expressible as a stubbed filesystem without deciding in
advance what the answer is.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from missile_defense.runs import library

EVALS_HEADER = (
    "update,mean_score,min_score,max_score,mean_wave,"
    "mean_cities_left,mean_accuracy,survived,episodes\n"
)


def make_run(
    root: Path,
    name: str,
    *,
    updates: int = 3,
    evals: dict[int, float] | None = None,
    checkpoints: tuple[int, ...] = (),
    recordings: tuple[int, ...] = (),
) -> Path:
    """A run directory in the shape `missile_defense.training.train` leaves one."""
    run = root / name
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "metrics.csv").write_text(
        "update,samples,return,entropy,policy_loss,value_loss,clip_fraction,steps_per_second\n"
        + "".join(f"{i},{i * 100},1.0,0.5,0.1,0.2,0.1,1000\n" for i in range(1, updates + 1)),
        encoding="utf-8",
    )
    if evals:
        (run / "evals.csv").write_text(
            EVALS_HEADER
            + "".join(
                f"{update},{score},0,0,1,0,0.5,0,32\n" for update, score in sorted(evals.items())
            ),
            encoding="utf-8",
        )
    for iteration in checkpoints:
        (run / "checkpoints" / f"policy-{iteration:05d}.pt").write_bytes(b"x" * 2048)
    for iteration in recordings:
        (run / f"update-{iteration:05d}.mdr").write_bytes(b"y" * 512)
    return run


# ---- discovery ---------------------------------------------------------------


def test_a_directory_holding_a_run_is_one(tmp_path: Path) -> None:
    make_run(tmp_path, "runs", updates=5)
    found = library.discover(tmp_path / "runs")
    assert [run.run_id for run in found] == ["runs"]
    assert found[0].updates == 5


def test_a_directory_holding_experiments_lists_them_all(tmp_path: Path) -> None:
    """The other shape, and the one a fortnight of work produces.

    Both have to work without anyone being asked which they have — that is the
    same rule `missile_defense.runs.sources.run_choices` already follows for the picker.
    """
    root = tmp_path / "runs"
    for name in ("alpha", "beta", "gamma"):
        make_run(root, name)
    assert {run.run_id for run in library.discover(root)} == {"alpha", "beta", "gamma"}


def test_a_directory_with_no_metrics_is_not_a_run(tmp_path: Path) -> None:
    """`runs/` itself usually is not one, and a row for a directory is a bug."""
    (tmp_path / "runs" / "notes").mkdir(parents=True)
    assert library.discover(tmp_path / "runs") == []
    assert library.load_run(tmp_path / "runs") is None


def test_an_incomplete_run_still_lists(tmp_path: Path) -> None:
    """A run killed after two updates has no evals, no checkpoints, no episodes.

    It is still the thing the person is looking for when they open the library —
    usually to delete it — so it appears, with the empties said rather than
    guessed.
    """
    make_run(tmp_path, "stub", updates=2)
    run = library.load_run(tmp_path / "stub")
    assert run is not None
    assert run.best_score is None
    assert run.best_checkpoint is None
    assert run.checkpoints == 0
    assert run.recordings == 0


def test_a_run_being_written_to_reads_as_live(tmp_path: Path) -> None:
    run = library.load_run(make_run(tmp_path, "now"))
    assert run is not None
    assert run.live
    assert run.state == "live"


def test_a_run_nothing_has_touched_for_a_while_reads_as_idle(tmp_path: Path) -> None:
    path = make_run(tmp_path, "old")
    stale = time.time() - (library.LIVE_AFTER_S * 3)
    import os  # noqa: PLC0415 — only this test needs to forge a timestamp

    os.utime(path / "metrics.csv", (stale, stale))
    run = library.load_run(path)
    assert run is not None
    assert run.state == library.STATE_IDLE


def test_runs_are_listed_by_when_they_last_moved(tmp_path: Path) -> None:
    """Newest first: while training, the run you want is the one that just wrote."""
    import os  # noqa: PLC0415

    root = tmp_path / "runs"
    for index, name in enumerate(("oldest", "middle", "newest")):
        path = make_run(root, name)
        when = time.time() - ((3 - index) * 1000)
        os.utime(path / "metrics.csv", (when, when))
    assert [run.run_id for run in library.discover(root)] == ["newest", "middle", "oldest"]


# ---- storage -----------------------------------------------------------------


def test_storage_is_split_by_what_it_would_cost_to_lose(tmp_path: Path) -> None:
    """Checkpoints are hours of compute; recordings are minutes of playback.

    Cleanup plans against exactly this split, so a total alone would not be
    enough — "reclaim 4 GB" and "reclaim 4 GB of checkpoints" are different
    offers.
    """
    path = make_run(tmp_path, "big", checkpoints=(100, 200), recordings=(100, 200, 300))
    run = library.load_run(path)
    assert run is not None
    assert run.storage.checkpoints == 2 * 2048
    assert run.storage.recordings == 3 * 512
    assert run.storage.other > 0  # the CSVs
    assert run.storage.total == run.storage.checkpoints + run.storage.recordings + run.storage.other


# ---- the best checkpoint -----------------------------------------------------


def test_the_best_checkpoint_is_the_best_one_that_still_exists(tmp_path: Path) -> None:
    """The trap: the best *score* often has no checkpoint of its own.

    Checkpoints are written every `--checkpoint-every` updates and evaluations
    happen on a different cadence, so a run's peak evaluation routinely has no
    file behind it. Promoting the best score while shipping whatever checkpoint
    survived would be a model card that lies about what it is.
    """
    path = make_run(
        tmp_path,
        "peaky",
        evals={100: 5_000.0, 200: 50_000.0, 300: 20_000.0},
        checkpoints=(100, 300),  # 200 — the best — was never kept
    )
    found = library.best_evaluated_checkpoint(path)
    assert found is not None
    chosen, row = found
    assert chosen.name == "policy-00300.pt"
    assert row.mean_score == 20_000.0


def test_no_evaluation_means_no_best_checkpoint(tmp_path: Path) -> None:
    path = make_run(tmp_path, "unscored", checkpoints=(100,))
    assert library.best_evaluated_checkpoint(path) is None


def test_the_run_reports_its_best_score_even_when_the_checkpoint_is_gone(
    tmp_path: Path,
) -> None:
    """Two different questions, and the library answers both honestly.

    "How good did this run get?" is a fact about the run. "What can I promote?"
    is a fact about what is on disk. Conflating them would either understate the
    run or offer a checkpoint that never scored that.
    """
    run = library.load_run(make_run(tmp_path, "gone", evals={100: 9_000.0}, checkpoints=()))
    assert run is not None
    assert run.best_score == 9_000.0
    assert run.best_checkpoint is None


# ---- naming ------------------------------------------------------------------


def test_a_run_is_named_without_moving_it(tmp_path: Path) -> None:
    """Renaming the directory would break every path inside the run.

    Every `--resume` anyone wrote down, the trainer window watching it, the
    trainer's own open file handles. A name is a label, not an address.
    """
    path = make_run(tmp_path, "runs-7")
    library.rename(path, "amber anvil")
    run = library.load_run(path)
    assert run is not None
    assert run.run_id == "runs-7"  # unchanged, and still where it was
    assert run.display_name == "amber anvil"
    assert run.name == "amber anvil"
    assert path.is_dir()


def test_a_run_with_no_name_shows_its_directory(tmp_path: Path) -> None:
    run = library.load_run(make_run(tmp_path, "runs-2"))
    assert run is not None
    assert run.name == "runs-2"


def test_a_hand_mangled_metadata_file_loses_the_name_and_not_the_run(tmp_path: Path) -> None:
    """The trainer's file, in a directory a person can open in an editor.

    Refusing to list the run over a broken name would lose them the run, which
    is a far worse outcome than losing a name they can retype.
    """
    path = make_run(tmp_path, "runs-3")
    (path / library.LIBRARY_NAME).write_text("{ not json", encoding="utf-8")
    run = library.load_run(path)
    assert run is not None
    assert run.name == "runs-3"


def test_renaming_a_live_run_replaces_the_file_atomically(tmp_path: Path) -> None:
    """Written while the trainer writes everything else in the same directory."""
    path = make_run(tmp_path, "live")
    library.rename(path, "first")
    library.rename(path, "second")
    assert library.read_metadata(path).display_name == "second"
    assert not list(path.glob("*.tmp"))


def test_a_default_name_is_readable_and_never_collides() -> None:
    """Two short words, because the complaint is that `runs-11` and `runs-12`
    are the same shape and a timestamp is worse."""
    first = library.default_name(seed=0)
    assert "-" in first and first.islower()
    assert library.default_name([first], seed=0) != first
    # And it keeps going rather than giving up or repeating.
    taken = {library.default_name(seed=n) for n in range(40)}
    assert len(taken) > 20


def test_a_typed_name_becomes_a_directory_name_a_shell_can_use() -> None:
    """The name is typed by a person and the directory is typed at a prompt,
    put in a `--resume` and quoted by nobody."""
    assert library.run_id_for("Entity policy, 3 seeds") == "entity-policy-3-seeds"
    assert library.run_id_for("  spaced  out  ") == "spaced-out"
    # A name made entirely of punctuation is still a run, not a directory called
    # `` or `-`.
    assert library.run_id_for("!?!") == "run"


def test_a_taken_directory_name_is_suffixed_rather_than_refused(tmp_path: Path) -> None:
    """Refusing at the moment somebody is trying to start training would be a
    dialog in the way; a name that is not taken is the answer they wanted."""
    assert library.run_id_for("delta", ["delta"]) == "delta-2"
    assert library.run_id_for("delta", ["delta", "delta-2"]) == "delta-3"
    # Case-insensitively: two directories differing only in case are the same
    # directory on Windows and macOS.
    assert library.run_id_for("Delta", ["delta"]) == "delta-2"

    root = tmp_path / "runs"
    make_run(root, "high-delta")
    assert library.new_run_dir(root, "High Delta") == root / "high-delta-2"


def test_a_new_run_avoids_a_directory_no_trainer_has_written_into_yet(tmp_path: Path) -> None:
    """`discover` cannot see a directory without a `metrics.csv`, and `mkdir`
    can: a new run that landed on top of one would mix two runs together."""
    root = tmp_path / "runs"
    (root / "amber-anvil").mkdir(parents=True)  # started, then cancelled
    assert not library.discover(root)
    assert library.new_run_dir(root, "amber-anvil") == root / "amber-anvil-2"


def test_a_new_run_in_a_library_that_does_not_exist_yet_is_named_anyway(tmp_path: Path) -> None:
    """The first run on a fresh machine. Nothing is created to find that out."""
    root = tmp_path / "never-made"
    assert library.new_run_dir(root, "First Run") == root / "first-run"
    assert not root.exists()


# ---- pins --------------------------------------------------------------------


def test_a_pinned_recording_is_remembered_by_name(tmp_path: Path) -> None:
    """By name and not by path: a restored archive lands somewhere else, and a
    pin that pointed at the old absolute path would protect nothing."""
    path = make_run(tmp_path, "pins", recordings=(100, 200))
    library.pin(path, path / "update-00100.mdr")
    run = library.load_run(path)
    assert run is not None
    assert run.pinned == ("update-00100.mdr",)

    library.unpin(path, path / "update-00100.mdr")
    reread = library.load_run(path)
    assert reread is not None
    assert reread.pinned == ()


def test_pinning_the_same_recording_twice_pins_it_once(tmp_path: Path) -> None:
    path = make_run(tmp_path, "pins", recordings=(100,))
    library.pin(path, path / "update-00100.mdr")
    library.pin(path, path / "update-00100.mdr")
    assert library.read_metadata(path).pinned == ["update-00100.mdr"]


def test_a_name_and_a_pin_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """One file holds both, so each writer has to read before it writes."""
    path = make_run(tmp_path, "both", recordings=(100,))
    library.rename(path, "kept")
    library.pin(path, path / "update-00100.mdr")
    library.set_note(path, "the one that learned to wait")
    metadata = library.read_metadata(path)
    assert metadata.display_name == "kept"
    assert metadata.pinned == ["update-00100.mdr"]
    assert metadata.note == "the one that learned to wait"


# ---- the managed root --------------------------------------------------------


def test_a_path_outside_the_managed_root_is_refused(tmp_path: Path) -> None:
    """Asked before anything is written or deleted.

    "Delete everything under the path I was handed" is a sentence that has ended
    badly for other programs, and a run directory arrives from a picker, an
    environment variable or a command line.
    """
    root = tmp_path / "runs"
    root.mkdir()
    assert library.within(root, root / "alpha")
    assert library.within(root, root)
    assert not library.within(root, tmp_path / "elsewhere")
    # The one a string prefix gets wrong.
    assert not library.within(root, tmp_path / "runs-2")


# ---- presentation ------------------------------------------------------------


def test_a_row_says_what_the_run_is_and_where_it_got_to(tmp_path: Path) -> None:
    path = make_run(tmp_path, "runs-4", updates=12, evals={10: 4_321.0}, checkpoints=(10,))
    library.rename(path, "brisk-harbour")
    run = library.load_run(path)
    assert run is not None
    line = library.describe(run)
    assert "brisk-harbour" in line
    assert "runs-4" in line  # the id too: it is what a path or a --resume uses
    assert "12 updates" in line
    assert "4,321" in line


def test_a_promoted_model_is_named_after_the_run_and_the_update(tmp_path: Path) -> None:
    """Because a run can be promoted more than once, and two `amber-anvil` rows
    in a league table are exactly the confusion naming was meant to end."""
    path = make_run(tmp_path, "runs-5", evals={800: 1.0}, checkpoints=(800,))
    library.rename(path, "amber-anvil")
    run = library.load_run(path)
    assert run is not None
    name = library.model_name_for(run, path / "checkpoints" / "policy-00800.pt")
    assert name == "amber-anvil @800"


def test_metadata_survives_a_round_trip_through_json(tmp_path: Path) -> None:
    path = make_run(tmp_path, "rt")
    library.write_metadata(
        path, library.Metadata(display_name="x", note="y", pinned=["b.mdr", "a.mdr"])
    )
    stored = json.loads((path / library.LIBRARY_NAME).read_text(encoding="utf-8"))
    assert stored["pinned"] == ["a.mdr", "b.mdr"]  # sorted, so a diff is stable
    assert library.read_metadata(path).display_name == "x"


def test_the_library_and_the_dashboard_call_a_stopped_run_the_same_thing() -> None:
    """One vocabulary for one fact.

    A finished run was "stopped" in the run list and "IDLE" in the dashboard's
    pill — the same condition under two words, which reads as two facts. The
    names now live in `missile_defense.runs.library` and both views spell them from there; this
    fails if a third spelling appears.

    Skipped without PySide6 rather than failing: the trainer is an optional
    part of this project (LGPL-3 where the rest is MIT, so the game never
    depends on it), and the quality gate runs where it is not installed.
    """
    pytest.importorskip("PySide6", reason="the trainer is optional; see README")
    from missile_defense.ui import app

    assert library.STATE_IDLE in app.STATUS
    assert library.STATE_LIVE in app.STATUS
    assert app.STATUS[library.STATE_IDLE][0] == "IDLE"
    assert app.STATUS[library.STATE_LIVE][0] == "LIVE"
