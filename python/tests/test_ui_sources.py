# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for reading a run's artifacts while the trainer is still writing them.

These are the cases the console gets wrong if the tail is naive (docs/ROADMAP.md,
M8, risk 2), and they are testable at all only because ``md.ui.sources`` holds no
Qt: pytest writes a CSV a fragment at a time and asserts what comes back.
"""

from __future__ import annotations

import os
from pathlib import Path

from md.ui.sources import (
    BASELINE_MEAN_SCORE,
    MAX_RUN_CHOICES,
    NO_CHECKPOINTS,
    EvalRow,
    checkpoint_note,
    evals_tail,
    find_runs,
    human_age,
    human_size,
    last_modified,
    list_checkpoints,
    list_recordings,
    log_tail,
    metrics_tail,
    next_run_dir,
    run_choices,
)

HEADER = "update,samples,return,entropy,policy_loss,value_loss,clip_fraction,steps_per_second\r\n"


def _row(update: int, mean_return: str = "4.8700") -> str:
    return f"{update},{update * 131072},{mean_return},1.6020,-0.0032,0.3490,0.0034,214000.0\r\n"


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(text)


def test_a_missing_file_is_an_empty_batch_not_an_error(tmp_path: Path) -> None:
    # A console opened before the run starts is the normal case, not a failure.
    tail = metrics_tail(tmp_path)
    batch = tail.poll()
    assert batch.rows == ()
    assert not batch.restarted


def test_only_the_rows_appended_since_the_last_poll_come_back(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _append(path, HEADER + _row(1) + _row(2))
    tail = metrics_tail(tmp_path)
    assert [row.update for row in tail.poll().rows] == [1, 2]

    assert tail.poll().rows == ()  # nothing new: the file has not moved
    _append(path, _row(3))
    batch = tail.poll()
    assert [row.update for row in batch.rows] == [3]
    assert batch.rows[0].samples == 3 * 131072


def test_a_half_written_line_is_held_back_until_it_ends(tmp_path: Path) -> None:
    # The one that is *wrong* rather than slow: parsing "3,393216,4.8" as a row
    # would draw a value the run never had.
    path = tmp_path / "metrics.csv"
    _append(path, HEADER + _row(1))
    tail = metrics_tail(tmp_path)
    assert [row.update for row in tail.poll().rows] == [1]

    torn = _row(2)
    _append(path, torn[:12])
    assert tail.poll().rows == ()

    _append(path, torn[12:])
    batch = tail.poll()
    assert [row.update for row in batch.rows] == [2]
    assert batch.rows[0].mean_return == 4.87


def test_a_truncated_file_is_read_again_from_the_top(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _append(path, HEADER + _row(1) + _row(2) + _row(3))
    tail = metrics_tail(tmp_path)
    assert len(tail.poll().rows) == 3

    path.write_text(HEADER + _row(1), encoding="utf-8", newline="")
    batch = tail.poll()
    assert batch.restarted  # so the console throws its curves away
    assert [row.update for row in batch.rows] == [1]


def test_a_file_that_disappears_invalidates_what_was_read(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _append(path, HEADER + _row(1))
    tail = metrics_tail(tmp_path)
    assert len(tail.poll().rows) == 1

    path.unlink()
    assert tail.poll().restarted
    assert not tail.poll().restarted  # said once, not every second afterwards


def test_a_nan_return_is_a_gap_rather_than_a_zero(tmp_path: Path) -> None:
    # The trainer writes nan until the first episodes finish; charting that as
    # zero invents a dip, so it has to arrive as "no measurement".
    _append(tmp_path / "metrics.csv", HEADER + _row(1, "nan"))
    row = metrics_tail(tmp_path).poll().rows[0]
    assert row.mean_return is None
    assert row.entropy == 1.602


def test_columns_are_matched_by_name_so_a_new_column_does_not_shift_values(
    tmp_path: Path,
) -> None:
    _append(
        tmp_path / "metrics.csv",
        "update,samples,extra,return,entropy\r\n1,131072,7,4.87,1.60\r\n",
    )
    row = metrics_tail(tmp_path).poll().rows[0]
    assert (row.mean_return, row.entropy) == (4.87, 1.60)
    assert row.value_loss is None  # absent, so nothing is drawn for it


def test_a_header_appended_mid_file_re_maps_the_columns(tmp_path: Path) -> None:
    # What a schema change looks like to a reader that started before it.
    path = tmp_path / "metrics.csv"
    _append(path, "update,samples,return\r\n1,131072,4.87\r\n")
    tail = metrics_tail(tmp_path)
    assert tail.poll().rows[0].mean_return == 4.87

    _append(path, "update,samples,entropy,return\r\n2,262144,1.61,5.20\r\n")
    row = tail.poll().rows[0]
    assert (row.update, row.entropy, row.mean_return) == (2, 1.61, 5.20)


def test_evals_carry_the_baseline_comparable_summary(tmp_path: Path) -> None:
    _append(
        tmp_path / "evals.csv",
        "update,mean_score,min_score,max_score,mean_wave,mean_cities_left,"
        "mean_accuracy,survived,episodes\r\n"
        "50,3014.50,1200,5400,6.500,0.000,0.4200,0,32\r\n",
    )
    row = evals_tail(tmp_path).poll().rows[0]
    assert (row.update, row.mean_score, row.episodes, row.survived) == (50, 3014.5, 32, 0)
    assert row.mean_accuracy == 0.42


def test_a_row_without_a_score_is_not_a_row(tmp_path: Path) -> None:
    _append(tmp_path / "evals.csv", "update,mean_score\r\n50,\r\n51,3014\r\n")
    assert [row.update for row in evals_tail(tmp_path).poll().rows] == [51]


def test_recordings_are_listed_newest_first(tmp_path: Path) -> None:
    for name, when in (("update-00025.mdr", 1000), ("update-00050.mdr", 2000)):
        path = tmp_path / name
        path.write_bytes(b"x" * 80)
        os.utime(path, (when, when))
    (tmp_path / "metrics.csv").write_text("not a recording", encoding="utf-8")

    found = list_recordings(tmp_path)
    assert [r.name for r in found] == ["update-00050", "update-00025"]
    assert [r.update for r in found] == [50, 25]
    assert found[0].size == 80


def test_a_recording_with_no_update_in_its_name_still_lists(tmp_path: Path) -> None:
    (tmp_path / "at-400.mdr").write_bytes(b"x")
    (tmp_path / "from-checkpoint.mdr").write_bytes(b"x")
    found = {r.name: r.update for r in list_recordings(tmp_path)}
    assert found == {"at-400": 400, "from-checkpoint": None}


def test_listing_a_directory_that_is_not_there_is_empty(tmp_path: Path) -> None:
    assert list_recordings(tmp_path / "nope") == []
    assert last_modified(tmp_path / "nope" / "metrics.csv") is None


def test_runs_one_level_down_are_found_newest_first(tmp_path: Path) -> None:
    # What a `runs/` directory looks like after a few experiments: no run of its
    # own, one per sub-directory. The console has to say so rather than "empty".
    for name, when in (("sweep-a", 1000), ("sweep-b", 2000)):
        (tmp_path / name).mkdir()
        path = tmp_path / name / "metrics.csv"
        _append(path, HEADER + _row(1))
        os.utime(path, (when, when))
    (tmp_path / "checkpoints").mkdir()  # not a run: no metrics.csv in it

    assert [path.name for path in find_runs(tmp_path)] == ["sweep-b", "sweep-a"]
    assert find_runs(tmp_path / "sweep-a") == []
    assert find_runs(tmp_path / "not-there") == []


def _run_dir(parent: Path, name: str, when: int) -> Path:
    (parent / name).mkdir(parents=True, exist_ok=True)
    path = parent / name / "metrics.csv"
    _append(path, HEADER + _row(1))
    os.utime(path, (when, when))
    return parent / name


def test_the_picker_offers_the_runs_inside_a_container(tmp_path: Path) -> None:
    # Nested inside tmp_path on purpose: `run_choices` looks at what is *beside*
    # the directory too, and pytest's own tmp root is full of other tests' runs.
    runs = tmp_path / "runs"
    runs.mkdir()
    _run_dir(runs, "sweep-a", 1000)
    _run_dir(runs, "sweep-b", 2000)

    choices = run_choices(runs)
    assert [path.name for path in choices][:2] == ["sweep-b", "sweep-a"]
    # The attached directory is always offered, even holding no run of its own —
    # a picker that cannot show what the window is showing is a bug.
    assert runs.resolve() in choices


def test_the_picker_offers_the_runs_beside_the_attached_one(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    attached = _run_dir(runs, "sweep-a", 1000)
    _run_dir(runs, "sweep-b", 2000)

    choices = run_choices(attached)
    assert [path.name for path in choices] == ["sweep-b", "sweep-a"]


def test_the_picker_does_not_grow_without_bound(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for index in range(MAX_RUN_CHOICES + 5):
        _run_dir(runs, f"sweep-{index:02d}", 1000 + index)
    choices = run_choices(runs / "sweep-00")
    assert len(choices) <= MAX_RUN_CHOICES + 1  # the cap, plus what is attached
    assert (runs / "sweep-00").resolve() in choices


def test_starting_over_picks_the_next_free_directory(tmp_path: Path) -> None:
    # Reset never empties a directory: the checkpoints of the run being abandoned
    # are exactly what you want back when the new settings turn out worse.
    runs = tmp_path / "runs"
    runs.mkdir()
    assert next_run_dir(runs) == tmp_path / "runs-2"

    (tmp_path / "runs-2").mkdir()
    assert next_run_dir(runs) == tmp_path / "runs-3"
    assert next_run_dir(tmp_path / "runs-2") == tmp_path / "runs-3"


def test_a_name_that_already_ends_in_a_number_counts_on_from_it(tmp_path: Path) -> None:
    assert next_run_dir(tmp_path / "sweep-7") == tmp_path / "sweep-8"
    # ...but a dash that is not a counter is part of the name.
    assert next_run_dir(tmp_path / "big-batch") == tmp_path / "big-batch-2"


def test_ages_and_sizes_read_at_a_glance() -> None:
    assert human_age(0.5) == "just now"  # the one case that takes no "ago"
    assert human_age(12) == "12 s ago"
    assert human_age(240) == "4 min ago"
    assert human_age(7200) == "2 h ago"
    assert human_age(600_000) == "6 d ago"
    assert human_size(812) == "812 B"
    assert human_size(79_000) == "79 kB"
    assert human_size(1_200_000) == "1.2 MB"


# ---- checkpoints -------------------------------------------------------------


def _checkpoint(run_dir: Path, name: str, when: int, size: int = 5_000_000) -> Path:
    path = run_dir / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    os.utime(path, (when, when))
    return path


def test_a_run_that_has_saved_nothing_lists_no_checkpoints(tmp_path: Path) -> None:
    assert list_checkpoints(tmp_path) == []


def test_checkpoints_come_back_newest_first(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "policy-00100.pt", 1000)
    _checkpoint(tmp_path, "policy-00200.pt", 2000)
    _checkpoint(tmp_path, "policy-final.pt", 3000)

    found = list_checkpoints(tmp_path)
    assert [c.name for c in found] == ["policy-final", "policy-00200", "policy-00100"]


def test_a_numbered_checkpoint_carries_its_update(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "policy-00200.pt", 2000)
    assert list_checkpoints(tmp_path)[0].iteration == 200


def test_the_final_checkpoint_has_no_update_to_claim(tmp_path: Path) -> None:
    # It is whatever the run ended on, and the file does not say which.
    _checkpoint(tmp_path, "policy-final.pt", 3000)
    assert list_checkpoints(tmp_path)[0].iteration is None


def test_only_torch_files_are_checkpoints(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "policy-00100.pt", 1000)
    _checkpoint(tmp_path, "notes.txt", 1000)
    assert [c.name for c in list_checkpoints(tmp_path)] == ["policy-00100"]


def test_the_model_note_says_what_would_produce_a_checkpoint(tmp_path: Path) -> None:
    assert checkpoint_note([], {}) == NO_CHECKPOINTS


def test_the_model_note_names_the_newest_and_counts_the_rest(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "policy-00100.pt", 1000, size=1_000_000)
    _checkpoint(tmp_path, "policy-00200.pt", 2000, size=1_000_000)

    note = checkpoint_note(list_checkpoints(tmp_path), {})
    assert note.startswith("policy-00200")
    assert "2 saved" in note


def test_the_model_note_scores_the_checkpoint_by_its_own_update(tmp_path: Path) -> None:
    # Not by the most recent evaluation: those are usually the same row and
    # occasionally are not, and mislabelling one as the other is the one lie
    # this panel must not tell.
    _checkpoint(tmp_path, "policy-00200.pt", 2000)
    # Written against the baseline rather than a literal: what it is worth is
    # the scripted agent's business and it moves when the game's scoring does.
    evals = {
        200: EvalRow(200, BASELINE_MEAN_SCORE + 1964, None, None, None, None, None, None, None),
        250: EvalRow(250, BASELINE_MEAN_SCORE - 5000, None, None, None, None, None, None, None),
    }
    note = checkpoint_note(list_checkpoints(tmp_path), evals)
    assert f"scored {BASELINE_MEAN_SCORE + 1964:,.0f}" in note
    assert "1,964 ahead of baseline" in note
    assert f"{BASELINE_MEAN_SCORE - 5000:,.0f}" not in note


def test_an_unscored_checkpoint_simply_says_nothing_about_a_score(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "policy-final.pt", 3000)
    assert "scored" not in checkpoint_note(list_checkpoints(tmp_path), {})


# ---- the trainer's own log ---------------------------------------------------


def test_the_log_tail_yields_the_lines_appended_since_the_last_poll(tmp_path: Path) -> None:
    # The console reads this for a run it did not start; the run writes it
    # itself (md.runlog), which is what makes an attached run's log pane work.
    path = tmp_path / "train.log"
    _append(path, "update 1 | return 4.87\n")
    tail = log_tail(tmp_path)
    assert tail.poll().rows == ("update 1 | return 4.87",)

    _append(path, "update 2 | return 5.10\n")
    assert tail.poll().rows == ("update 2 | return 5.10",)
    assert tail.poll().rows == ()


def test_a_half_written_log_line_is_held_back(tmp_path: Path) -> None:
    path = tmp_path / "train.log"
    _append(path, "update 1 | ret")
    tail = log_tail(tmp_path)
    assert tail.poll().rows == ()

    _append(path, "urn 4.87\n")
    assert tail.poll().rows == ("update 1 | return 4.87",)


def test_a_fresh_run_in_the_same_directory_restarts_the_log(tmp_path: Path) -> None:
    path = tmp_path / "train.log"
    _append(path, "an older, longer run said quite a lot\n")
    tail = log_tail(tmp_path)
    tail.poll()

    path.write_text("update 1\n", encoding="utf-8")
    batch = tail.poll()
    assert batch.restarted
    assert batch.rows == ("update 1",)
