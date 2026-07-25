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
    evals_tail,
    human_age,
    human_size,
    last_modified,
    list_recordings,
    metrics_tail,
    next_run_dir,
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
