# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for reading a run's artifacts while the trainer is still writing them.

These are the cases the console gets wrong if the tail is naive (docs/ROADMAP.md,
M8, risk 2), and they are testable at all only because ``md.ui.sources`` holds no
Qt: pytest writes a CSV a fragment at a time and asserts what comes back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from md.benchmark import (
    CANONICAL_AIM_TRAIL,
    CANONICAL_FRAME_SKIP,
    CANONICAL_INFERENCE_DEVICE,
    CANONICAL_LADDER,
    CANONICAL_MAX_TICKS,
    CANONICAL_REACTION_DELAY,
    CANONICAL_SEED_OFFSET,
    CANONICAL_SPLIT,
    NO_LADDER,
    SEEDS_PER_SPLIT,
    VALIDATION_LADDER,
    VALIDATION_SEED_OFFSET,
    VALIDATION_SPLIT,
)
from md.ui.sources import (
    BASELINE_MEAN_SCORE,
    MAX_RUN_CHOICES,
    NO_CHECKPOINTS,
    RECENT_POINTS,
    EvalRow,
    Peak,
    baseline_lines,
    checkpoint_note,
    curve_note,
    eval_protocol_label,
    evals_tail,
    find_runs,
    human_age,
    human_duration,
    human_rate,
    human_size,
    is_canonical_benchmark,
    ladder_note,
    last_modified,
    list_checkpoints,
    list_recordings,
    log_tail,
    matching_eval_protocol,
    metrics_tail,
    next_run_dir,
    peak_note,
    planned_updates,
    readout_note,
    row_ladder,
    run_choices,
    same_eval_series,
    shared_ladder,
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


def test_an_atomically_replaced_file_is_read_again_from_the_top(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _append(path, HEADER + _row(1))
    tail = metrics_tail(tmp_path)
    assert [row.update for row in tail.poll().rows] == [1]

    replacement = tmp_path / ".metrics.csv.new"
    _append(replacement, HEADER + _row(9) + _row(10))
    os.replace(replacement, path)

    batch = tail.poll()
    assert batch.restarted
    assert [row.update for row in batch.rows] == [9, 10]


def test_a_csv_tail_can_be_rewound_when_its_consumer_changes_protocol(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evals.csv"
    _append(
        path,
        "update,mean_score,seed_split\r\n10,100.0,validation\r\n20,200.0,heldout\r\n",
    )
    tail = evals_tail(tmp_path)
    assert [row.update for row in tail.poll().rows] == [10, 20]
    assert tail.poll().rows == ()

    tail.rewind()
    batch = tail.poll()
    assert [row.update for row in batch.rows] == [10, 20]
    assert not batch.restarted


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


def test_evals_carry_the_summary_and_reproduction_protocol(tmp_path: Path) -> None:
    _append(
        tmp_path / "evals.csv",
        "update,mean_score,min_score,max_score,mean_wave,mean_cities_left,"
        "mean_accuracy,survived,episodes,seed_split,seed_offset,seed_count,frame_skip,"
        "max_ticks,inference_device\r\n"
        "50,3014.50,1200,5400,6.500,0.000,0.4200,0,32,"
        "validation,0,32,7,9999,cpu\r\n",
    )
    row = evals_tail(tmp_path).poll().rows[0]
    assert (row.update, row.mean_score, row.episodes, row.survived) == (50, 3014.5, 32, 0)
    assert row.mean_accuracy == 0.42
    assert (row.seed_split, row.seed_count, row.frame_skip) == (VALIDATION_SPLIT, 32, 7)
    assert row.seed_offset == VALIDATION_SEED_OFFSET
    assert (row.max_ticks, row.inference_device) == (9_999, "cpu")


def test_the_handicap_a_score_was_earned_under_is_read_with_the_rest(tmp_path: Path) -> None:
    # The trainer writes these two columns and `is_canonical_benchmark` compares
    # them, so a reader that quietly dropped them left every row at `None` —
    # which is *not* "presumably canonical". Every real run read as "nonstandard
    # protocol" and no score could ever be shown against the ladder.
    _append(
        tmp_path / "evals.csv",
        "update,mean_score,seed_split,seed_offset,seed_count,frame_skip,"
        "max_ticks,inference_device,aim_trail,reaction_delay\r\n"
        f"50,3014.50,{CANONICAL_SPLIT},{CANONICAL_SEED_OFFSET},{SEEDS_PER_SPLIT},"
        f"{CANONICAL_FRAME_SKIP},{CANONICAL_MAX_TICKS},{CANONICAL_INFERENCE_DEVICE},"
        f"{CANONICAL_AIM_TRAIL},{CANONICAL_REACTION_DELAY}\r\n",
    )
    row = evals_tail(tmp_path).poll().rows[0]
    assert (row.aim_trail, row.reaction_delay) == (CANONICAL_AIM_TRAIL, CANONICAL_REACTION_DELAY)
    assert is_canonical_benchmark(row)


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
        200: EvalRow(
            200,
            BASELINE_MEAN_SCORE + 1964,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            seed_split=CANONICAL_SPLIT,
            seed_offset=CANONICAL_SEED_OFFSET,
            seed_count=SEEDS_PER_SPLIT,
            frame_skip=CANONICAL_FRAME_SKIP,
            max_ticks=CANONICAL_MAX_TICKS,
            inference_device=CANONICAL_INFERENCE_DEVICE,
            aim_trail=CANONICAL_AIM_TRAIL,
            reaction_delay=CANONICAL_REACTION_DELAY,
        ),
        250: EvalRow(
            250,
            BASELINE_MEAN_SCORE - 5000,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            seed_split=CANONICAL_SPLIT,
            seed_offset=CANONICAL_SEED_OFFSET,
            seed_count=SEEDS_PER_SPLIT,
            frame_skip=CANONICAL_FRAME_SKIP,
            max_ticks=CANONICAL_MAX_TICKS,
            inference_device=CANONICAL_INFERENCE_DEVICE,
            aim_trail=CANONICAL_AIM_TRAIL,
            reaction_delay=CANONICAL_REACTION_DELAY,
        ),
    }
    note = checkpoint_note(list_checkpoints(tmp_path), evals)
    assert f"scored {BASELINE_MEAN_SCORE + 1964:,.0f}" in note
    assert "beats HIGH by 1,964" in note
    assert f"{BASELINE_MEAN_SCORE - 5000:,.0f}" not in note


def test_validation_checkpoint_note_never_claims_a_baseline_delta(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "policy-00200.pt", 2000)
    row = EvalRow(
        200,
        BASELINE_MEAN_SCORE + 10_000,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        seed_split=VALIDATION_SPLIT,
        seed_offset=VALIDATION_SEED_OFFSET,
        seed_count=SEEDS_PER_SPLIT,
        frame_skip=CANONICAL_FRAME_SKIP,
        max_ticks=CANONICAL_MAX_TICKS,
        inference_device=CANONICAL_INFERENCE_DEVICE,
        aim_trail=CANONICAL_AIM_TRAIL,
        reaction_delay=CANONICAL_REACTION_DELAY,
    )
    note = checkpoint_note(list_checkpoints(tmp_path), {200: row})
    assert f"validation score {row.mean_score:,.0f}" in note
    assert "baseline" not in note
    # It is placed on a ladder — the one its own block was measured on. The two
    # ladders' top rungs differ, so the delta names which one was used, and that
    # is exactly what makes the wrong ladder a lie worth catching.
    over_validation = row.mean_score - VALIDATION_LADDER.rungs[-1].mean_score
    over_canonical = row.mean_score - CANONICAL_LADDER.rungs[-1].mean_score
    assert f"beats HIGH by {over_validation:,.0f}" in note
    assert f"beats HIGH by {over_canonical:,.0f}" not in note, (
        "a validation score got the canonical ladder"
    )


def _eval_row(update: int, split: str, offset: int, device: str) -> EvalRow:
    """A protocol-complete eval row, at the cadence and cap both blocks use."""
    return EvalRow(
        update,
        50_000.0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        seed_split=split,
        seed_offset=offset,
        seed_count=SEEDS_PER_SPLIT,
        frame_skip=CANONICAL_FRAME_SKIP,
        max_ticks=CANONICAL_MAX_TICKS,
        inference_device=device,
    )


def test_the_ladder_is_three_ascending_rungs_topped_by_the_published_baseline() -> None:
    # What the score chart draws. Ascending because the chart styles them in
    # that order — faintest first — and topped by the published number because
    # everything else in the project quotes it as *the* baseline.
    lines = baseline_lines(CANONICAL_LADDER)
    assert [value for value, _ in lines] == sorted(value for value, _ in lines)
    assert [value for value, _ in lines] == [rung.mean_score for rung in CANONICAL_LADDER.rungs]
    assert lines[-1][0] == BASELINE_MEAN_SCORE
    # Built from the ladder rather than written out: these numbers move whenever
    # the handicap is retuned, and a test that pinned them would have to be
    # edited every time without ever having checked anything but itself.
    assert [label for _, label in lines] == [
        f"scripted {rung.skill} {rung.mean_score:,.0f} · canonical"
        for rung in CANONICAL_LADDER.rungs
    ]


def test_every_line_says_which_block_it_was_measured_on() -> None:
    # The two ladders are a few hundred points apart, so a legend that named
    # only the skill would let a validation line be read as the published
    # benchmark — the one confusion this whole protocol machinery exists to stop.
    labels = [label for _, label in baseline_lines(VALIDATION_LADDER)]
    assert labels == [
        f"scripted {rung.skill} {rung.mean_score:,.0f} · validation"
        for rung in VALIDATION_LADDER.rungs
    ]
    assert baseline_lines(NO_LADDER) == ()


def test_the_ladder_note_says_what_is_beaten_and_what_is_left() -> None:
    low, medium, high = (rung.mean_score for rung in CANONICAL_LADDER.rungs)

    # Below the first rung there is nothing to celebrate, only a distance.
    assert ladder_note(low - 1_585.46875, CANONICAL_LADDER) == "1,585 to LOW"
    # In the middle, both halves: a run 2,000 short of MEDIUM has still learned
    # something, and "behind the baseline" would have said only that it had not.
    assert ladder_note(medium - 2_000, CANONICAL_LADDER) == "beats LOW · 2,000 to MEDIUM"
    assert ladder_note(medium, CANONICAL_LADDER) == f"beats MEDIUM · {high - medium:,.0f} to HIGH"
    # And past the top, the one claim the project makes.
    assert ladder_note(high + 1_457.65625, CANONICAL_LADDER) == "beats HIGH by 1,458"
    assert ladder_note(high, CANONICAL_LADDER) == "beats HIGH by 0"


def test_a_score_is_measured_against_its_own_block_and_no_other() -> None:
    # A score between the two blocks' top rungs beats HIGH on one and not on the
    # other. Reading it against the wrong ladder is not a rounding difference; it
    # is the opposite verdict, which is why the ladder comes from the row.
    lower, higher = sorted(
        ladder.rungs[-1].mean_score for ladder in (VALIDATION_LADDER, CANONICAL_LADDER)
    )
    between = (lower + higher) / 2
    beaten = (
        VALIDATION_LADDER if VALIDATION_LADDER.rungs[-1].mean_score == lower else CANONICAL_LADDER
    )
    not_beaten = CANONICAL_LADDER if beaten is VALIDATION_LADDER else VALIDATION_LADDER
    assert ladder_note(between, beaten).startswith("beats HIGH by")
    assert "to HIGH" in ladder_note(between, not_beaten)


def test_a_run_scoring_itself_on_validation_gets_the_validation_ladder() -> None:
    # What the console draws for the hours a run is actually training. The
    # backend is not part of it: the scripted agent has none, and pinning CPU
    # would blank the ladder for every GPU run.
    rows = [
        _eval_row(update, VALIDATION_SPLIT, VALIDATION_SEED_OFFSET, "cuda") for update in (10, 20)
    ]
    assert shared_ladder(rows) == VALIDATION_LADDER
    assert row_ladder(rows[0]) == VALIDATION_LADDER


def test_no_ladder_spans_two_blocks_or_an_unmeasured_protocol() -> None:
    validation = _eval_row(10, VALIDATION_SPLIT, VALIDATION_SEED_OFFSET, "cuda")
    canonical = _eval_row(20, CANONICAL_SPLIT, CANONICAL_SEED_OFFSET, CANONICAL_INFERENCE_DEVICE)
    assert shared_ladder([validation, canonical]) == NO_LADDER
    assert shared_ladder([]) == NO_LADDER
    # A cadence nothing was measured at, and a row from before protocols were
    # recorded at all.
    assert row_ladder(EvalRow(**{**validation.__dict__, "frame_skip": 1})) == NO_LADDER
    assert row_ladder(EvalRow(10, 50_000, None, None, None, None, None, None, None)) == NO_LADDER


def test_run_scores_only_compare_when_every_protocol_field_matches() -> None:
    validation = EvalRow(
        50,
        10_000,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        seed_split=VALIDATION_SPLIT,
        seed_offset=VALIDATION_SEED_OFFSET,
        seed_count=SEEDS_PER_SPLIT,
        frame_skip=4,
        max_ticks=120_000,
        inference_device="cpu",
    )
    same = EvalRow(**{**validation.__dict__, "mean_score": 11_000})
    different_cadence = EvalRow(**{**validation.__dict__, "frame_skip": 1})
    legacy = EvalRow(50, 10_000, None, None, None, None, None, None, None)

    assert matching_eval_protocol(validation, same)
    assert not matching_eval_protocol(validation, different_cadence)
    assert not matching_eval_protocol(legacy, legacy)
    assert same_eval_series(validation, same)
    assert not same_eval_series(validation, different_cadence)
    assert same_eval_series(legacy, legacy)
    assert not same_eval_series(validation, legacy)
    assert eval_protocol_label(validation) == "validation"
    assert eval_protocol_label(legacy) == "protocol unknown"


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


# ---- what the numbers say ----------------------------------------------------
# The tiles' peak line and the charts' footnote: the console's own arithmetic,
# tested here rather than read off a screenshot.


def test_a_peak_is_the_best_value_and_the_update_it_was_on() -> None:
    peak = Peak()
    assert peak.offer(50, 91_000.0)
    assert not peak.offer(100, 88_400.0)  # a run regresses; the peak does not move
    assert peak.offer(150, 118_900.0)
    assert (peak.value, peak.update) == (118_900.0, 150)


def test_a_missing_measurement_is_never_a_peak() -> None:
    # The trainer writes nan for the mean return until the first episodes finish,
    # and a gap in the curve is not a high point in it.
    peak = Peak()
    assert not peak.offer(1, None)
    assert peak_note(peak, "{:,.1f}") == ""

    peak.offer(2, 4.87)
    assert not peak.offer(3, None)
    assert (peak.value, peak.update) == (4.87, 2)


def test_a_peak_is_forgotten_when_another_run_writes_into_the_file() -> None:
    peak = Peak()
    peak.offer(400, 128_900.0)
    peak.clear()
    assert peak_note(peak, "{:,.0f}") == ""


def test_the_peak_note_names_the_value_and_its_update() -> None:
    peak = Peak()
    peak.offer(400, 128_900.0)
    assert peak_note(peak, "{:,.0f}") == "peak 128,900 · update 400"


def test_a_curve_with_no_points_has_nothing_to_say() -> None:
    assert curve_note([], "%.2f") == ""


def test_the_curve_note_averages_the_recent_half_and_names_the_window() -> None:
    # Four points: the window is the last two, and it says so — these charts are
    # sampled at different rates, so "the last 50" is not a fixed span of a run.
    assert curve_note([1.0, 3.0, 5.0, 7.0], "%.2f") == "μ2 6.00 ±1.00 · Δ +4.00"


def test_the_curve_note_window_stops_growing_at_the_cap() -> None:
    note = curve_note([1.0] * (4 * RECENT_POINTS), "%.1f")
    assert note.startswith(f"μ{RECENT_POINTS} 1.0 ±0.0")


def test_the_curve_note_signs_a_fall_as_well_as_a_rise() -> None:
    assert curve_note([9.0, 9.0, 4.0, 4.0], "%.1f").endswith("Δ -5.0")


def test_a_curve_too_short_to_compare_two_windows_shows_no_change() -> None:
    assert curve_note([4.87], "%.2f") == "μ1 4.87 ±0.00"


def test_the_readout_names_the_point_it_snapped_to() -> None:
    assert readout_note(812, 4.8712, "%.2f") == "update 812 · 4.87"


def test_the_readout_carries_the_compared_run_when_there_is_one() -> None:
    assert readout_note(812, 4.87, "%.2f", "runs-2", 4.51) == "update 812 · 4.87 · runs-2 4.51"
    # Named but with nothing at that update: the other run says nothing, and the
    # readout does not invent a value for it.
    assert readout_note(812, 4.87, "%.2f", "runs-2", None) == "update 812 · 4.87"


def test_a_rate_reads_in_the_unit_the_trainer_prints_it_in() -> None:
    # The one number that answers "is the accelerator doing anything?". 42k is a
    # saturated 5090 on the relational architecture; 446k is the flat one.
    assert human_rate(38_930.6) == "39k steps/s"
    assert human_rate(446_000) == "446k steps/s"
    assert human_rate(1_250_000) == "1.2M steps/s"
    assert human_rate(812) == "812 steps/s"
    # Nothing to say rather than "0 steps/s", which reads as a stalled run.
    assert human_rate(None) == ""
    assert human_rate(0.0) == ""


def test_a_duration_gets_coarser_the_longer_it_is() -> None:
    # Nobody plans around the minutes of a thirty-hour run.
    assert human_duration(45) == "45 s"
    assert human_duration(117) == "2 min"
    assert human_duration(6_241) == "1 h 44 min"  # `good`, measured
    assert human_duration(99_864) == "1 d 3 h"  # `best`, measured
    assert human_duration(7_200) == "2 h"  # exact hours drop the minutes
    assert human_duration(-5) == "0 s"


def test_the_horizon_comes_from_the_run_that_wrote_it(tmp_path: Path) -> None:
    assert planned_updates(tmp_path) is None  # no config.json at all

    (tmp_path / "config.json").write_text(
        json.dumps({"train": {"updates": 4000, "resume": None}}), encoding="utf-8"
    )
    assert planned_updates(tmp_path) == 4000

    # A resumed run's `updates` counts *additional* updates, and the iteration it
    # continued from lives in a checkpoint this module cannot open. Half an
    # answer would put a confident "4 h left" on a run with a day to go.
    (tmp_path / "config.json").write_text(
        json.dumps({"train": {"updates": 4000, "resume": "checkpoints/policy-final.pt"}}),
        encoding="utf-8",
    )
    assert planned_updates(tmp_path) is None

    for broken in ("", "not json", "[]", '{"train": 7}', '{"train": {"updates": 0}}'):
        (tmp_path / "config.json").write_text(broken, encoding="utf-8")
        assert planned_updates(tmp_path) is None
