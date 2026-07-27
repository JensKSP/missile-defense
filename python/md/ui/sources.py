# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What a training run leaves on disk, read incrementally. No Qt in here.

The console never talks to the trainer: the artifacts *are* the interface
(docs/ROADMAP.md, M8), which is what lets it attach to a run started from a
terminal and lets a crashed window leave the run untouched.

Reading a file while another process appends to it has three failure modes, and
they are the reason this module exists rather than a `read_text().splitlines()`:

* **Re-reading the whole file each refresh.** After 5,000 updates that is 5,000
  lines re-parsed every second, forever. *So:* remember the byte offset reading
  stopped at and resume from there.
* **The half-written line.** The trainer may be mid-write, leaving
  ``247,32768,4.8`` with no newline because the rest has not been flushed.
  Parsed as a row that is a *wrong* value, not merely a slow one — this is the
  one that actually bites. *So:* only complete lines are parsed, and the trailing
  fragment is carried into the next poll.
* **The file was replaced.** A fresh run in the same directory leaves the saved
  offset past the end, and reading resumes there — returning nothing, forever.
  *So:* a file shorter than the offset is a new file, and the caller is told to
  throw its curves away.

Being Qt-free, all three are pytest-able against a CSV written a line at a time.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Generic, TypeVar, cast

from ..benchmark import (
    CANONICAL_BASELINE_MEAN_SCORE,
    NO_LADDER,
    VALIDATION_SPLIT,
    Ladder,
    canonical_baseline_comparable,
    ladder_for,
    ladder_standing,
)

#: Files a run writes, relative to its output directory (``--out-dir``).
METRICS_NAME = "metrics.csv"
#: What the trainer was started with, written once at start-up.
CONFIG_NAME = "config.json"
EVALS_NAME = "evals.csv"
RECORDING_SUFFIX = ".mdr"
CHECKPOINTS_NAME = "checkpoints"
CHECKPOINT_SUFFIX = ".pt"
#: The trainer's own copy of what it printed (:mod:`md.runlog`), which is how a
#: run this console never started still gets a log pane.
LOG_NAME = "train.log"

#: Compatibility alias; benchmark.py owns the value and the protocol it belongs to.
BASELINE_MEAN_SCORE = CANONICAL_BASELINE_MEAN_SCORE

T = TypeVar("T")


# ---- rows -------------------------------------------------------------------


@dataclass(frozen=True)
class MetricRow:
    """One line of ``metrics.csv`` — one training update.

    Every measurement is optional because the trainer writes ``nan`` for the mean
    return until the first episodes finish (episodes are thousands of ticks long),
    and because a column may simply not exist in a CSV written by an older run.
    A missing number is drawn as a gap rather than as a zero.
    """

    update: int
    samples: int
    mean_return: float | None
    entropy: float | None
    policy_loss: float | None
    value_loss: float | None
    clip_fraction: float | None
    steps_per_second: float | None


@dataclass(frozen=True)
class EvalRow:
    """One line of ``evals.csv`` and the protocol that produced its score.

    Old files have no protocol fields. They remain readable, but the console
    cannot honestly compare them with a baseline or another run until the seed
    split, offset and count, cadence, cap and inference backend are known.
    """

    update: int
    mean_score: float
    min_score: float | None
    max_score: float | None
    mean_wave: float | None
    mean_cities_left: float | None
    mean_accuracy: float | None
    survived: int | None
    episodes: int | None
    seed_split: str | None = None
    seed_offset: int | None = None
    seed_count: int | None = None
    frame_skip: int | None = None
    max_ticks: int | None = None
    inference_device: str | None = None
    #: The human handicap this score was earned under. `None` for every row
    #: written before it existed — and `None` is not "probably canonical": those
    #: runs faced an agent that never mis-clicked, so their numbers answer a
    #: different question and must not be ranked beside these.
    aim_trail: float | None = None
    reaction_delay: int | None = None
    # The full per-episode statistics (the plan's Task 11). Every one is
    # optional for the same reason the rest are: a run written by an older
    # trainer simply has no such column, and the console has to keep reading it.
    # They are what the analysis view draws — how long the policy survived, how
    # much damage it took, and how the ammunition was spent.
    mean_ticks: float | None = None
    mean_waves_cleared: float | None = None
    mean_cities_lost: float | None = None
    mean_bases_left: float | None = None
    mean_bases_lost: float | None = None
    mean_ammo_left: float | None = None
    mean_bonus_cities: float | None = None
    mean_mirv_splits: float | None = None
    mean_shots: float | None = None
    mean_kills: float | None = None
    mean_hits: float | None = None
    mean_hit_rate: float | None = None
    #: Shots binned by how many threats their blast killed, summed over the whole
    #: seed set. `shots_0kill` is ammunition that hit nothing at all.
    shots_0kill: int | None = None
    shots_1kill: int | None = None
    shots_2kill: int | None = None
    shots_3kill: int | None = None
    shots_4plus: int | None = None


def is_canonical_benchmark(row: EvalRow) -> bool:
    """Whether ``row`` can be compared with the published scripted baseline."""

    return canonical_baseline_comparable(
        seed_split=row.seed_split,
        seed_offset=row.seed_offset,
        seed_count=row.seed_count,
        frame_skip=row.frame_skip,
        max_ticks=row.max_ticks,
        inference_device=row.inference_device,
        aim_trail=row.aim_trail,
        reaction_delay=row.reaction_delay,
    )


def matching_eval_protocol(left: EvalRow, right: EvalRow) -> bool:
    """Whether two scores were produced under the same fully known protocol."""

    left_protocol = eval_protocol(left)
    right_protocol = eval_protocol(right)
    return all(value is not None for value in left_protocol) and left_protocol == right_protocol


def same_eval_series(left: EvalRow, right: EvalRow) -> bool:
    """Whether adjacent rows may share a line, including legacy-to-legacy."""

    left_protocol = eval_protocol(left)
    right_protocol = eval_protocol(right)
    both_legacy = all(value is None for value in (*left_protocol, *right_protocol))
    return both_legacy or (
        all(value is not None for value in left_protocol) and left_protocol == right_protocol
    )


def eval_protocol(row: EvalRow) -> tuple[str | int | None, ...]:
    """The fields that define score comparability, in stable CSV order."""

    return (
        row.seed_split,
        row.seed_offset,
        row.seed_count,
        row.frame_skip,
        row.max_ticks,
        row.inference_device,
    )


def eval_protocol_label(row: EvalRow) -> str:
    """Compact, honest label for a score tile or checkpoint note."""

    if is_canonical_benchmark(row):
        return "held-out canonical"
    if row.seed_split == VALIDATION_SPLIT:
        return "validation"
    if row.seed_split:
        return f"{row.seed_split} (nonstandard protocol)"
    return "protocol unknown"


def eval_protocol_note(row: EvalRow) -> str:
    """Protocol label plus the fields needed to reproduce the measurement."""

    parts = [eval_protocol_label(row)]
    if row.seed_offset is not None:
        parts.append(f"seed offset {row.seed_offset}")
    if row.seed_count is not None:
        parts.append(f"{row.seed_count} seeds")
    if row.frame_skip is not None:
        parts.append(f"frame skip {row.frame_skip}")
    if row.max_ticks is not None:
        parts.append(f"cap {row.max_ticks:,}")
    if row.inference_device is not None:
        parts.append(row.inference_device)
    return " · ".join(parts)


# ---- the scripted ladder ----------------------------------------------------


def row_ladder(row: EvalRow) -> Ladder:
    """The scripted ladder ``row``'s score may be read against, if any."""

    return ladder_for(
        seed_split=row.seed_split,
        seed_offset=row.seed_offset,
        seed_count=row.seed_count,
        frame_skip=row.frame_skip,
        max_ticks=row.max_ticks,
        inference_device=row.inference_device,
        aim_trail=row.aim_trail,
        reaction_delay=row.reaction_delay,
    )


def shared_ladder(rows: Sequence[EvalRow]) -> Ladder:
    """The one ladder every row in ``rows`` may be read against.

    Nothing at all when they disagree. The score curve is a single line and
    takes a single set of reference lines, so a segment that mixes blocks has no
    honest ladder — and the caller must not fall back on the newest row's,
    which would silently re-label every older point.
    """

    ladders = {row_ladder(row) for row in rows}
    return ladders.pop() if len(ladders) == 1 else NO_LADDER


def baseline_lines(ladder: Ladder) -> tuple[tuple[float, str], ...]:
    """``ladder`` as the score chart wants it: ``(value, legend label)``.

    Ascending, which is the order the chart styles them in. Each label carries
    the number *and* the block: the number because a legend read from across the
    room should not need the axis to say what "medium" is worth, and the block
    because the two ladders differ by a few hundred points and a line that did
    not say which one it was would be the easiest possible thing to misread.
    """

    return tuple(
        (rung.mean_score, f"scripted {rung.skill} {rung.mean_score:,.0f} · {ladder.block}")
        for rung in ladder.rungs
    )


def ladder_note(score: float, ladder: Ladder) -> str:
    """Where ``score`` stands on ``ladder``, in one phrase.

    Always two facts where there are two to give — what it has beaten and what
    is left — because "behind the baseline" is the same sentence for a policy
    that has learned nothing and for one that is 2,000 points short of an expert
    agent, and those are not the same run.

    ``ladder`` must be the one measured on this score's own seed block
    (:func:`row_ladder`); against any other the arithmetic is invented.
    """

    cleared, remaining = ladder_standing(score, ladder)
    if remaining is not None:
        to_next = f"{remaining.mean_score - score:,.0f} to {remaining.label}"
        return f"beats {cleared.label} · {to_next}" if cleared is not None else to_next
    if cleared is None:  # the empty ladder has nothing to say about anything
        return ""
    return f"beats {cleared.label} by {score - cleared.mean_score:,.0f}"


@dataclass(frozen=True)
class Recording:
    """A watchable episode: ``runs/update-00025.mdr``."""

    path: Path
    update: int | None
    size: int
    modified: float

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class Checkpoint:
    """A saved policy: ``runs/checkpoints/policy-00800.pt``.

    ``iteration`` is ``None`` for ``policy-final.pt``, which is deliberately not
    numbered — it is whatever the run ended on, and inventing a number for it
    here would be guessing at one the file does not carry.
    """

    path: Path
    iteration: int | None
    size: int
    modified: float

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class Batch(Generic[T]):
    """Rows appended since the last poll, and whether the file restarted."""

    rows: tuple[T, ...]
    restarted: bool


# ---- the tail ---------------------------------------------------------------


class LineTail:
    """Yields the complete lines appended to a text file since the last poll.

    The three failure modes in this module's docstring all live here, so a
    reader of any of a run's line-oriented files gets them handled once —
    ``metrics.csv``, ``evals.csv`` and the trainer's own ``train.log`` differ
    only in what a line *means*.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._fragment = b""
        self._identity: tuple[int, int] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def poll(self) -> Batch[str]:
        try:
            stat = self._path.stat()
            size = stat.st_size
        except OSError:
            # Not there yet, or gone. A run that has not started is the normal
            # case, so it is an empty batch and not an error; a file that has
            # *vanished* invalidates whatever was drawn from it.
            if self._offset or self._fragment:
                self.rewind()
                return Batch((), restarted=True)
            return Batch((), restarted=False)

        identity = (stat.st_dev, stat.st_ino)
        restarted = self._identity is not None and identity != self._identity
        if restarted:
            self.rewind()
        if size < self._offset:
            self.rewind()
            restarted = True
        self._identity = identity

        with self._path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)

        # Bytes, not text: a chunk boundary can fall inside a line, and decoding
        # only whole lines keeps a split multi-byte character from ever happening.
        head, newline, self._fragment = (self._fragment + chunk).rpartition(b"\n")
        if not newline:
            return Batch((), restarted)
        # The trailing CR of a CRLF file goes with the newline it belongs to.
        # This project's own writers use `newline=""` so they never produce one,
        # but a run directory synced off another machine may have been through a
        # tool that did — and a stray CR is invisible until it is not.
        lines = tuple(
            line.decode("utf-8", errors="replace").removesuffix("\r") for line in head.split(b"\n")
        )
        return Batch(lines, restarted)

    def rewind(self) -> None:
        self._offset = 0
        self._fragment = b""
        self._identity = None


def log_tail(run_dir: Path) -> LineTail:
    """Tail ``<run_dir>/train.log`` — what the run has printed.

    The trainer writes this itself (:mod:`md.runlog`), which is what gives the
    console a log pane for a run it did not start. Its own child's stdout comes
    down a pipe instead; this is for every other way a run can be launched.
    """
    return LineTail(run_dir / LOG_NAME)


class CsvTail(Generic[T]):
    """Yields the rows appended to a CSV since the last :meth:`poll`.

    Rows are handed to ``parse``, which returns ``None`` for anything it does not
    recognise — that is also how header lines are skipped. Columns are matched by
    *name*, so a run that gains a column is read correctly by an older console and
    vice versa.
    """

    def __init__(self, path: Path, parse: Callable[[Mapping[str, str]], T | None]) -> None:
        self._lines = LineTail(path)
        self._parse = parse
        self._fields: list[str] = []

    @property
    def path(self) -> Path:
        return self._lines.path

    def rewind(self) -> None:
        """Read the CSV from its header again on the next :meth:`poll`.

        The dashboard uses this when the primary run changes evaluation
        protocol. Rows that were irrelevant while the tail advanced to EOF can
        become the exact rows needed for the new protocol.
        """

        self._lines.rewind()
        self._fields = []

    def poll(self) -> Batch[T]:
        batch = self._lines.poll()
        if batch.restarted:
            # A different run writing into the same file may have different
            # columns, so the remembered header goes with the offset.
            self._fields = []
        rows: list[T] = []
        for line in batch.rows:
            row = self._row(line)
            if row is not None:
                rows.append(row)
        return Batch(tuple(rows), batch.restarted)

    def _row(self, line: str) -> T | None:
        parsed: list[list[str]] = list(csv.reader([line]))
        cells = parsed[0] if parsed else []
        if not cells:
            return None
        if _is_header(cells):
            # Also catches a header appended mid-file, which is what a schema
            # change looks like to a reader that started before it.
            self._fields = cells
            return None
        return self._parse(dict(zip(self._fields, cells, strict=False)))


def _is_header(cells: Sequence[str]) -> bool:
    """Header lines start with a column name; data lines start with the update."""
    try:
        float(cells[0])
    except ValueError:
        return True
    return False


# ---- parsing ----------------------------------------------------------------


def _number(row: Mapping[str, str], key: str) -> float | None:
    """A finite float, or ``None`` when the column is absent, blank or ``nan``."""
    try:
        value = float(row[key])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _count(row: Mapping[str, str], key: str) -> int | None:
    value = _number(row, key)
    return None if value is None else int(value)


def _text(row: Mapping[str, str], key: str) -> str | None:
    value = row.get(key, "").strip()
    return value or None


def _metric_row(row: Mapping[str, str]) -> MetricRow | None:
    update = _count(row, "update")
    if update is None:
        return None
    return MetricRow(
        update=update,
        samples=_count(row, "samples") or 0,
        mean_return=_number(row, "return"),
        entropy=_number(row, "entropy"),
        policy_loss=_number(row, "policy_loss"),
        value_loss=_number(row, "value_loss"),
        clip_fraction=_number(row, "clip_fraction"),
        steps_per_second=_number(row, "steps_per_second"),
    )


def _eval_row(row: Mapping[str, str]) -> EvalRow | None:
    update = _count(row, "update")
    score = _number(row, "mean_score")
    if update is None or score is None:
        return None
    return EvalRow(
        update=update,
        mean_score=score,
        min_score=_number(row, "min_score"),
        max_score=_number(row, "max_score"),
        mean_wave=_number(row, "mean_wave"),
        mean_cities_left=_number(row, "mean_cities_left"),
        mean_accuracy=_number(row, "mean_accuracy"),
        survived=_count(row, "survived"),
        episodes=_count(row, "episodes"),
        seed_split=_text(row, "seed_split"),
        seed_offset=_count(row, "seed_offset"),
        seed_count=_count(row, "seed_count"),
        frame_skip=_count(row, "frame_skip"),
        max_ticks=_count(row, "max_ticks"),
        inference_device=_text(row, "inference_device"),
        mean_ticks=_number(row, "mean_ticks"),
        mean_waves_cleared=_number(row, "mean_waves_cleared"),
        mean_cities_lost=_number(row, "mean_cities_lost"),
        mean_bases_left=_number(row, "mean_bases_left"),
        mean_bases_lost=_number(row, "mean_bases_lost"),
        mean_ammo_left=_number(row, "mean_ammo_left"),
        mean_bonus_cities=_number(row, "mean_bonus_cities"),
        mean_mirv_splits=_number(row, "mean_mirv_splits"),
        mean_shots=_number(row, "mean_shots"),
        mean_kills=_number(row, "mean_kills"),
        mean_hits=_number(row, "mean_hits"),
        mean_hit_rate=_number(row, "mean_hit_rate"),
        shots_0kill=_count(row, "shots_0kill"),
        shots_1kill=_count(row, "shots_1kill"),
        shots_2kill=_count(row, "shots_2kill"),
        shots_3kill=_count(row, "shots_3kill"),
        shots_4plus=_count(row, "shots_4plus"),
    )


def metrics_tail(run_dir: Path) -> CsvTail[MetricRow]:
    """Tail ``<run_dir>/metrics.csv`` — one row per update."""
    return CsvTail(run_dir / METRICS_NAME, _metric_row)


def evals_tail(run_dir: Path) -> CsvTail[EvalRow]:
    """Tail ``<run_dir>/evals.csv`` — one row per scored evaluation."""
    return CsvTail(run_dir / EVALS_NAME, _eval_row)


# ---- peaks ------------------------------------------------------------------


class Peak:
    """The highest a measurement has been, and the update it was there.

    A run is not monotone. PPO peaks and then regresses — a moving target
    destabilises the critic, entropy collapses — so the newest number is not the
    best one, and a tile showing only the newest cannot answer "has this run
    already been better than it is now?". ``md.train`` keeps exactly this for the
    eval score, because it decides which policy ``policy-best.pt`` is; the
    console keeps it for the numbers it puts on screen.

    Fed each row as the tail hands it over rather than computed from a curve: the
    rows arrive once, and a widget is the wrong place to keep a fact about a run.
    """

    def __init__(self) -> None:
        self.value: float | None = None
        self.update: int | None = None

    def offer(self, update: int, value: float | None) -> bool:
        """Take ``value`` if it beats the peak. True when the peak moved.

        A missing measurement is never a peak: the trainer writes ``nan`` for the
        mean return until the first episodes finish, and a gap is not a high.
        """
        if value is None or (self.value is not None and value <= self.value):
            return False
        self.value = value
        self.update = update
        return True

    def clear(self) -> None:
        """Forget it — a different run is writing into this file now."""
        self.value = None
        self.update = None


class Latest:
    """The newest measurement that actually exists, and when it was taken.

    Not the same question as :class:`Peak`, and not the same as "the newest row".
    The trainer writes ``nan`` for the mean return until a run's first episodes
    finish — thousands of ticks — so straight after a start or a ``--resume``
    the newest row genuinely has no return in it. A tile fed from that row alone
    shows a dash, which claims no measurement exists when one does.

    So this keeps the last real one and the update it came from. Showing it
    without saying *when* would be the opposite error: a stale number presented
    as current. :meth:`note` is what stops that, and it is why the update is
    kept here rather than only the value.
    """

    def __init__(self) -> None:
        self.value: float | None = None
        self.update: int | None = None

    def offer(self, update: int, value: float | None) -> None:
        if value is not None:
            self.value = value
            self.update = update

    def note(self, meaning: str, *, current_update: int | None = None) -> str:
        """``shaped, scaled — not a score`` while current; adds the age when not."""
        if self.value is None or self.update is None:
            return meaning
        if current_update is None or current_update == self.update:
            return meaning
        return f"{meaning} · last at update {self.update:,}"

    def clear(self) -> None:
        self.value = None
        self.update = None


# ---- recordings -------------------------------------------------------------


def list_recordings(run_dir: Path) -> list[Recording]:
    """Episodes in ``run_dir``, newest first.

    Not recursive, and by modification time rather than by name: while a run is
    going the interesting episode is the one just written, whatever it is called.
    """
    found: list[Recording] = []
    try:
        entries: Iterable[Path] = list(run_dir.iterdir())
    except OSError:
        return []
    for path in entries:
        if path.suffix != RECORDING_SUFFIX:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue  # deleted between listing and stat'ing; it simply is not there
        found.append(
            Recording(
                path=path,
                update=_update_in(path.stem),
                size=stat.st_size,
                modified=stat.st_mtime,
            )
        )
    found.sort(key=lambda r: (r.modified, r.name), reverse=True)
    return found


def _update_in(stem: str) -> int | None:
    """The update number in ``update-00025``, when the name carries one."""
    _, dash, tail = stem.rpartition("-")
    if not dash or not tail.isdigit():
        return None
    return int(tail)


# ---- checkpoints ------------------------------------------------------------


def list_checkpoints(run_dir: Path) -> list[Checkpoint]:
    """Saved policies in ``run_dir/checkpoints``, newest first.

    By modification time, like the recordings and for the same reason: the one
    you want is almost always the last one written, and ``policy-final.pt`` sorts
    to the top by simply being last — which is also where it belongs, being the
    only one guaranteed to exist.
    """
    found: list[Checkpoint] = []
    try:
        entries: Iterable[Path] = list((run_dir / CHECKPOINTS_NAME).iterdir())
    except OSError:
        return []
    for path in entries:
        if path.suffix != CHECKPOINT_SUFFIX:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue  # written and removed between listing and stat'ing
        found.append(
            Checkpoint(
                path=path,
                iteration=_update_in(path.stem),
                size=stat.st_size,
                modified=stat.st_mtime,
            )
        )
    found.sort(key=lambda c: (c.modified, c.name), reverse=True)
    return found


def last_modified(path: Path) -> float | None:
    """Modification time, or ``None`` when the file is not there."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def find_runs(directory: Path) -> list[Path]:
    """Sub-directories of ``directory`` that hold a run, newest first.

    Runs accumulate: an experiment gets its own ``--out-dir`` and the old one is
    kept rather than overwritten. So a directory with no ``metrics.csv`` of its
    own is usually not "nothing here" but "the runs are one level down", and a
    console that cannot tell the difference sends you back to the shell to find
    out which is which.
    """
    try:
        children = [path for path in directory.iterdir() if path.is_dir()]
    except OSError:
        return []
    runs = [path for path in children if (path / METRICS_NAME).exists()]
    runs.sort(key=lambda path: (last_modified(path / METRICS_NAME) or 0.0, path.name), reverse=True)
    return runs


#: Enough to move between experiments without the picker becoming a file browser.
MAX_RUN_CHOICES = 12


def run_choices(run_dir: Path) -> list[Path]:
    """Run directories worth offering beside ``run_dir``, newest first.

    A run directory takes two shapes in practice: ``runs/`` holding one run, and
    ``runs/`` holding a directory per experiment. So the candidates are whatever
    is attached, the runs *inside* it, and the runs *beside* it — which covers
    both without asking anyone to think about which shape they have.

    ``run_dir`` is always in the list even when it holds no run at all: it is
    what the window is showing, and a picker that cannot show the current
    selection is a bug rather than a tidy list.
    """
    candidates = [run_dir.resolve(), *find_runs(run_dir), *find_runs(run_dir.parent)]
    unique: dict[Path, Path] = {}
    for path in candidates:
        unique.setdefault(path.resolve(), path.resolve())
    ordered = sorted(
        unique.values(),
        key=lambda path: (last_modified(path / METRICS_NAME) or 0.0, path.name),
        reverse=True,
    )
    chosen = ordered[:MAX_RUN_CHOICES]
    if run_dir.resolve() not in chosen:  # cut by the cap; it still has to be there
        chosen.append(run_dir.resolve())
    return chosen


def next_run_dir(run_dir: Path) -> Path:
    """The next free ``runs-2``, ``runs-3``… beside ``run_dir``.

    What Reset offers. Starting over means a *new* directory, never emptying the
    old one: the checkpoints of the run you are abandoning are exactly what you
    will want when the new settings turn out to be worse.
    """
    stem, dash, tail = run_dir.name.rpartition("-")
    base = stem if dash and tail.isdigit() else run_dir.name
    number = int(tail) + 1 if dash and tail.isdigit() else 2
    candidate = run_dir.with_name(f"{base}-{number}")
    while candidate.exists():
        number += 1
        candidate = run_dir.with_name(f"{base}-{number}")
    return candidate


# ---- glanceable formatting --------------------------------------------------
# Pure, so the console's most-read text is covered by tests rather than by eye.


def peak_note(peak: Peak, spec: str) -> str:
    """``peak 128,900 · update 400`` — the best a tile's number has been.

    Empty until there is one, because a tile that says "peak —" has spent a line
    to tell you nothing.
    """
    if peak.value is None or peak.update is None:
        return ""
    return f"peak {spec.format(peak.value)} · update {peak.update:,}"


#: The longest window the charts' statistics are taken over. Long enough to see
#: through PPO's per-update noise, short enough to still mean *lately*.
RECENT_POINTS = 50


def curve_note(values: Sequence[float], value_format: str) -> str:
    """``μ50 4.61 ±0.31 · Δ +0.42`` — the statistics in a chart's corner.

    Three facts a curve does not give you by being looked at. Where it is *now*
    net of the noise (μ), how noisy it is (σ, which is what tells you whether a
    rise is real), and whether it is still moving (Δ: this window's mean against
    the one before it). The last is the question a training curve is actually
    asked, and neither the newest point nor the peak can answer it.

    The window is the curve's last half capped at :data:`RECENT_POINTS`, so it
    says something on the tenth point as well as on the ten-thousandth, and it is
    *named* in the text: these charts are sampled at different rates — one point
    per update, one per ``--eval-every`` updates — so "the last 50" is a different
    span of a run on each of them.

    ``value_format`` is the printf format the chart's own axis is labelled with,
    so the digits under the plot and the digits beside it agree.
    """
    if not values:
        return ""
    window = min(RECENT_POINTS, max(len(values) // 2, 1))
    recent = values[-window:]
    mean = fmean(recent)
    parts = [f"μ{window} {value_format % mean} ±{value_format % pstdev(recent)}"]
    before = values[-2 * window : -window]
    if before:
        # Signed explicitly: "0.42" and "-0.42" differ by one character, and which
        # way a run is going is the whole reason this number is here.
        change = mean - fmean(before)
        parts.append(f"Δ {'+' if change >= 0 else '-'}{value_format % abs(change)}")
    return " · ".join(parts)


def readout_note(
    update: float,
    value: float,
    value_format: str,
    compare_name: str = "",
    compare_value: float | None = None,
) -> str:
    """``update 812 · 4.87 · runs-2 4.51`` — the point under the pointer.

    The *nearest recorded point*, never an interpolation between two: a curve is
    a sequence of measurements, and inventing a value between them would be the
    chart making up a number the run never produced.
    """
    text = f"update {update:,.0f} · {value_format % value}"
    if compare_name and compare_value is not None:
        text += f" · {compare_name} {value_format % compare_value}"
    return text


def human_rate(steps_per_second: float | None) -> str:
    """``"39k steps/s"`` — how fast the run is collecting and learning.

    The one number that answers "is the accelerator actually doing anything?",
    which the trainer has printed since M6 and this window did not show at all.
    Read against docs/NVIDIA.md: the relational architecture is GPU-bound at
    around 42k on a 5090, and the flat one runs ten times that.
    """
    if steps_per_second is None or steps_per_second <= 0:
        return ""
    if steps_per_second >= 1_000_000:
        return f"{steps_per_second / 1_000_000:.1f}M steps/s"
    if steps_per_second >= 1_000:
        return f"{steps_per_second / 1_000:.0f}k steps/s"
    return f"{steps_per_second:.0f} steps/s"


def human_duration(seconds: float) -> str:
    """``"2 min"``, ``"1 h 45 min"``, ``"1 d 4 h"`` — how long something has left.

    Coarser the longer it is, because nobody plans around the minutes of a
    thirty-hour run, and two units at most: the point is "go to bed" or "wait for
    it", not a countdown.
    """
    seconds = max(seconds, 0.0)
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = int(minutes // 60)
    if hours < 24:
        remainder = int(minutes - hours * 60)
        return f"{hours} h {remainder} min" if remainder else f"{hours} h"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h" if hours else f"{days} d"


def planned_updates(run_dir: Path) -> int | None:
    """How many updates this run was started to do, from its own `config.json`.

    ``None`` when the file is missing, unreadable, or records a `--resume`: a
    resumed run's `updates` is a count of *additional* updates, and the iteration
    it continued from is in a checkpoint this module deliberately cannot open
    (that would need torch). Half an answer would be worse than none — it would
    put a confident "4 h left" on a run with a day to go.
    """
    try:
        raw: object = json.loads((run_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    section = cast("dict[str, object]", raw).get("train")
    if not isinstance(section, dict):
        return None
    train = cast("dict[str, object]", section)
    if train.get("resume") is not None:
        return None
    updates = train.get("updates")
    return updates if isinstance(updates, int) and updates > 0 else None


def human_age(seconds: float) -> str:
    """``"just now"``, ``"12 s ago"``, ``"4 min ago"``, ``"2 h ago"``, ``"6 d ago"``.

    The whole phrase, because "just now" is the one case that does not take an
    "ago" — and a caller that appends one writes "just now ago". Deliberately
    coarse, too: this answers "is anything still happening", and a ticking
    seconds counter is exactly the flicker the design rules out.
    """
    if seconds < 2:
        return "just now"
    for limit, unit, size in ((60, "s", 1), (3600, "min", 60), (86400, "h", 3600)):
        if seconds < limit:
            return f"{int(seconds // size)} {unit} ago"
    return f"{int(seconds // 86400)} d ago"


def human_size(size: int) -> str:
    """``"812 B"``, ``"79 kB"``, ``"1.2 MB"`` — one number, no decimals below MB."""
    if size < 1000:
        return f"{size} B"
    if size < 1_000_000:
        return f"{size / 1000:.0f} kB"
    return f"{size / 1_000_000:.1f} MB"


#: What the model panel says when a run has saved nothing yet. An empty state
#: names the flag that would change it, like the others.
NO_CHECKPOINTS = (
    "No checkpoint saved yet — the trainer writes one every --checkpoint-every updates."
)


def checkpoint_note(checkpoints: Sequence[Checkpoint], evals: Mapping[int, EvalRow]) -> str:
    """Which policy is on disk, how many there are, and what it scored.

    The score is looked up by the checkpoint's *own* update rather than taken
    from the most recent evaluation. Those are usually the same row and
    occasionally are not, and quietly labelling update 750's score as update
    800's would make the one panel that exists to say "this is the model you
    have" the one that lies about it.
    """
    if not checkpoints:
        return NO_CHECKPOINTS
    newest = checkpoints[0]
    parts = [newest.name, f"{len(checkpoints)} saved"]
    row = evals.get(newest.iteration) if newest.iteration is not None else None
    if row is not None:
        if is_canonical_benchmark(row):
            parts.append(f"scored {row.mean_score:,.0f}")
        else:
            parts.append(f"{eval_protocol_label(row)} score {row.mean_score:,.0f}")
        # Whichever ladder its own block was measured on, and nothing when that
        # block has none: the panel says what the model *is*, so a rung it was
        # never measured against has no business in the sentence.
        ladder = row_ladder(row)
        if ladder:
            parts.append(ladder_note(row.mean_score, ladder))
    return " · ".join(parts)
