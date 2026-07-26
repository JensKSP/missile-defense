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
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

#: Files a run writes, relative to its output directory (``--out-dir``).
METRICS_NAME = "metrics.csv"
EVALS_NAME = "evals.csv"
RECORDING_SUFFIX = ".mdr"
CHECKPOINTS_NAME = "checkpoints"
CHECKPOINT_SUFFIX = ".pt"

#: The scripted agent's mean score over the canonical seeds (docs/ROADMAP.md, M4)
#: — the line the console draws across the score curve. ``md.train`` keeps its own
#: copy of this number rather than importing it, because the console must not be a
#: dependency of the trainer and the trainer must not be one of the console.
BASELINE_MEAN_SCORE = 18_036.0

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
    """One line of ``evals.csv`` — the policy scored on the canonical seeds.

    This is the only number in a run that is directly comparable to the scripted
    baseline's 18,036: same seeds, same C++ ``summarize``, greedy play. The
    training return in ``metrics.csv`` is shaped, scaled and undiscounted, so it
    lives in units of its own.
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


class CsvTail(Generic[T]):
    """Yields the rows appended to a CSV since the last :meth:`poll`.

    Rows are handed to ``parse``, which returns ``None`` for anything it does not
    recognise — that is also how header lines are skipped. Columns are matched by
    *name*, so a run that gains a column is read correctly by an older console and
    vice versa.
    """

    def __init__(self, path: Path, parse: Callable[[Mapping[str, str]], T | None]) -> None:
        self._path = path
        self._parse = parse
        self._offset = 0
        self._fragment = b""
        self._fields: list[str] = []

    @property
    def path(self) -> Path:
        return self._path

    def poll(self) -> Batch[T]:
        try:
            size = self._path.stat().st_size
        except OSError:
            # Not there yet, or gone. A run that has not started is the normal
            # case, so it is an empty batch and not an error; a file that has
            # *vanished* invalidates whatever was drawn from it.
            if self._offset or self._fragment:
                self._rewind()
                return Batch((), restarted=True)
            return Batch((), restarted=False)

        restarted = False
        if size < self._offset:
            self._rewind()
            restarted = True

        with self._path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)

        # Bytes, not text: a chunk boundary can fall inside a line, and decoding
        # only whole lines keeps a split multi-byte character from ever happening.
        head, newline, self._fragment = (self._fragment + chunk).rpartition(b"\n")
        if not newline:
            return Batch((), restarted)

        rows: list[T] = []
        for line in head.split(b"\n"):
            row = self._row(line.decode("utf-8", errors="replace"))
            if row is not None:
                rows.append(row)
        return Batch(tuple(rows), restarted)

    def _rewind(self) -> None:
        self._offset = 0
        self._fragment = b""
        self._fields = []

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
    )


def metrics_tail(run_dir: Path) -> CsvTail[MetricRow]:
    """Tail ``<run_dir>/metrics.csv`` — one row per update."""
    return CsvTail(run_dir / METRICS_NAME, _metric_row)


def evals_tail(run_dir: Path) -> CsvTail[EvalRow]:
    """Tail ``<run_dir>/evals.csv`` — one row per scored evaluation."""
    return CsvTail(run_dir / EVALS_NAME, _eval_row)


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
        delta = row.mean_score - BASELINE_MEAN_SCORE
        verdict = "ahead of" if delta > 0 else "behind"
        parts.append(f"scored {row.mean_score:,.0f}, {abs(delta):,.0f} {verdict} baseline")
    return " · ".join(parts)
