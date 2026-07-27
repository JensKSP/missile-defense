# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What runs exist, what they cost, and what they are called. No Qt, no torch.

A run directory accumulates: an experiment gets its own `--out-dir`, the old one
is kept because its checkpoints are exactly what you want when the new settings
turn out worse, and after a fortnight there are eleven of them called `runs-2`
through `runs-12`. This module is what turns that into a list a person can read.

**Naming is a first-class requirement, not a nicety** (docs/ROADMAP.md, M8, asked
for directly). `runs-2` is not an identity anyone can hold in their head while
comparing four agents, and a bare checkpoint path is worse. So every run carries
two names: the directory it lives in, which is **immutable** and is what every
file inside it refers to, and a **display name** a person can change at any time
without moving a single byte. The second lives in `LIBRARY.json`, written by the
console and never by the trainer — which is the whole reason renaming is safe
while a run is going.

`LIBRARY.json` also holds the pins. A pinned recording is one cleanup must not
delete (Task 9), and pinning is a judgement — *this* is the episode where it
learned to wait for clusters — that nothing else in the tree can infer.

Everything here reads; nothing starts, stops or deletes a run. Discovery has to
work on a directory that a trainer is writing into right now, and on one synced
from another machine that never had a trainer at all.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import modelcard
from .control import Control

# `md.ui.sources` and not a copy of it: it is where the knowledge of a run's
# file names and formats already lives, it is Qt-free by test
# (`test_ui_boundary.py`), and a second parser for `evals.csv` would be the one
# that drifts. The layering does read backwards — a service under `md/` reaching
# into `md/ui/` — and the honest fix is to move `sources` down a level, which is
# a rename across files another agent is editing right now. Noted rather than
# done; nothing here depends on the direction being wrong.
from .ui import sources

#: The console's own per-run file: the display name, the pins, the note. Upper
#: case like `TUNING.json`, which is the other file a *person* drives rather
#: than the trainer — the run's own artifacts are all lower case.
LIBRARY_NAME = "LIBRARY.json"

#: How long since the last write before a run counts as no longer going.
#:
#: Only a fallback now, for runs written by a trainer old enough not to leave a
#: `RUNNING` marker. Asking the operating system is both faster and more honest:
#: this threshold lags a finished run by up to ninety seconds, and — the worse
#: half — calls a *slow* run dead. An update that takes longer than this is a
#: perfectly healthy large model on a modest card, and reporting it as stopped
#: invites someone to start a second trainer in the same directory.
LIVE_AFTER_S = 90.0


@dataclass(frozen=True)
class Storage:
    """What a run costs, split by what it would cost to lose.

    Checkpoints are hours of compute; recordings are minutes of playback;
    everything else is text. Cleanup (Task 9) plans against exactly this split,
    so it is computed once here rather than in the widget that shows it.
    """

    checkpoints: int = 0
    recordings: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.checkpoints + self.recordings + self.other


@dataclass(frozen=True)
class Run:
    """One training run, as the library lists it."""

    path: Path
    #: The directory name. Immutable, and what every path inside the run uses.
    run_id: str
    #: Editable, and what a person actually reads. Falls back to the id.
    display_name: str
    #: `live` while something is still writing, `stopped` once it is not,
    #: `empty` for a directory with no `metrics.csv` in it at all.
    state: str
    updates: int
    best_score: float | None
    best_update: int | None
    #: The highest-scoring *evaluated* checkpoint, which is not always the last
    #: one: PPO peaks and then regresses, which is why the trainer keeps a
    #: `policy-best.pt` separate from its final one.
    best_checkpoint: Path | None
    last_activity: float | None
    storage: Storage
    recordings: int
    checkpoints: int
    note: str = ""
    pinned: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """What to show. The display name, or the id when there is none."""
        return self.display_name or self.run_id

    @property
    def live(self) -> bool:
        return self.state == "live"


# ---- the console's own per-run file ------------------------------------------


@dataclass
class Metadata:
    """`LIBRARY.json` — everything about a run that a person, not a trainer, set."""

    display_name: str = ""
    note: str = ""
    pinned: list[str] = field(default_factory=list[str])


def read_metadata(run: Path) -> Metadata:
    """Never raises. A missing or malformed file is simply no metadata.

    A run directory can be anything a user points the console at, including one
    with a `LIBRARY.json` a text editor mangled. Refusing to list the run over
    that would lose them the run, which is a far worse outcome than losing a
    name they can retype.
    """
    try:
        # `object`, not `Any`: every field is narrowed before it is used, which
        # is what keeps a hand-edited file from reaching the rest of the console
        # as something with the wrong shape.
        raw: object = json.loads((run / LIBRARY_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Metadata()
    if not isinstance(raw, dict):
        return Metadata()
    fields: dict[str, object] = {str(key): value for key, value in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]
    pinned = fields.get("pinned")
    names: list[str] = []
    if isinstance(pinned, list):
        names = [str(name) for name in pinned]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return Metadata(
        display_name=str(fields.get("display_name", "")),
        note=str(fields.get("note", "")),
        pinned=names,
    )


def write_metadata(run: Path, metadata: Metadata) -> Path:
    """Replace `LIBRARY.json` atomically.

    Atomic because a run may be *going*: the console writes this while the
    trainer writes everything else in the same directory, and a half-written
    name that a later read rejects would silently lose the rename.
    """
    run.mkdir(parents=True, exist_ok=True)
    destination = run / LIBRARY_NAME
    temporary = destination.with_name(f"{LIBRARY_NAME}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "display_name": metadata.display_name,
                "note": metadata.note,
                "pinned": sorted(set(metadata.pinned)),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def rename(run: Path, display_name: str) -> Path:
    """Give a run a name a person chose. The directory never moves.

    Renaming the directory instead would break every path inside the run, every
    `--resume` anyone has written down, and the console window that is watching
    it. A name is a label, not an address.
    """
    metadata = read_metadata(run)
    metadata.display_name = display_name.strip()
    return write_metadata(run, metadata)


def set_note(run: Path, note: str) -> Path:
    metadata = read_metadata(run)
    metadata.note = note.strip()
    return write_metadata(run, metadata)


def pin(run: Path, recording: Path) -> Path:
    """Mark a recording as one cleanup must keep.

    Stored as a bare filename rather than a path: the run directory can be moved
    or restored from an archive under a different name, and a pin that pointed
    at the old absolute path would silently stop protecting anything.
    """
    metadata = read_metadata(run)
    if recording.name not in metadata.pinned:
        metadata.pinned.append(recording.name)
    return write_metadata(run, metadata)


def unpin(run: Path, recording: Path) -> Path:
    metadata = read_metadata(run)
    metadata.pinned = [name for name in metadata.pinned if name != recording.name]
    return write_metadata(run, metadata)


# ---- naming ------------------------------------------------------------------

#: Words a generated name is built from. Deliberately concrete and unrelated to
#: each other, because the job of a default name is to be *distinguishable at a
#: glance* — "brisk-harbour" and "amber-lantern" are told apart instantly where
#: "runs-11" and "runs-12" are not, which is the entire complaint.
_ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "dusk",
    "eager",
    "fleet",
    "grave",
    "high",
    "keen",
    "lone",
    "north",
    "prime",
    "quick",
    "rapid",
    "sharp",
    "still",
)
_NOUNS = (
    "anvil",
    "beacon",
    "cinder",
    "delta",
    "ember",
    "falcon",
    "harbour",
    "ion",
    "lantern",
    "meridian",
    "nova",
    "orbit",
    "pylon",
    "quarry",
    "ridge",
    "summit",
)


#: Everything that is not a letter or a digit, collapsed to one dash. A display
#: name is typed by a person — `Entity policy, 3 seeds` — and a directory name is
#: typed at a shell prompt, put in a `--resume` and quoted by nobody, so the two
#: cannot be the same string. `md.league.make_id` applies the same rule to a
#: promoted model's directory, for the same reason.
_ID_UNSAFE = re.compile(r"[^a-z0-9]+")


def run_id_for(display_name: str, taken: Iterable[str] = ()) -> str:
    """A directory name for a run somebody has just named.

    Uniqueness is enforced by suffixing rather than by rejecting: the answer to
    "that name is taken" is a name that is not, and a dialog that refuses at the
    moment somebody is trying to start training is a dialog that gets in the
    way. Case-insensitively, because two directories differing only in case are
    the same directory on Windows and macOS.
    """
    stem = _ID_UNSAFE.sub("-", display_name.strip().lower()).strip("-")
    if not stem:
        stem = "run"  # a name made entirely of punctuation is still a run
    used = {name.strip().lower() for name in taken}
    if stem not in used:
        return stem
    index = 2
    while f"{stem}-{index}" in used:
        index += 1
    return f"{stem}-{index}"


def new_run_dir(root: Path, display_name: str) -> Path:
    """Where a run called ``display_name`` should be written. Creates nothing.

    Unique against **every directory in ``root``**, not only the ones that are
    runs: a directory a trainer has not written `metrics.csv` into yet is
    invisible to :func:`discover` and entirely visible to `mkdir`, and a new run
    that landed on top of one would mix two runs together.
    """
    try:
        taken = [child.name for child in root.iterdir() if child.is_dir()]
    except OSError:
        taken = []  # the library directory itself does not exist yet
    return root / run_id_for(display_name, taken)


def default_name(existing: Iterable[str] = (), *, seed: int | None = None) -> str:
    """A name for a new run that is better than a timestamp.

    A timestamp is unique and unmemorable, which is the wrong trade: you read
    this in a list while deciding which of four experiments to keep, and
    `2026-07-26 21:04` and `2026-07-26 22:11` are the same shape. Two short
    words are not.

    Collisions are avoided rather than tolerated — the point is to be a handle,
    and two runs called `amber-anvil` would be worse than two called `runs-2`
    and `runs-3`, which at least differ.
    """
    taken = {name.strip().lower() for name in existing}
    start = int(time.time()) if seed is None else seed
    for offset in range(len(_ADJECTIVES) * len(_NOUNS)):
        index = start + offset
        adjective = _ADJECTIVES[index % len(_ADJECTIVES)]
        noun = _NOUNS[(index // len(_ADJECTIVES)) % len(_NOUNS)]
        candidate = f"{adjective}-{noun}"
        if candidate not in taken:
            return candidate
    # Every pair is taken, which is 256 runs in one directory. Fall back to
    # something that cannot collide rather than returning a duplicate.
    return f"run-{start:x}"


# ---- discovery ---------------------------------------------------------------


def _tree_size(root: Path, suffix: str | None = None) -> int:
    total = 0
    try:
        for entry in root.iterdir():
            if entry.is_dir():
                total += _tree_size(entry, suffix)
            elif suffix is None or entry.suffix == suffix:
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue  # deleted between listing and stat'ing
    except OSError:
        return total
    return total


def storage_of(run: Path) -> Storage:
    """What this run occupies, split three ways."""
    checkpoints = _tree_size(run / sources.CHECKPOINTS_NAME)
    recordings = sum(recording.size for recording in sources.list_recordings(run))
    return Storage(
        checkpoints=checkpoints,
        recordings=recordings,
        other=max(0, _tree_size(run) - checkpoints - recordings),
    )


def _eval_rows(run: Path) -> list[sources.EvalRow]:
    """Every scored evaluation in this run, oldest first.

    Read whole rather than tailed: the library is a list built once per visit,
    not a live view, and a tail would carry an offset that outlives the answer.
    """
    tail = sources.evals_tail(run)
    return list(tail.poll().rows)


def best_evaluated_checkpoint(run: Path) -> tuple[Path, sources.EvalRow] | None:
    """The highest-scoring checkpoint that actually exists on disk.

    Two halves, and the second is the one that bites: the *best score* is a row
    in `evals.csv`, and the *checkpoint for it* may have been deleted — the
    trainer writes one every `--checkpoint-every` updates and evaluates on a
    different cadence, so a run's best evaluation frequently has no checkpoint
    of its own. Promoting the best *score* while shipping whatever checkpoint
    happened to survive would be a model card that lies.

    So: the best row whose checkpoint is present, `policy-best.pt` included —
    which the trainer writes precisely because PPO peaks and then regresses.
    """
    rows = _eval_rows(run)
    if not rows:
        return None
    checkpoints = {
        checkpoint.iteration: checkpoint.path
        for checkpoint in sources.list_checkpoints(run)
        if checkpoint.iteration is not None
    }
    best: tuple[Path, sources.EvalRow] | None = None
    for row in rows:
        path = checkpoints.get(row.update)
        if path is None:
            continue
        if best is None or row.mean_score > best[1].mean_score:
            best = (path, row)
    return best


def _is_live(path: Path, modified: float) -> bool:
    """Whether something is writing this run, asked of the OS where possible.

    The `RUNNING` marker names the trainer's PID, so a finished run reads as
    finished immediately and a killed one does too — the marker outlives the
    process, the process does not outlive the check. Only a run with no marker
    at all falls back to the timestamp, which is what an older trainer's output
    looks like.
    """
    control = Control(path)
    if control.owner() is not None:
        return control.running()
    return (time.time() - modified) < LIVE_AFTER_S


def load_run(path: Path) -> Run | None:
    """Read one run directory, or ``None`` if there is no run in it.

    ``None`` and not an empty ``Run``: `runs/` itself usually holds one run per
    experiment rather than a run, and a library that listed the parent as an
    empty run would put a row on screen for a directory.
    """
    if not path.is_dir():
        return None
    metrics = path / sources.METRICS_NAME
    modified = sources.last_modified(metrics)
    if modified is None:
        return None

    metadata = read_metadata(path)
    rows = _eval_rows(path)
    best = max(rows, key=lambda row: row.mean_score) if rows else None
    best_checkpoint = best_evaluated_checkpoint(path)
    recordings = sources.list_recordings(path)
    checkpoints = sources.list_checkpoints(path)

    # One row per update, and the file can be large; counted rather than kept.
    updates = 0
    tail = sources.metrics_tail(path)
    updates = len(tail.poll().rows)

    return Run(
        path=path,
        run_id=path.name,
        display_name=metadata.display_name,
        state="live" if _is_live(path, modified) else "stopped",
        updates=updates,
        best_score=best.mean_score if best else None,
        best_update=best.update if best else None,
        best_checkpoint=best_checkpoint[0] if best_checkpoint else None,
        last_activity=modified,
        storage=storage_of(path),
        recordings=len(recordings),
        checkpoints=len(checkpoints),
        note=metadata.note,
        pinned=tuple(sorted(metadata.pinned)),
    )


def discover(root: Path) -> list[Run]:
    """Every run at or inside ``root``, most recently active first.

    Both shapes, because both exist in practice and nobody should have to say
    which they have: `runs/` holding one run, and `runs/` holding a directory
    per experiment. Not recursive beyond one level — a run directory holds
    `checkpoints/`, and descending into it would find nothing and cost a walk.
    """
    found: list[Run] = []
    here = load_run(root)
    if here is not None:
        found.append(here)
    try:
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        children = []
    for child in children:
        run = load_run(child)
        if run is not None:
            found.append(run)
    found.sort(key=lambda run: (run.last_activity or 0.0, run.run_id), reverse=True)
    return found


def within(root: Path, candidate: Path) -> bool:
    """Whether ``candidate`` is inside the managed root.

    Asked before anything is written or deleted. A run directory can be given by
    a picker, an environment variable or a command line, and "delete everything
    under the path I was handed" is a sentence that has ended badly for other
    programs. `os.path.commonpath` rather than string prefixes, so `/runs-2` is
    not treated as being inside `/runs`.
    """
    try:
        return os.path.commonpath([root.resolve(), candidate.resolve()]) == str(root.resolve())
    except (OSError, ValueError):
        return False  # different drives on Windows, or a path that cannot resolve


def describe(run: Run) -> str:
    """One line for a list row: what it is and where it got to."""
    parts = [run.name]
    if run.display_name and run.display_name != run.run_id:
        parts.append(f"({run.run_id})")
    parts.append(f"{run.updates:,} updates")
    if run.best_score is not None:
        parts.append(f"best {run.best_score:,.0f}")
    parts.append(sources.human_size(run.storage.total))
    return " · ".join(parts)


def model_name_for(run: Run, checkpoint: Path) -> str:
    """What a model promoted from this run should be called by default.

    The run's name, because that is the identity the person has been holding in
    their head for the last three hours — plus the update, because a run can be
    promoted more than once and "amber-anvil" twice in a league table is exactly
    the confusion naming was supposed to end.
    """
    iteration = sources.list_checkpoints(run.path)
    for entry in iteration:
        if entry.path == checkpoint and entry.iteration is not None:
            return f"{run.name} @{entry.iteration}"
    return run.name


def recordings_of(run: Run) -> Sequence[sources.Recording]:
    return sources.list_recordings(run.path)


def is_pinned(run: Run, recording: sources.Recording) -> bool:
    return recording.path.name in run.pinned


def modelcard_of(run: Run) -> modelcard.ModelCard | None:
    """What the run is training, if it said. Used to fill a promotion dialog."""
    return modelcard.read(run.path)
