# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Freeing disk without losing anything, and archives that come back.

Two jobs that share one property: **both destroy things, so both plan first.**

:func:`plan_cleanup` returns exactly what it would keep and remove and how many
bytes that is, without touching a file. The UI shows the plan; the same plan is
then executed. A "free up space" button that computed its own answer at the
moment you pressed it would be a different answer from the one you agreed to.

:func:`create_archive` writes a ZIP with a manifest and a checksum per entry;
:func:`verify_archive` re-reads it; and **nothing is deleted until verification
has passed**. That ordering is the whole design: the failure to prevent is a run
that was archived, deleted, and then found to be unreadable.

## Unsafe extraction is the risk

A `.zip` is a list of paths that a naive extractor will happily write anywhere.
`zipfile.ZipFile.extractall` sanitises absolute paths and `..` since Python 3.6.5
but does *not* refuse them — it silently rewrites them, so a malicious entry
becomes a real file in an unexpected place rather than an error. And an archive
here may have been handed over by someone else, which is the same trust boundary
`.mdp` has.

So :func:`verify_archive` **refuses** rather than sanitises, and it does so
before a single byte is extracted:

* an absolute path, a drive letter, or any `..` component;
* a symlink or any entry that is not a regular file;
* a declared uncompressed size above :data:`MAX_ENTRY_BYTES`, or a total above
  :data:`MAX_TOTAL_BYTES` — a zip bomb is a small file that claims to be a large
  one, and the claim is in the header where it can be checked for free;
* a duplicate entry name, because which of the two wins is a property of the
  extractor and not of the archive;
* a checksum that does not match, or an entry the manifest does not list.

Restoring never overwrites. It refuses an existing target rather than merging
into it, because a half-merged run is one whose `metrics.csv` and checkpoints
came from different runs and nothing later would notice.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import library
from .ui import sources

#: The manifest inside the archive. Read before anything is extracted.
MANIFEST_NAME = "MANIFEST.json"

#: What this reader understands. An archive from a future version is refused
#: with its version named, rather than half-read.
ARCHIVE_VERSION = 1

#: Ceilings, checked against the *declared* sizes in the zip's own headers so a
#: bomb is refused before it is decompressed. Generous — a run with a thousand
#: checkpoints is a real thing — and finite, which is the point.
MAX_ENTRY_BYTES = 4 * 1024**3
MAX_TOTAL_BYTES = 64 * 1024**3


class ArchiveError(Exception):
    """An archive that cannot be trusted, and why."""


def _seconds(value: object) -> float:
    """A timestamp out of a manifest, or zero. Never raises: a wrong `created`
    is a cosmetic problem and must not stop an archive being restored."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


# ---- cleanup -----------------------------------------------------------------


@dataclass(frozen=True)
class CleanupPlan:
    """Exactly what would go, exactly what would stay, and what that frees.

    Shown before it is executed and then executed *as shown*. Recomputing at the
    moment of the click would be a different answer from the one agreed to.
    """

    keep: tuple[Path, ...]
    remove: tuple[Path, ...]
    reclaim_bytes: int

    @property
    def empty(self) -> bool:
        return not self.remove


def plan_cleanup(run: library.Run, pins: Collection[Path] = ()) -> CleanupPlan:
    """What could be removed from ``run`` without losing anything that matters.

    Three things are never offered, and each is a thing you cannot get back:

    * **the summary artifacts** — `metrics.csv`, `evals.csv`, `config.json`,
      `model.json`, `train.log`. They are what the trainer draws and they are
      kilobytes; deleting them to save space would be trading the whole record
      of a run for nothing.
    * **the best evaluated checkpoint**, and `policy-best.pt` and
      `policy-final.pt` by name. The best is the one worth promoting, and the
      other two are what the trainer itself decided to keep.
    * **pinned recordings** — a pin is a judgement nothing else can infer.

    Everything else is intermediate: checkpoints from update 100 when there are
    checkpoints from update 900, and episodes nobody marked.
    """
    protected: set[Path] = set()
    for name in (
        sources.METRICS_NAME,
        sources.EVALS_NAME,
        "config.json",
        "model.json",
        sources.LOG_NAME,
        library.LIBRARY_NAME,
    ):
        protected.add(run.path / name)

    best = library.best_evaluated_checkpoint(run.path)
    if best is not None:
        protected.add(best[0])
    for name in ("policy-best.pt", "policy-final.pt"):
        protected.add(run.path / sources.CHECKPOINTS_NAME / name)

    pinned = {Path(pin).name for pin in pins} | set(run.pinned)

    keep: list[Path] = []
    remove: list[Path] = []
    reclaimed = 0
    for checkpoint in sources.list_checkpoints(run.path):
        if checkpoint.path in protected:
            keep.append(checkpoint.path)
        else:
            remove.append(checkpoint.path)
            reclaimed += checkpoint.size
    for recording in sources.list_recordings(run.path):
        if recording.path.name in pinned:
            keep.append(recording.path)
        else:
            remove.append(recording.path)
            reclaimed += recording.size

    keep.extend(sorted(path for path in protected if path.exists()))
    return CleanupPlan(tuple(sorted(set(keep))), tuple(sorted(remove)), reclaimed)


def apply_cleanup(plan: CleanupPlan, root: Path) -> int:
    """Remove exactly what the plan named. Returns the bytes actually freed.

    ``root`` is the managed directory and every path is checked against it
    before anything is unlinked — a plan is data, it can be stale or forged, and
    "delete the paths in this list" is a sentence that has ended badly for other
    programs.
    """
    freed = 0
    for path in plan.remove:
        if not library.within(root, path):
            raise ArchiveError(f"{path} is outside {root}; refusing to remove it")
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue  # already gone, or held open on Windows; not a failure
        freed += size
    return freed


# ---- archives ----------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One file in an archive, as the manifest describes it."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ArchiveManifest:
    """What an archive claims to hold, once it has been checked."""

    version: int
    run_id: str
    display_name: str
    created: float
    entries: tuple[Entry, ...] = ()
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.entries)


@dataclass
class ArchiveSelection:
    """What to put in. Defaults to everything that is not regenerable."""

    checkpoints: bool = True
    recordings: bool = True
    #: The CSVs, the config, the model card and the log. Always worth having and
    #: measured in kilobytes, so there is no option to leave them out.
    summary: bool = True
    extra: list[Path] = field(default_factory=list["Path"])


def _digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), size


def _selected(run: library.Run, selection: ArchiveSelection) -> list[Path]:
    chosen: list[Path] = []
    if selection.summary:
        for name in (
            sources.METRICS_NAME,
            sources.EVALS_NAME,
            "config.json",
            "model.json",
            sources.LOG_NAME,
            library.LIBRARY_NAME,
        ):
            candidate = run.path / name
            if candidate.is_file():
                chosen.append(candidate)
    if selection.checkpoints:
        chosen.extend(entry.path for entry in sources.list_checkpoints(run.path))
    if selection.recordings:
        chosen.extend(entry.path for entry in sources.list_recordings(run.path))
    chosen.extend(path for path in selection.extra if path.is_file())
    return sorted(set(chosen))


def create_archive(
    run: library.Run, destination: Path, selection: ArchiveSelection | None = None
) -> Path:
    """Write ``run`` to a ZIP at ``destination``, atomically.

    Written to a sibling temporary and renamed, so an interrupted archive is not
    a file that looks finished. Every entry is hashed on the way in and the
    hashes go in the manifest, which is what makes verification possible without
    the original.
    """
    chosen = _selected(run, selection or ArchiveSelection())
    if not chosen:
        raise ArchiveError(f"{run.path} has nothing worth archiving")

    entries: list[Entry] = []
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in chosen:
                name = path.relative_to(run.path).as_posix()
                sha, size = _digest(path)
                entries.append(Entry(name, size, sha))
                archive.write(path, name)
            manifest = ArchiveManifest(
                version=ARCHIVE_VERSION,
                run_id=run.run_id,
                display_name=run.name,
                created=time.time(),
                entries=tuple(entries),
                note=run.note,
            )
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(
                    {
                        "version": manifest.version,
                        "run_id": manifest.run_id,
                        "display_name": manifest.display_name,
                        "created": manifest.created,
                        "note": manifest.note,
                        "entries": [
                            {"name": e.name, "size": e.size, "sha256": e.sha256}
                            for e in manifest.entries
                        ],
                    },
                    indent=1,
                ),
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_name(name: str) -> None:
    """Refuse anything that would not land where the extractor was told to put it.

    Refused rather than sanitised. `extractall` rewrites these silently, which
    turns a hostile entry into a real file somewhere unexpected instead of an
    error — and an archive can arrive from somebody else.
    """
    if not name or name.endswith("/"):
        raise ArchiveError(f"archive entry {name!r} is not a file")
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith(("/", "\\")):
        raise ArchiveError(f"archive entry {name!r} is an absolute path")
    if ".." in path.parts:
        raise ArchiveError(f"archive entry {name!r} escapes the archive with '..'")
    if ":" in path.parts[0]:  # `C:` on Windows, which is also absolute
        raise ArchiveError(f"archive entry {name!r} names a drive")


def verify_archive(path: Path) -> ArchiveManifest:
    """Read and check an archive. Extracts nothing; raises on anything suspect.

    Called before a restore *and* before the original is deleted, which is the
    ordering the whole module exists to guarantee.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                raw: object = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            except KeyError as error:
                raise ArchiveError(f"{path}: no {MANIFEST_NAME} — not one of ours") from error
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArchiveError(f"{path}: the manifest is not readable JSON") from error
            if not isinstance(raw, dict):
                raise ArchiveError(f"{path}: the manifest is not an object")
            fields: dict[str, object] = {str(k): v for k, v in raw.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]

            version = fields.get("version")
            if version != ARCHIVE_VERSION:
                raise ArchiveError(
                    f"{path}: archive version {version}, this build reads {ARCHIVE_VERSION}"
                )

            declared: dict[str, Entry] = {}
            listed = fields.get("entries")
            if not isinstance(listed, list):
                raise ArchiveError(f"{path}: the manifest lists no entries")
            for item in listed:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(item, dict):
                    raise ArchiveError(f"{path}: a manifest entry is not an object")
                # Narrowed to `{str: object}` before any field is read, for the
                # same reason `missile_defense.policy_format` does: this file may have been
                # handed over, and a loose type here is how a string reaches a
                # size check.
                entry: dict[str, object] = {str(k): v for k, v in item.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]
                name = str(entry.get("name", ""))
                _safe_name(name)
                if name in declared:
                    raise ArchiveError(f"{path}: {name!r} is listed twice")
                size = entry.get("size")
                declared[name] = Entry(
                    name,
                    int(size) if isinstance(size, int) and not isinstance(size, bool) else 0,
                    str(entry.get("sha256", "")),
                )

            total = 0
            seen: set[str] = set()
            for info in archive.infolist():
                if info.filename == MANIFEST_NAME:
                    continue
                _safe_name(info.filename)
                if info.filename in seen:
                    raise ArchiveError(f"{path}: {info.filename!r} appears twice in the archive")
                seen.add(info.filename)
                # Symlinks and devices are encoded in the external attributes'
                # high bits. `extractall` would write the link *target* as a
                # file, which is a quieter surprise than it sounds.
                #
                # Only the **type** field is consulted, and only when it is set.
                # A first pass tested the whole mode against `S_IFREG` and so
                # rejected every entry `zipfile.writestr` produces — it stores
                # permissions with no type bits at all, which is normal and not
                # a symlink. Caught by the tests for the two checks *behind*
                # this one, which could never be reached.
                kind = (info.external_attr >> 16) & 0o170000
                if kind not in (0, 0o100000):
                    raise ArchiveError(f"{path}: {info.filename!r} is not a regular file")
                if info.file_size > MAX_ENTRY_BYTES:
                    raise ArchiveError(f"{path}: {info.filename!r} declares an implausible size")
                total += info.file_size
                if total > MAX_TOTAL_BYTES:
                    raise ArchiveError(f"{path}: declares more than {MAX_TOTAL_BYTES} bytes")
                if info.filename not in declared:
                    raise ArchiveError(
                        f"{path}: {info.filename!r} is in the archive but not in the manifest"
                    )

            missing = set(declared) - seen
            if missing:
                raise ArchiveError(
                    f"{path}: the manifest lists {sorted(missing)}, which are absent"
                )

            for name, listed_entry in declared.items():
                sha = hashlib.sha256()
                with archive.open(name) as handle:
                    while chunk := handle.read(1 << 20):
                        sha.update(chunk)
                if sha.hexdigest() != listed_entry.sha256:
                    raise ArchiveError(f"{path}: {name!r} does not match its checksum")

            return ArchiveManifest(
                version=ARCHIVE_VERSION,
                run_id=str(fields.get("run_id", "")),
                display_name=str(fields.get("display_name", "")),
                created=_seconds(fields.get("created")),
                entries=tuple(declared.values()),
                note=str(fields.get("note", "")),
            )
    except zipfile.BadZipFile as error:
        raise ArchiveError(f"{path}: not a readable archive ({error})") from error
    except OSError as error:
        raise ArchiveError(f"{path}: could not be read ({error})") from error


def restore_archive(path: Path, destination: Path) -> Path:
    """Verify, then extract into a **new** directory. Never overwrites.

    An existing target is refused rather than merged into: a half-merged run has
    a `metrics.csv` from one run and checkpoints from another, and nothing
    downstream would notice.
    """
    manifest = verify_archive(path)
    if destination.exists():
        raise ArchiveError(f"{destination} already exists; restore to a new directory")

    staging = destination.with_name(destination.name + ".incoming")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for entry in manifest.entries:
                target = staging / entry.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry.name) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def archive_and_remove(
    run: library.Run, destination: Path, *, root: Path, selection: ArchiveSelection | None = None
) -> tuple[Path, int]:
    """Archive a run, verify the archive, and only then delete the original.

    The ordering *is* the feature. Anything else risks the one outcome nobody
    recovers from: a run archived, deleted, and then found unreadable.
    """
    written = create_archive(run, destination, selection)
    verify_archive(written)  # raises before a single file is removed
    if not library.within(root, run.path):
        raise ArchiveError(f"{run.path} is outside {root}; refusing to remove it")
    freed = library.storage_of(run.path).total
    shutil.rmtree(run.path)
    return written, freed


def delete_run(run: library.Run, root: Path) -> int:
    """Delete a run outright. Returns the bytes freed.

    The one operation here that keeps no copy of anything, which is why it is
    the one with the shortest implementation and the longest guard. Two refusals
    stand in front of `rmtree`:

    * anything outside ``root``, for the reason :func:`apply_cleanup` says — a
      run path arrives from a picker, an environment variable or a command line,
      and "delete everything under the path I was handed" is a sentence that has
      ended badly for other programs;
    * ``root`` **itself**, because `runs/` holding a single run is a shape
      :func:`library.discover` supports, and deleting that run would take the
      library with it. That one is not a deletion anybody meant, so it is an
      error rather than a confirmation.

    Whether the run is still *going* is deliberately not checked here: liveness
    is a ninety-second-old timestamp, and a layer that unlinks files should
    refuse on facts rather than on a guess. The trainer asks that question where
    it is fresh, before it ever gets this far.
    """
    if not library.within(root, run.path):
        raise ArchiveError(f"{run.path} is outside {root}; refusing to remove it")
    if run.path.resolve() == root.resolve():
        raise ArchiveError(
            f"{run.path} is the library directory itself, not a run inside it; "
            "refusing to remove it"
        )
    freed = library.storage_of(run.path).total
    shutil.rmtree(run.path)
    return freed


def describe(plan: CleanupPlan) -> str:
    """`14 files · 1.2 GB` — what a button offers to do, before it does it."""
    if plan.empty:
        return "nothing to remove"
    return f"{len(plan.remove)} files · {sources.human_size(plan.reclaim_bytes)}"


def entries_of(paths: Iterable[Path]) -> Sequence[str]:
    return [path.name for path in paths]
