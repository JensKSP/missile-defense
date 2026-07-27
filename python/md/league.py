# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The local Model League: promoted policies that outlive the runs they came from.

**Promotion is a copy, deliberately.** Pointing at `runs-7/checkpoints/policy-00800.pt`
would be cheaper and would break the first time that run is cleaned up, archived
or deleted — and those are the operations Task 9 exists to make safe. A promoted
model is `models/<id>/policy.mdp` plus a `model.json` beside it, and nothing in
the run directory is load-bearing afterwards.

**It is also a conversion.** What lands is an `.mdp`, not the `.pt`: the game and
the native evaluator read that and never a pickle (docs/API.md §7), so a model
that cannot be exported and re-read cannot be promoted at all. That is the
intended failure — a league entry the game cannot load is worse than a refusal.

Two names, the same split as a run's (:mod:`md.library`). The **id** is a stable
directory name, derived once and never changed, and it is what every result,
match and archive refers to. The **display name** is what a person reads and can
change whenever they like. The display name travels *into* `model.json`, which
is where the game reads it for the HUD and the league reads it for the table:
paths are not names, and `policy-best.pt` says nothing about which run produced
it.

**Display names are unique, and that is enforced here rather than asked for.**
The name is the only thing a person sees — in the league table, in the game's
MODELS menu, in a head-to-head result — so two models called `deadline-1330` are
two rows nobody can tell apart, and picking the wrong one is silent. Ids stay
unique by suffixing, because an id is a path and a path must always resolve;
names cannot do the same, since `deadline-1330-2` is not a name anybody chose. So
a colliding name is refused with :class:`DuplicateName`, and the caller decides:
pick another name, or **replace** the model that has it (`replace=`), which is
the same swap done atomically and takes the old model's results with it.

Every write is atomic and validated. A half-written league entry is one the game
finds and refuses at the worst possible moment, so an entry either exists whole
or does not exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, policy_format

#: How long the exporter gets. It reads one checkpoint and writes one file, so
#: this is a bound on a hang rather than a budget — a runtime that has to page
#: several gigabytes of torch in from a cold disk is the slow honest case.
EXPORT_TIMEOUT_S = 300.0

#: What every model directory holds. Two files, and both are read by code that
#: has no Python in it — `policy.mdp` by `md::agent::Policy`, `model.json` by
#: anything that wants the name without loading the weights.
POLICY_NAME = "policy.mdp"
CARD_NAME = "model.json"

#: How the results of an evaluation are stored, beside the model rather than in
#: one index file. One writer per directory means a tournament writing four
#: results cannot lose three of them to a lost update.
RESULTS_NAME = "results.json"


class LeagueError(Exception):
    """A promotion or an import that could not be completed, and why."""


class DuplicateName(LeagueError):
    """The chosen name is already a model's — and this is which model.

    Carries the entry rather than only its name because every caller has the
    same next question: *replace that one, or pick another name?* Neither can be
    answered without knowing what is already there — when it was promoted, what
    it scored — and re-finding it from a message would be a second lookup that
    could disagree with the first.
    """

    def __init__(self, existing: Model) -> None:
        super().__init__(
            f"the league already has a model called {existing.name!r} ({existing.model_id})"
        )
        self.existing = existing


@dataclass(frozen=True)
class Model:
    """One promoted policy."""

    path: Path
    #: Stable, immutable, and what every result and match refers to.
    model_id: str
    display_name: str
    #: Where it came from, for a card nobody has to take on trust.
    source_run: str = ""
    source_checkpoint: str = ""
    trained_updates: int = 0
    promoted_at: float = 0.0
    note: str = ""
    #: Canonical results, newest last. A model with none is unranked, which is
    #: a state the table shows rather than hides.
    results: tuple[Mapping[str, object], ...] = ()

    @property
    def policy(self) -> Path:
        return self.path / POLICY_NAME

    @property
    def name(self) -> str:
        return self.display_name or self.model_id

    @property
    def best_result(self) -> Mapping[str, object] | None:
        """The highest canonical mean score this model has recorded."""
        ranked = [r for r in self.results if r.get("canonical")]
        if not ranked:
            return None
        return max(ranked, key=lambda r: _number(r.get("mean_score")))


# ---- identity ----------------------------------------------------------------

#: What a model id may contain. Restrictive on purpose: an id is a directory
#: name, it ends up in archive entry paths (Task 9) and on command lines, and
#: every character outside this set is a way for one of those to go wrong.
_ID_SAFE = re.compile(r"[^a-z0-9]+")


def make_id(display_name: str, taken: Sequence[str] = ()) -> str:
    """A stable directory name for ``display_name``, unique among ``taken``.

    Derived from the name rather than random, because someone will read this in
    a path and a shell prompt and `amber-anvil-800` tells them which model it is
    where a UUID does not. Uniqueness is enforced by suffixing rather than by
    rejecting: promotion is a thing you do at the end of three hours of
    training, and failing it over a name collision would be poor manners.
    """
    stem = _ID_SAFE.sub("-", display_name.strip().lower()).strip("-")
    if not stem:
        stem = "model"
    if stem not in taken:
        return stem
    index = 2
    while f"{stem}-{index}" in taken:
        index += 1
    return f"{stem}-{index}"


def same_name(left: str, right: str) -> bool:
    """Whether two display names are the same name to a person.

    Case-insensitively and with runs of whitespace flattened, because `Amber
    Anvil`, `amber anvil` and `Amber  Anvil` are one name on a menu and telling
    them apart is a job for a machine, not for someone choosing a model to play.
    `casefold` and not `lower`, so this holds for names that are not ASCII.
    """
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def find_by_name(display_name: str, root: Path | None = None) -> Model | None:
    """The model already called ``display_name``, if any. Asked before writing.

    Compares against the name a person *reads* — the display name, falling back
    to the id for an entry that has none — since that is the string the clash is
    about. Two models whose ids differ and whose names match are exactly the
    confusion this prevents.
    """
    name = display_name.strip()
    if not name:
        return None
    for model in models(root):
        if same_name(model.name, name):
            return model
    return None


# ---- reading -----------------------------------------------------------------


def _number(value: object) -> float:
    """A score out of a stored result, or zero. Never raises: a results file is
    on disk, and a malformed one must not stop a table from being drawn."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _read_json(path: Path) -> dict[str, object]:
    """A JSON object as `{str: object}`, or empty for anything else.

    `object` and not `Any`: every field below is narrowed before it is used, so
    a card somebody edited by hand cannot reach the league table as a float
    where a string belongs.
    """
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]


def load_model(path: Path) -> Model | None:
    """Read one model directory, or ``None`` when it does not hold a whole model.

    Both files, and the policy has to *be there* — a directory with a card and
    no weights is what a promotion interrupted halfway would leave, and listing
    it would put a row on screen that nothing can play.
    """
    if not (path / POLICY_NAME).is_file():
        return None
    card = _read_json(path / CARD_NAME)
    stored: object = _read_json(path / RESULTS_NAME).get("results")
    results: list[Mapping[str, object]] = []
    if isinstance(stored, list):
        for entry in stored:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(entry, dict):
                results.append({str(key): value for key, value in entry.items()})  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]
    return Model(
        path=path,
        model_id=path.name,
        display_name=str(card.get("display_name", "")),
        source_run=str(card.get("source_run", "")),
        source_checkpoint=str(card.get("source_checkpoint", "")),
        trained_updates=int(_number(card.get("trained_updates"))),
        promoted_at=_number(card.get("promoted_at")),
        note=str(card.get("note", "")),
        results=tuple(results),
    )


def matches_dir(root: Path | None = None) -> Path:
    """Where recorded matches live: a `matches/` sibling of the league.

    Beside the models rather than inside one, because a match belongs to two of
    them and putting it under either would make deleting that one take the
    comparison with it. Creates nothing.
    """
    return (paths.models_dir() if root is None else root).parent / "matches"


def models(root: Path | None = None) -> list[Model]:
    """Every promoted model, newest first.

    Dot-directories are skipped: a promotion in flight is a whole model sitting
    in `.<id>.incoming`, and a replacement holds the superseded one in
    `.<id>.superseded` for as long as the swap takes. Both are complete enough
    to load, and neither is in the league — listing one would put a row on
    screen that vanishes a moment later, or worse, a duplicate of the row beside
    it.
    """
    directory = paths.models_dir() if root is None else root
    found: list[Model] = []
    try:
        children = sorted(
            child for child in directory.iterdir() if child.is_dir() and child.name[:1] != "."
        )
    except OSError:
        return []
    for child in children:
        model = load_model(child)
        if model is not None:
            found.append(model)
    found.sort(key=lambda model: (model.promoted_at, model.model_id), reverse=True)
    return found


def find(model_id: str, root: Path | None = None) -> Model | None:
    directory = paths.models_dir() if root is None else root
    return load_model(directory / model_id)


# ---- promotion ---------------------------------------------------------------


@dataclass
class Promotion:
    """What is about to be promoted, separated from the act of promoting it.

    Planning apart from mutating is the same shape :mod:`md.runtime` uses and
    exists for the same reason: the dialog needs to show exactly what will
    happen — which checkpoint, under what name, at what score — and a function
    that both decided and did it could only be shown by doing it.
    """

    checkpoint: Path
    display_name: str
    source_run: str = ""
    trained_updates: int = 0
    note: str = ""
    #: Anything else worth carrying into the card: the canonical score the
    #: checkpoint recorded, the seed protocol, the simulator version.
    metadata: dict[str, str | int | float] = field(default_factory=dict[str, "str | int | float"])


def _destination_id(display_name: str, directory: Path, replace: Model | None) -> str:
    """Which directory this entry will occupy, refusing a name already in use.

    Asked *before* a checkpoint is read or a byte is written, so a name clash
    costs nothing and can be answered by typing a different name. The two
    outcomes are the whole of the naming policy: a free name gets a fresh id, a
    taken one either raises or — when the caller has been told and chose to go
    ahead — reuses the id of the model it is replacing.
    """
    clash = find_by_name(display_name, directory)
    if replace is None:
        if clash is not None:
            raise DuplicateName(clash)
        taken = [child.name for child in directory.iterdir() if child.is_dir()]
        return make_id(display_name, taken)
    if clash is not None and clash.path != replace.path:
        # Renaming onto a *third* model's name while replacing a second. The
        # league would end up with the duplicate this whole path exists to
        # prevent, so it is refused as if no replacement had been asked for.
        raise DuplicateName(clash)
    if load_model(replace.path) is None:
        raise LeagueError(f"{replace.model_id} is no longer in the league; nothing to replace")
    return replace.model_id


def _install(staging: Path, destination: Path, *, replacing: bool) -> None:
    """Move a validated staging directory into place, replacing at most one entry.

    The old entry is moved aside before the new one is moved in and is only
    deleted once that has succeeded, so the failure mode is a leftover
    `.superseded` directory — invisible to :func:`models` — rather than a league
    with a hole in it. Two renames cannot be made one atomic step on any
    filesystem here, so the ordering is chosen to make the *reachable* states
    the harmless ones.
    """
    if not destination.exists():
        if replacing:  # pragma: no cover — `_destination_id` just loaded it
            raise LeagueError(f"{destination} disappeared while it was being replaced")
        staging.rename(destination)
        return
    if not replacing:
        # `make_id` already avoided every name it could see, so this is a
        # second promotion racing the first rather than a collision.
        raise LeagueError(f"{destination} appeared while promoting; try again")

    superseded = destination.with_name(f".{destination.name}.superseded")
    shutil.rmtree(superseded, ignore_errors=True)
    destination.rename(superseded)
    try:
        staging.rename(destination)
    except OSError:
        destination_gone = not destination.exists()
        if destination_gone:
            superseded.rename(destination)  # put the old one back; nothing was lost
        raise
    shutil.rmtree(superseded, ignore_errors=True)


def _export(
    checkpoint: Path,
    destination: Path,
    metadata: Mapping[str, str | int | float],
    python: str | None,
) -> None:
    """Convert a checkpoint here, or in the interpreter that has torch.

    **The one console action that needs torch.** Everything else about a run is
    read from the files it left, which is what lets a console with no CUDA and
    no torch anywhere near it watch, compare and archive runs. Opening a `.pt`
    is the exception, and importing torch into the console to do it is exactly
    the dependency this design refuses.

    So promotion borrows the interpreter it would start a *run* with. That is
    already the machine's answer to "which Python can do torch things", already
    verified, and already the one whose torch matches the checkpoint's. Where
    torch is beside this process — a checkout, and every test here — nothing is
    spawned and the difference is invisible.

    The failure this replaces: promoting from a packaged console raised
    `ModuleNotFoundError: No module named 'torch'` out of a Qt slot, which is a
    button that does nothing.
    """
    from . import export_policy  # noqa: PLC0415 — imports numpy, not torch

    # Absolute, and deliberately *not* resolved. A venv's `bin/python` is a
    # symlink to the system interpreter and so is the training runtime's, so
    # resolving both makes every venv on the machine look like the same one —
    # and this call took the in-process path straight back into the missing
    # torch it exists to route around.
    here = python is None or os.path.abspath(python) == os.path.abspath(sys.executable)
    if here:
        try:
            export_policy.export_checkpoint(checkpoint, destination, metadata=metadata)
        except ImportError as error:  # torch is not beside this console
            raise LeagueError(
                "promoting a model needs the training runtime, and this console has "
                f"no torch of its own ({error}). Set up training from the console, "
                "or run it from a checkout with torch installed."
            ) from error
        return

    command = [
        str(python),
        "-m",
        "md.export_policy",
        str(checkpoint),
        str(destination),
        "--metadata",
        json.dumps(dict(metadata)),
    ]
    # The runtime is a bare venv with torch in it; `md` itself lives here, in
    # this checkout or this installation, so it has to be put on the path the
    # same way starting a trainer does (`md.ui.runner.training_environ`).
    environ = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[1])
    existing = environ.get("PYTHONPATH", "")
    if package_root not in existing.split(os.pathsep):
        environ["PYTHONPATH"] = (
            f"{package_root}{os.pathsep}{existing}" if existing else package_root
        )
    try:
        finished = subprocess.run(  # noqa: S603 — the interpreter this console starts runs with
            command, capture_output=True, text=True, env=environ, timeout=EXPORT_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise LeagueError(f"the training runtime could not be run: {error}") from error
    if finished.returncode != 0:
        # stderr is the exporter's own sentence (`md.export_policy.main`), which
        # was written to be read by a person. Anything else means it died before
        # reaching that, and then the whole of what it said is what there is.
        detail = finished.stderr.strip() or finished.stdout.strip() or "no output"
        raise LeagueError(detail)


def promote(
    plan: Promotion,
    root: Path | None = None,
    *,
    replace: Model | None = None,
    python: str | None = None,
) -> Model:
    """Export ``plan.checkpoint`` into the league, atomically.

    The whole operation happens in a temporary directory beside the destination
    and is renamed into place only once the exported policy has been **read back
    and validated**. So a failure — an unexportable architecture, a full disk, a
    checkpoint from a different simulator — leaves the league exactly as it was,
    and a success is an entry the game can load.

    ``replace`` names the model this one supersedes, which is what a caller
    passes after :class:`DuplicateName` and the person said yes. It keeps that
    model's **id**, so a path anybody wrote down still resolves, and drops its
    **results**, because those were measured on the weights being replaced and
    carrying them over would be a table that lies. Without it, a name already in
    the league raises :class:`DuplicateName` before the checkpoint is even read.

    ``python`` is the interpreter to open the checkpoint with — the managed
    training runtime, normally, since reading a `.pt` needs torch and the
    console is built never to have it. Leave it out where torch is beside this
    process, which is what a checkout and every test here are.

    Raises :class:`LeagueError`, with the reason, for anything that stops it.
    """
    from . import export_policy  # noqa: PLC0415 — the exporter, not torch (see `_export`)

    directory = paths.models_dir() if root is None else root
    directory.mkdir(parents=True, exist_ok=True)
    model_id = _destination_id(plan.display_name, directory, replace)

    staging = directory / f".{model_id}.incoming"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        metadata: dict[str, str | int | float] = dict(plan.metadata)
        metadata["display_name"] = plan.display_name
        try:
            _export(plan.checkpoint, staging / POLICY_NAME, metadata, python)
        except export_policy.ExportError as error:
            raise LeagueError(str(error)) from error

        # Read it back before anything is renamed into place. Exporting and
        # then trusting the result would let a league entry exist that only the
        # writer can read, which is the one failure this ordering removes.
        try:
            policy_format.read(staging / POLICY_NAME)
        except policy_format.PolicyFormatError as error:
            raise LeagueError(f"the exported policy could not be read back: {error}") from error

        card: dict[str, object] = {
            "id": model_id,
            "display_name": plan.display_name,
            "source_run": plan.source_run,
            "source_checkpoint": plan.checkpoint.name,
            "trained_updates": plan.trained_updates,
            "promoted_at": time.time(),
            "note": plan.note,
            **plan.metadata,
        }
        (staging / CARD_NAME).write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
        _install(staging, directory / model_id, replacing=replace is not None)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    model = load_model(directory / model_id)
    if model is None:  # pragma: no cover — the rename above just succeeded
        raise LeagueError(f"{model_id} did not survive promotion")
    return model


def import_policy(
    source: Path, display_name: str, root: Path | None = None, *, replace: Model | None = None
) -> Model:
    """Add an `.mdp` somebody else produced, after checking this build can run it.

    The same validation promotion does and in the same order, because the file
    is *more* suspect here, not less: it may have been downloaded. Nothing is
    written until `policy_format.read` has accepted it.

    Names collide here more often than anywhere else — a downloaded `.mdp`
    carries whatever its author called it, and re-importing one you already have
    is the ordinary case — so ``replace`` means the same as it does in
    :func:`promote`, and an unnamed clash raises :class:`DuplicateName`.
    """
    try:
        policy = policy_format.read(source)
    except policy_format.PolicyFormatError as error:
        raise LeagueError(str(error)) from error

    directory = paths.models_dir() if root is None else root
    directory.mkdir(parents=True, exist_ok=True)
    name = display_name.strip() or str(policy.metadata.get("display_name", "")) or source.stem
    model_id = _destination_id(name, directory, replace)

    staging = directory / f".{model_id}.incoming"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(source, staging / POLICY_NAME)
        policy_format.read(staging / POLICY_NAME)  # the copy, not the original
        card: dict[str, object] = {
            "id": model_id,
            "display_name": name,
            "source_run": "",
            "source_checkpoint": str(source),
            "trained_updates": int(_number(policy.metadata.get("trained_updates"))),
            "promoted_at": time.time(),
            "note": "imported",
        }
        (staging / CARD_NAME).write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
        _install(staging, directory / model_id, replacing=replace is not None)
    except (OSError, policy_format.PolicyFormatError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise LeagueError(f"{source}: could not be imported ({error})") from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    model = load_model(directory / model_id)
    if model is None:  # pragma: no cover
        raise LeagueError(f"{model_id} did not survive import")
    return model


def rename(model: Model, display_name: str) -> Model:
    """Change what a model is called. The id, and every reference to it, stands.

    Refuses a name another model already has, for the reason the module header
    gives: the name is the only thing anyone sees, so two of them are two rows
    nobody can tell apart. Renaming a model to the name it already has is not a
    collision with itself and is allowed to be a no-op.
    """
    name = display_name.strip()
    clash = find_by_name(name, model.path.parent)
    if clash is not None and clash.path != model.path:
        raise DuplicateName(clash)
    card = _read_json(model.path / CARD_NAME)
    card["display_name"] = name
    temporary = model.path / f"{CARD_NAME}.tmp"
    temporary.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
    temporary.replace(model.path / CARD_NAME)
    found = load_model(model.path)
    if found is None:  # pragma: no cover
        raise LeagueError(f"{model.model_id} disappeared while being renamed")
    return found


def size_of(model: Model) -> int:
    """What this entry occupies on disk: weights, card and recorded results."""
    total = 0
    for entry in model.path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue  # deleted between listing and stat'ing
    return total


def delete(model: Model, root: Path | None = None) -> int:
    """Remove a model from the league for good. Returns the bytes freed.

    **This is also how a model leaves the game**: the MODELS menu lists exactly
    what is in this directory, so an entry deleted here stops being offered the
    next time that screen is opened. There is nowhere else to remove it from,
    and no second copy — `Export…` first if the weights are worth keeping, since
    a promoted model is often the only surviving artifact of a run that has been
    cleaned up.

    Recorded matches are deliberately left alone. They live in `matches/`
    precisely because a match belongs to *two* models, and taking the comparison
    away with one contestant would delete evidence about the other.

    Two guards stand in front of `rmtree`, for the reason
    :func:`md.archive.delete_run` gives at greater length — a path that arrived
    from a picker or an environment variable is not automatically a thing you
    may recursively delete:

    * the entry must be a direct child of the league directory, so a `Model`
      built from somewhere else cannot be removed through this door;
    * it must not *be* that directory, which would take the whole league.
    """
    directory = (paths.models_dir() if root is None else root).resolve()
    path = model.path.resolve()
    if path == directory:
        raise LeagueError(f"{model.path} is the league directory itself; refusing to remove it")
    if path.parent != directory:
        raise LeagueError(f"{model.path} is not in {directory}; refusing to remove it")
    freed = size_of(model)
    shutil.rmtree(model.path)
    return freed


def record_result(model: Model, result: Mapping[str, object]) -> Model:
    """Append one evaluation to a model's history, atomically.

    Appended rather than replaced: a model's results are its record, and a
    tournament that overwrote the previous one would make "has this got better
    since?" unanswerable. Ranking reads only the canonical ones (Task 7).
    """
    history: object = _read_json(model.path / RESULTS_NAME).get("results")
    entries: list[object] = list(history) if isinstance(history, list) else []  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    entries.append({**result, "recorded_at": time.time()})
    temporary = model.path / f"{RESULTS_NAME}.tmp"
    temporary.write_text(json.dumps({"results": entries}, indent=1) + "\n", encoding="utf-8")
    temporary.replace(model.path / RESULTS_NAME)
    found = load_model(model.path)
    if found is None:  # pragma: no cover
        raise LeagueError(f"{model.model_id} disappeared while recording a result")
    return found
