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

Every write is atomic and validated. A half-written league entry is one the game
finds and refuses at the worst possible moment, so an entry either exists whole
or does not exist.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, policy_format

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


def models(root: Path | None = None) -> list[Model]:
    """Every promoted model, newest first."""
    directory = paths.models_dir() if root is None else root
    found: list[Model] = []
    try:
        children = sorted(child for child in directory.iterdir() if child.is_dir())
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


def promote(plan: Promotion, root: Path | None = None) -> Model:
    """Export ``plan.checkpoint`` into the league, atomically.

    The whole operation happens in a temporary directory beside the destination
    and is renamed into place only once the exported policy has been **read back
    and validated**. So a failure — an unexportable architecture, a full disk, a
    checkpoint from a different simulator — leaves the league exactly as it was,
    and a success is an entry the game can load.

    Raises :class:`LeagueError`, with the reason, for anything that stops it.
    """
    from . import export_policy  # noqa: PLC0415 — pulls in torch; only promotion does

    directory = paths.models_dir() if root is None else root
    directory.mkdir(parents=True, exist_ok=True)
    existing = [child.name for child in directory.iterdir() if child.is_dir()]
    model_id = make_id(plan.display_name, existing)

    staging = directory / f".{model_id}.incoming"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        metadata: dict[str, str | int | float] = dict(plan.metadata)
        metadata["display_name"] = plan.display_name
        try:
            export_policy.export_checkpoint(
                plan.checkpoint, staging / POLICY_NAME, metadata=metadata
            )
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

        destination = directory / model_id
        if destination.exists():
            # `make_id` already avoided every name it could see, so this is a
            # second promotion racing the first rather than a collision.
            raise LeagueError(f"{destination} appeared while promoting; try again")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    model = load_model(directory / model_id)
    if model is None:  # pragma: no cover — the rename above just succeeded
        raise LeagueError(f"{model_id} did not survive promotion")
    return model


def import_policy(source: Path, display_name: str, root: Path | None = None) -> Model:
    """Add an `.mdp` somebody else produced, after checking this build can run it.

    The same validation promotion does and in the same order, because the file
    is *more* suspect here, not less: it may have been downloaded. Nothing is
    written until `policy_format.read` has accepted it.
    """
    try:
        policy = policy_format.read(source)
    except policy_format.PolicyFormatError as error:
        raise LeagueError(str(error)) from error

    directory = paths.models_dir() if root is None else root
    directory.mkdir(parents=True, exist_ok=True)
    existing = [child.name for child in directory.iterdir() if child.is_dir()]
    name = display_name.strip() or str(policy.metadata.get("display_name", "")) or source.stem
    model_id = make_id(name, existing)

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
        staging.rename(directory / model_id)
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
    """Change what a model is called. The id, and every reference to it, stands."""
    card = _read_json(model.path / CARD_NAME)
    card["display_name"] = display_name.strip()
    temporary = model.path / f"{CARD_NAME}.tmp"
    temporary.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")
    temporary.replace(model.path / CARD_NAME)
    found = load_model(model.path)
    if found is None:  # pragma: no cover
        raise LeagueError(f"{model.model_id} disappeared while being renamed")
    return found


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
