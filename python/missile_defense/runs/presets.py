# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Named sets of training options. No Qt, no torch.

A run is configured by twenty-odd knobs with good defaults, and in practice
people use a handful of *combinations* of them: one to check the machinery
turns over, one they trust, one they leave going overnight. Naming those is the
difference between "what did I set last time?" and picking a name.

Three come with the trainer, and their values are not invented:

* **fast** is the throughput configuration from `docs/NVIDIA.md` — the batch
  size at which the card saturates — over a short horizon. It is for watching
  the loop turn and for checking a machine, not for producing a policy.
* **good** is exactly the recipe that produced the bundled `learned-high.mdp`
  (`docs/ROADMAP.md`): relational architecture, 1,024 envs × 256 steps, 1,000
  updates, scoring 23,067 on the held-out block. It is the one preset with a
  *measured* outcome attached to it.
* **best** scales that recipe along the axes the documentation recommends — four
  times the samples per update and four times the updates — and is therefore a
  considered bet rather than a measured result. Its description says so, because
  a preset called "best" that quietly promises a number nobody has measured
  would be the most expensive kind of lie: one paid for in hours of GPU time.

  What it does *not* do is scale the minibatch with the batch. The first version
  of this preset did, and ran out of memory on a 32 GiB card before the first
  update finished: on the relational architecture peak memory follows the
  minibatch, not the batch (:mod:`missile_defense.runs.footprint`). Every built-in is now checked
  against that model by a test, and the trainer shows the estimate before a run
  starts.

The built-ins cannot be edited or deleted, and a saved preset cannot take one of
their names. That is not tidiness — the names appear in this file's
documentation and in `docs/TRAINING.md`, so "good" has to keep meaning what is
written there. Copy one under a new name and change that instead; the trainer's
*Save as…* does exactly this.

Values are stored the way the Start dialog produces them and the way the command
line consumes them: field name to string, with only the fields that differ from
the trainer's own defaults. So a preset is always readable as flags, and the
trainer never becomes the only way to start the run it describes
(`docs/ROADMAP.md`, M8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from . import paths

#: What a preset may not be called, beyond the built-in names: the Start dialog
#: shows this for "no preset — whatever is in the form".
CUSTOM = "custom"


@dataclass(frozen=True)
class Preset:
    """One named configuration: what it is for, and what it changes."""

    name: str
    #: One line, shown under the picker. Says what it is *for*, and — where the
    #: numbers came from a measurement — which one.
    summary: str
    #: Trainer field name to value, exactly as the Start dialog emits it. Only
    #: fields that differ from the trainer's defaults; everything absent is
    #: deliberately left to the dataclass.
    options: Mapping[str, str]
    #: Ships with the trainer: read-only, and its name is reserved.
    builtin: bool = False

    def as_record(self) -> dict[str, object]:
        """What :func:`save` writes. Built-ins are never written."""
        return {"name": self.name, "summary": self.summary, "options": dict(self.options)}


#: The three that ship. Ordered fastest-to-longest, which is also the order a
#: person meets them: prove it works, get a policy, then spend real time.
BUILTIN: tuple[Preset, ...] = (
    Preset(
        name="fast",
        summary=(
            "Throughput first: saturates the GPU, finishes in minutes, needs ~6 GiB. "
            "For checking the loop turns and the machine is set up — not for a policy."
        ),
        options={
            # The measured saturation point (docs/NVIDIA.md): past this, VRAM
            # doubles for a rounding error, and sample efficiency drops.
            "envs": "4096",
            # Half the usual rollout. The batch stays large, but each update
            # arrives twice as often, which is what "watch it turn over" wants.
            "steps": "128",
            # Minutes, not hours. Long enough for the return to start moving so
            # the curve is not flat, short enough to sit and watch.
            "updates": "100",
            # The flat network: cheapest per sample, and this preset is not
            # trying to learn the relational comparison.
            "architecture": "mlp",
            # An evaluation costs most of an update early on. At 100 updates,
            # scoring every 10 would spend a tenth of the run on the yardstick.
            "eval_every": "25",
            "record_every": "50",
            "checkpoint_every": "50",
        },
        builtin=True,
    ),
    Preset(
        name="good",
        summary=(
            "The recipe that produced the bundled model: 23,067 on the held-out "
            "block, clearing HIGH by 1.69x. About two hours on a 5090."
        ),
        options={
            # Exactly docs/ROADMAP.md's reference run. Changing any of these
            # makes the score above someone else's result, not this preset's.
            "architecture": "entity",
            "envs": "1024",
            "steps": "256",
            "updates": "1000",
        },
        builtin=True,
    ),
    Preset(
        name="best",
        summary=(
            "Four times the samples per update, over four times the updates: about "
            "30 hours and ~17 GiB on a 5090. A considered bet, not a measured result."
        ),
        options={
            "architecture": "entity",
            # 2,048 x 512 rather than 4,096 x 256: the same 1,048,576 samples an
            # update either way, and the longer rollout is what docs/TRAINING.md
            # recommends when later-wave resource decisions still receive weak
            # credit — which is exactly what the reference run's remaining gap to
            # HIGH looks like. Throughput at 2,048 envs is within ~10% of the
            # plateau (docs/NVIDIA.md), so the rollout is nearly free.
            "envs": "2048",
            "steps": "512",
            # **The one that stops this being an out-of-memory crash.** The
            # relational architecture's peak memory follows the *minibatch*, not
            # the batch (missile_defense.runs.footprint), so quadrupling the batch at the default 8
            # would ask for 60+ GiB. Sixty-four keeps each piece at 16,384
            # samples — half what `good` puts through at once — for a measured
            # 16.6 GiB peak against `good`'s 18.9. The data and the update are
            # identical; only the size of the pieces changes.
            "minibatches": "64",
            "updates": "4000",
            # Anneal over the whole run rather than stopping early: the schedule
            # defaults to the fresh run's --updates, and this states it so a
            # later --resume extends the run without re-deciding the horizon.
            "schedule_updates": "4000",
            # Four times the updates at four times the batch is a lot of scoring;
            # keep the yardstick frequent enough to see a regression, cheap
            # enough not to pay for it twice.
            "eval_every": "20",
            "checkpoint_every": "100",
        },
        builtin=True,
    ),
)

#: The built-ins by name, for the checks below.
BUILTIN_NAMES = frozenset(preset.name for preset in BUILTIN)


class PresetError(ValueError):
    """A rejected create, update or delete — with the reason, for a dialog."""


def normalise(name: str) -> str:
    """A preset's name, as stored and compared: trimmed, never empty."""

    cleaned = " ".join(name.split())
    if not cleaned:
        raise PresetError("a preset needs a name")
    if cleaned.casefold() == CUSTOM:
        raise PresetError(f"'{CUSTOM}' is what the picker calls no preset at all")
    return cleaned


def _path(path: Path | None) -> Path:
    return paths.presets_file() if path is None else path


def user_presets(path: Path | None = None) -> list[Preset]:
    """The saved ones, in the order they were saved. Never the built-ins.

    A file that is missing, empty, unreadable or not the shape this writes is
    *no presets* rather than an error: this is a convenience over the flags, and
    a corrupt options file must not be the reason a run cannot be started.
    """

    file = _path(path)
    try:
        payload: object = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    saved: list[Preset] = []
    for record in cast("list[object]", payload):
        preset = _preset_from(record)
        if preset is not None and preset.name not in BUILTIN_NAMES:
            saved.append(preset)
    return saved


def _preset_from(record: object) -> Preset | None:
    """One record, or ``None`` if it is not one. Hand-edited files exist.

    ``object`` and not ``Any``: every field is narrowed before it is used, so a
    file somebody edited cannot put a list where a run needs a flag's value.
    """

    if not isinstance(record, dict):
        return None
    fields = cast("dict[str, object]", record)
    name = fields.get("name")
    options = fields.get("options")
    if not isinstance(name, str) or not name.strip() or not isinstance(options, dict):
        return None
    summary = fields.get("summary")
    return Preset(
        name=name.strip(),
        summary=summary if isinstance(summary, str) else "",
        options={
            str(key): _flag_value(value)
            for key, value in cast("dict[str, object]", options).items()
            if isinstance(value, (str, int, float, bool))
        },
    )


def _flag_value(value: str | int | float | bool) -> str:  # noqa: PYI041 — the JSON types
    """One stored value as the command line wants it.

    `True` rather than `true`: the form's checkboxes and the trainer's argparse
    both speak Python's spelling, and a preset written by hand in JSON's would
    otherwise arrive as the string "true" and be read as a change nobody made.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def all_presets(path: Path | None = None) -> list[Preset]:
    """Built-ins first, then the saved ones. What the picker lists."""

    return [*BUILTIN, *user_presets(path)]


def find(name: str, path: Path | None = None) -> Preset | None:
    """The preset called ``name``, built-in or saved."""

    wanted = name.strip()
    return next((preset for preset in all_presets(path) if preset.name == wanted), None)


def save(
    name: str, options: Mapping[str, str], *, summary: str = "", path: Path | None = None
) -> Preset:
    """Create ``name``, or replace the saved preset that already has it.

    Create and update are one operation on purpose: the dialog's *Save as…* over
    an existing name and its *Update* are the same act, and a store that
    distinguished them would only be able to fail in a second way.
    """

    cleaned = normalise(name)
    if cleaned in BUILTIN_NAMES:
        raise PresetError(f"'{cleaned}' is a built-in preset — save a copy under another name")
    preset = Preset(cleaned, summary, {str(k): str(v) for k, v in options.items()})
    kept = [existing for existing in user_presets(path) if existing.name != cleaned]
    _write([*kept, preset], path)
    return preset


def delete(name: str, path: Path | None = None) -> None:
    """Forget a saved preset. Built-ins and unknown names are refused."""

    cleaned = normalise(name)
    if cleaned in BUILTIN_NAMES:
        raise PresetError(f"'{cleaned}' ships with the trainer and cannot be deleted")
    kept = [existing for existing in user_presets(path) if existing.name != cleaned]
    if len(kept) == len(user_presets(path)):
        raise PresetError(f"no saved preset called '{cleaned}'")
    _write(kept, path)


def _write(presets: Iterable[Preset], path: Path | None) -> None:
    """Replace the file atomically — the trainer may be reading it to draw a list.

    Indented and newline-terminated because this file is meant to be opened in
    an editor and copied between machines.
    """

    file = _path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([preset.as_record() for preset in presets], indent=2) + "\n"
    temporary = file.with_name(f".{file.name}.new")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(file)
