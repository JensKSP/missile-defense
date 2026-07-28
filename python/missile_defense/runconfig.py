# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What a run was started with, read back. No Qt, no torch.

The trainer writes ``runs/config.json`` at start-up so that six months later the
checkpoints are still there and the shell history is not. Writing it was only
half the job: until this module there was nothing that *read* it, so the answer
to "what was this run trained with?" was a file you opened in an editor, and
continuing a run meant retyping its flags from that file by hand.

Three callers, one parser:

* :mod:`missile_defense.train` fills in the settings a ``--resume`` must inherit, and prints
  the resolved configuration at start-up.
* The trainer's parameter view shows a run's knobs beside its curves.
* The trainer's Start dialog pours them into the form when it continues a run.

Neither Qt nor torch, for the usual reason (docs/ROADMAP.md, M8, risk 3): the
trainer must be able to read this and the training loop must not have to import a
window to write it. And nothing here raises — a `config.json` is a text file in
a directory a person can point anything at, and a run whose settings cannot be
read is still a run whose curves can be.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

#: Beside ``model.json`` in the run directory. One says what the run was started
#: with, the other says what it is training.
FILENAME = "config.json"

#: The sections, in the order they are shown and in the order a person asks
#: about them: the shape of the run, the optimiser, what the agent is paid for,
#: and how the two annealed coefficients move.
GROUPS = ("train", "ppo", "shaping", "schedule")

#: Settings that are *this run's*, not the recipe's, so a form or a command line
#: rebuilt from a config must not restate them: the output directory is supplied
#: by whoever is starting the run, and the checkpoint to continue from is a file
#: that exists and therefore a picker rather than a remembered path.
NOT_A_RECIPE = ("out_dir", "resume")

#: What a setting that was never given is shown as. Not blank: ``device=`` reads
#: as a line that got cut off, where ``device=none`` reads as a decision.
NOTHING = "none"

#: Groups whose fields are settable flags. ``schedule`` is not one of them: it is
#: *derived* from the four annealing endpoints in ``train`` and ``ppo``, and the
#: trainer restores it from the checkpoint rather than from a flag.
SETTABLE = ("train", "ppo", "shaping")


@dataclass(frozen=True)
class RunConfig:
    """One run's ``config.json``, split into the groups the trainer wrote."""

    path: Path
    train: Mapping[str, object]
    ppo: Mapping[str, object]
    shaping: Mapping[str, object]
    schedule: Mapping[str, object]

    def group(self, name: str) -> Mapping[str, object]:
        """One section by name, empty when the file did not carry it."""
        return cast("Mapping[str, object]", getattr(self, name, {}))

    @property
    def payload(self) -> dict[str, Mapping[str, object]]:
        """Every section, in :data:`GROUPS` order — what :func:`describe` takes."""
        return {name: self.group(name) for name in GROUPS}

    @property
    def out_dir(self) -> Path | None:
        """Where the run actually went. The trainer records the resolved path."""
        value = self.train.get("out_dir")
        return Path(str(value)) if value else None

    @property
    def resumed_from(self) -> Path | None:
        """The checkpoint this run continued, when it continued one.

        Worth asking of any run: its ``updates`` is then a count of *additional*
        updates, and its first update number is somebody else's last plus one.
        """
        value = self.train.get("resume")
        return Path(str(value)) if value else None


def read(where: Path) -> RunConfig | None:
    """Read a run's configuration. ``None`` when there is not one to read.

    ``where`` is a run directory or the file itself, because both are things a
    caller has in hand — the trainer holds a run directory, and a command line
    may well be given the path that was printed to it.
    """
    path = where / FILENAME if where.is_dir() else where
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    # `object`, not `Any`: every value is narrowed by `_section` before anything
    # sees it, which is what keeps a hand-edited file from reaching a caller as
    # something with the wrong shape.
    sections: dict[str, object] = {str(key): value for key, value in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]
    return RunConfig(
        path=path,
        train=_section(sections.get("train")),
        ppo=_section(sections.get("ppo")),
        shaping=_section(sections.get("shaping")),
        schedule=_section(sections.get("schedule")),
    )


def _section(value: object) -> dict[str, object]:
    """One group, narrowed. Anything that is not a mapping is no group at all."""
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]


def options(config: RunConfig | None) -> dict[str, str]:
    """The settable fields as ``{name: text}`` — the shape a form and a preset use.

    Deliberately the same shape :mod:`missile_defense.presets` stores and
    :func:`missile_defense.ui.params.command_line` consumes, so a run's own settings can be
    poured into the Start dialog exactly as a preset is, and come back out as
    flags. Values that were never set are left out rather than written as
    ``"None"``: ``device: null`` means "decide at run time", and a form that
    turned it into ``--device None`` would name a device nobody has.

    The namespace is flat, which is the trainer's own flag namespace, so
    ``Shaping.gamma`` and ``PPOConfig.gamma`` collide here as they do there. It
    resolves the same way — ``shaping`` is applied last, and the flat name
    belongs to ``--reward-gamma`` (:data:`missile_defense.ui.params.REWARD_FIELDS`).
    """
    if config is None:
        return {}
    found: dict[str, str] = {}
    for group in SETTABLE:
        for name, value in config.group(group).items():
            if name in NOT_A_RECIPE or value is None:
                continue
            found[name] = format_value(value)
    return found


def format_value(value: object) -> str:
    """One setting as text, the way the trainer's own flags spell it.

    ``True``/``False`` rather than ``1``/``0``, because that is what the
    trainer's boolean flag parser and the dialog's check boxes both read, and
    plain ``str`` for the numbers so ``1e-05`` stays ``1e-05``.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value)


def describe(
    payload: Mapping[str, Mapping[str, object]], *, width: int = 96, indent: str = "  "
) -> list[str]:
    """Every group as printable lines: a label column, then ``name=value`` pairs.

    For a terminal and for the log the trainer tees into the run directory
    (:mod:`missile_defense.runlog`), which is what puts the same block in the trainer's log
    pane. Wrapped rather than truncated — a setting that scrolled off the right
    of the window is a setting nobody can check — and continuation lines line up
    under the values, so the label column stays readable as a column.
    """
    groups = [(name, values) for name, values in payload.items() if values]
    if not groups:
        return []
    label_width = max(len(name) for name, _ in groups) + 2
    lines: list[str] = []
    for name, values in groups:
        label = f"{indent}{name.ljust(label_width)}"
        # A setting that was never given still gets a word. `device=` reads as a
        # truncated line; `device=none` reads as "this run did not pin one",
        # which is the fact.
        body = "  ".join(f"{key}={format_value(value) or NOTHING}" for key, value in values.items())
        lines.extend(
            textwrap.wrap(
                body,
                width=max(width, len(label) + 20),
                initial_indent=label,
                subsequent_indent=" " * len(label),
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return lines
