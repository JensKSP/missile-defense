# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The trainer's knobs, read out of the trainer's source. No Qt, no torch.

Every hyperparameter already has its reasoning written beside it in
``TrainConfig`` and ``PPOConfig`` — that is the project's whole documentation
style. The parameter form shows those same sentences as tooltips, so the UI
teaches rather than presenting twenty unexplained boxes, and so the two can never
drift: there is one copy of the text and the form reads it.

Reading the *source* rather than importing the dataclasses is not stubbornness —
``md.train`` imports torch, and this package must not (docs/ROADMAP.md, M8, risk
3). `ast` gives the fields, their annotations and their defaults; the reasoning
lives in ``#:`` comments, which are comments and therefore not in the tree at
all, so `tokenize` collects those separately and they are matched back by line.

If the trainer is not beside the console — an installed console watching a synced
directory — nothing here throws. There are simply no fields, and the form says
so.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path

#: The four that change a run's character. Everything else is behind *Advanced*
#: (docs/ROADMAP.md, M8, phase 3) — defaults that are good and reasoned should
#: not have to be re-read every time a run is started.
HEADLINE = ("envs", "steps", "updates", "learning_rate")

#: Not for the form: the console supplies the run directory itself, and resuming
#: from a checkpoint is a checkpoint-browser feature, not a text box.
HIDDEN = ("out_dir", "resume")

#: Where each group of knobs lives, in the order the form shows them.
SOURCES = (("train.py", "TrainConfig"), ("ppo.py", "PPOConfig"))


@dataclass(frozen=True)
class Param:
    """One field of a config dataclass, with the reasoning written beside it."""

    name: str
    kind: str  #: "int", "float" or "text" — which editor the form gives it
    default: str  #: as source, e.g. "1024"; empty when the default is None
    help: str
    owner: str  #: the dataclass it came from, for grouping

    @property
    def flag(self) -> str:
        """``entropy_coef`` → ``--entropy-coef``."""
        return "--" + self.name.replace("_", "-")

    @property
    def headline(self) -> bool:
        return self.name in HEADLINE


def read_params(package_dir: Path) -> list[Param]:
    """Every settable field of the trainer's two config dataclasses."""
    found: list[Param] = []
    for filename, class_name in SOURCES:
        try:
            source = (package_dir / filename).read_text(encoding="utf-8")
        except OSError:
            continue  # no trainer beside this console; the form says so
        found.extend(_fields_of(source, class_name))
    return found


def _fields_of(source: str, class_name: str) -> list[Param]:
    comments = _doc_comments(source)
    tree = ast.parse(source)
    fields: list[Param] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(
                statement.target, ast.Name
            ):
                continue
            name = statement.target.id
            if name in HIDDEN:
                continue
            default = ast.unparse(statement.value) if statement.value is not None else ""
            fields.append(
                Param(
                    name=name,
                    kind=_kind(ast.unparse(statement.annotation)),
                    default="" if default == "None" else default,
                    help=_help_above(comments, statement.lineno),
                    owner=class_name,
                )
            )
    return fields


def _kind(annotation: str) -> str:
    """Which editor a field gets. Anything unrecognised is free text."""
    if annotation.startswith("int"):
        return "int"
    if annotation.startswith("float"):
        return "float"
    return "text"


def _doc_comments(source: str) -> dict[int, str]:
    """Line number → the text of the ``#:`` comment on it, if any."""
    found: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and token.string.startswith("#:"):
            found[token.start[0]] = token.string[2:].strip()
    return found


def _help_above(comments: dict[int, str], line: int) -> str:
    """The run of ``#:`` comments immediately above a field, as one sentence."""
    lines: list[str] = []
    cursor = line - 1
    while cursor in comments:
        lines.append(comments[cursor])
        cursor -= 1
    return " ".join(reversed(lines))


def command_line(
    python: str,
    values: dict[str, str],
    *,
    out_dir: Path,
    resume: Path | None = None,
    module: str = "md.train",
) -> list[str]:
    """The command a configured run is started with.

    Only the values that were *changed* appear: a defaulted field is left to the
    dataclass, so the command line reads as the diff from the defaults rather
    than as a wall of restated numbers. It is shown in the dialog for the same
    reason — the console should leave you able to start the same run without it.

    ``resume`` comes last, and from a picker rather than the generic form: it is
    a *file that exists*, so offering it as a text box would be offering a way
    to mistype a path.
    """
    command = [python, "-u", "-m", module]
    for name, value in values.items():
        command += [f"--{name.replace('_', '-')}", value]
    command += ["--out-dir", str(out_dir)]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command
