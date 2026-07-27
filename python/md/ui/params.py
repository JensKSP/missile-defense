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

The price of reading source is that a default can be a *name*: the trainer writes
``aim_trail: float = CANONICAL_AIM_TRAIL`` rather than repeating ``0.84`` in two
places that could drift. So a name is followed to what it stands for, across
modules, before it is offered to a form — see :func:`_constant`.

If the trainer is not beside the console — an installed console watching a synced
directory — nothing here throws. There are simply no fields, and the form says
so.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import runconfig

#: Where the trainer's dataclasses live, for the tooltips and the defaults —
#: ``python/md``, beside the ``md/ui`` this file is in. Here rather than in the
#: window, because the library screen reads them too and importing the window
#: from a panel inside it is a circle.
TRAINER_SOURCES = Path(__file__).resolve().parents[1]

#: The four that change a run's character. Everything else is behind *Advanced*
#: (docs/ROADMAP.md, M8, phase 3) — defaults that are good and reasoned should
#: not have to be re-read every time a run is started.
HEADLINE = ("envs", "steps", "updates", "learning_rate")

#: Not for the form: the console supplies the run directory itself, and resuming
#: from a checkpoint is a checkpoint-browser feature, not a text box.
HIDDEN = ("out_dir", "resume")

#: Where each group of knobs lives, in the order the form shows them.
#:
#: `Shaping` is the third group and the one with the sharpest teeth: its two
#: non-potential terms genuinely change what the policy converges to, where
#: everything in `phi` provably does not. Its flags are prefixed (`--reward-…`)
#: because `Shaping.gamma` and `PPOConfig.gamma` are two different discounts.
SOURCES = (
    ("train.py", "TrainConfig"),
    ("ppo.py", "PPOConfig"),
    ("env.py", "Shaping"),
)

#: Config classes whose flags the trainer prefixes, and with what.
PREFIXES = {"Shaping": "reward-"}

#: The prefixed fields, by name, so `flag_for` can answer without re-reading the
#: trainer's source at every call. Stated rather than derived, and held to the
#: real dataclass by `test_ui_params.py` — a name that drifts out of this list
#: would produce a Start button emitting a flag the trainer rejects.
REWARD_FIELDS = frozenset(
    {
        "city_weight",
        "ammo_weight",
        "base_weight",
        "gamma",
        "enabled",
        "waste_penalty",
        "multikill_bonus",
    }
)

#: Fields that are a *choice* rather than a number or free text, and what the
#: choices are. Typing `--architecture entty` into a text box costs a run; a
#: dropdown cannot be misspelled. The values are the ones `md.ppo.build_policy`
#: accepts, and a test holds the two lists together.
#: `device` is deliberately *not* here: it is `str | None`, `cuda:1` is a
#: legitimate value, and a dropdown would take that away to prevent a mistake
#: nobody makes.
CHOICES: dict[str, tuple[str, ...]] = {
    "architecture": ("mlp", "entity"),
}

#: Sane bounds, as (minimum, maximum). Not the *possible* range — the range in
#: which a value is a training decision rather than a typo. `--envs 0` cannot
#: run, a negative learning rate maximises loss, and `--entropy-coef 500` is a
#: uniform policy with extra steps. A spin box that refuses them turns three
#: wasted hours into a value that will not enter.
#:
#: Deliberately generous at the top: the point is to catch a slipped decimal
#: place, not to have an opinion about what someone is experimenting with.
BOUNDS: dict[str, tuple[float, float]] = {
    "envs": (1, 65_536),
    "steps": (1, 8_192),
    "updates": (1, 1_000_000),
    "frame_skip": (1, 60),
    "max_ticks": (60, 10_000_000),
    "eval_every": (0, 100_000),
    "record_every": (0, 100_000),
    "checkpoint_every": (1, 100_000),
    "hidden": (8, 8_192),
    "epochs": (1, 64),
    "minibatches": (1, 512),
    "learning_rate": (1e-8, 1.0),
    "learning_rate_final": (0.0, 1.0),
    "entropy_coef": (0.0, 1.0),
    "entropy_coef_final": (0.0, 1.0),
    "value_coef": (0.0, 10.0),
    "clip": (1e-4, 1.0),
    "value_clip": (1e-4, 10.0),
    "max_grad_norm": (1e-4, 100.0),
    "gamma": (0.0, 1.0),
    "gae_lambda": (0.0, 1.0),
    # Reward weights. Wide, because what a city is worth relative to a shot is
    # exactly the thing worth experimenting with — but not negative, which would
    # pay the agent to lose one.
    "city_weight": (0.0, 10_000.0),
    "ammo_weight": (0.0, 10_000.0),
    "base_weight": (0.0, 10_000.0),
    "waste_penalty": (0.0, 1_000.0),
    "multikill_bonus": (0.0, 1_000.0),
}


@dataclass(frozen=True)
class Param:
    """One field of a config dataclass, with the reasoning written beside it."""

    name: str
    kind: str  #: "int", "float", "bool", "choice" or "text"
    default: str  #: as source, e.g. "1024"; empty when the default is None
    help: str
    owner: str  #: the dataclass it came from, for grouping

    @property
    def flag(self) -> str:
        """``entropy_coef`` → ``--entropy-coef``, with the owner's prefix."""
        prefix = PREFIXES.get(self.owner, "")
        return f"--{prefix}{self.name.replace('_', '-')}"

    @property
    def prefixed(self) -> bool:
        return self.owner in PREFIXES

    @property
    def choices(self) -> tuple[str, ...]:
        """The values this field may take, or empty when it is a free number."""
        return CHOICES.get(self.name, ())

    @property
    def bounds(self) -> tuple[float, float] | None:
        """The range in which this value is a decision rather than a typo."""
        return BOUNDS.get(self.name)

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
        found.extend(_fields_of(source, class_name, package_dir))
    return found


def _fields_of(source: str, class_name: str, package_dir: Path) -> list[Param]:
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
            default = _default_of(statement.value, tree, package_dir)
            fields.append(
                Param(
                    name=name,
                    kind=_kind(ast.unparse(statement.annotation), name),
                    default="" if default == "None" else default,
                    help=_help_above(comments, statement.lineno),
                    owner=class_name,
                )
            )
    return fields


# ---- defaults that are names -------------------------------------------------

#: How many times a name is followed before this gives up. The real chain is
#: three links across three modules — ``aim_trail`` is ``CANONICAL_AIM_TRAIL`` is
#: ``AIM_TRAIL`` is ``0.84`` — and the bound is here so a constant that (wrongly)
#: names itself is a field with an odd default rather than a form that hangs.
NAME_HOPS = 8


def _default_of(value: ast.expr | None, tree: ast.Module, package_dir: Path) -> str:
    """A field's default as source text, with a bare name followed to its value.

    The trainer names its constants rather than writing ``0.84`` in the two
    places that would then have to agree, so ``aim_trail: float =
    CANONICAL_AIM_TRAIL`` is the shape this has to cope with. Handing the *name*
    to the form is what a spin box cannot do anything with: ``int(...)`` on it
    raised out of a Qt slot, where an exception is printed and stepped over
    rather than being fatal — so Start silently did nothing, for ever, with the
    explanation on a terminal nobody was looking at.

    Anything that is not a bare name is left exactly as written. A default is a
    literal in every other case, and guessing at an expression would be inventing
    a value the trainer never had.
    """
    if value is None:
        return ""
    if isinstance(value, ast.Name):
        found = _constant(value.id, tree, package_dir, frozenset())
        if found is not None:
            return found
    return ast.unparse(value)


def _constant(name: str, tree: ast.Module, package_dir: Path, seen: frozenset[str]) -> str | None:
    """What ``name`` stands for, or ``None`` if it cannot be followed to a value.

    Two ways a module can answer: it binds the name itself, or it imported it
    from a sibling — and the real chain uses both, since ``md.benchmark`` binds
    ``CANONICAL_AIM_TRAIL`` to a name it imported from ``md._protocol``.

    Deliberately lazy. Only a default that *is* a name asks anything of this, so
    a trainer that spells all of its defaults out costs nothing, and one that
    does not costs the two sibling modules actually involved rather than a walk
    of the whole package.
    """
    bindings = _bindings(tree)
    for _ in range(NAME_HOPS):
        bound = bindings.get(name)
        if bound is None:
            break
        if not isinstance(bound, ast.Name):
            return ast.unparse(bound)
        name = bound.id
    for module, original in _imports_of(tree, name):
        if module in seen:
            continue  # a cycle between two modules is not a reason to stop working
        sibling = _parsed(package_dir / f"{module}.py")
        if sibling is None:
            continue
        found = _constant(original, sibling, package_dir, seen | {module})
        if found is not None:
            return found
    return None


def _bindings(tree: ast.Module) -> dict[str, ast.expr]:
    """Module-level ``NAME = <literal or name>``, and nothing more adventurous.

    A default that resolves to a call or a dict would be a value this form could
    not show and must not invent, so those are simply not collected: the name
    survives to the form as the trainer wrote it.
    """
    found: dict[str, ast.expr] = {}
    for node in tree.body:
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        else:
            continue
        value = node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant | ast.Name):
            found[target.id] = value
    return found


def _imports_of(tree: ast.Module, name: str) -> Iterator[tuple[str, str]]:
    """Sibling modules this one imports ``name`` from, as (module, its name there).

    Relative single-name imports only — ``from .benchmark import X``. An absolute
    import is a package this console cannot assume is beside it, and a
    ``from . import benchmark`` binds a module rather than a value.
    """
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) == name:
                yield node.module, alias.name


def _parsed(path: Path) -> ast.Module | None:
    """``path`` as a syntax tree, or ``None`` when it is missing or unparseable.

    Neither is worth raising over: this runs to make a form's default prettier,
    and a console that refuses to open a dialog because a module it was only
    curious about has a syntax error has made things worse.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


def _kind(annotation: str, name: str = "") -> str:
    """Which editor a field gets. Anything unrecognised is free text."""
    if name in CHOICES:
        return "choice"
    if annotation.startswith("bool"):
        return "bool"
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


# ---- what a run was actually started with ------------------------------------

#: Which `config.json` group each config dataclass was written into, so a stored
#: setting can be paired with the field — and the reasoning — it came from.
GROUP_OF = {"TrainConfig": "train", "PPOConfig": "ppo", "Shaping": "shaping"}


@dataclass(frozen=True)
class Setting:
    """One knob of a run that has already been started.

    The stored value, the default it was chosen against, and whether those two
    differ — which is the question a parameter view is really asked. Twenty-six
    numbers are unreadable; the four that this run *changed* are the run.
    """

    group: str
    name: str
    value: str
    #: The trainer's own default, as its source spells it. Empty when this
    #: console has no trainer beside it to read, or the field is not one.
    default: str
    changed: bool
    help: str


def settings_of(config: runconfig.RunConfig | None, fields: Sequence[Param] = ()) -> list[Setting]:
    """Everything a run recorded about itself, in the order the trainer wrote it.

    Driven by the *stored* file rather than by the field list: a run trained by a
    newer trainer carries knobs this console has never heard of, and hiding them
    would be answering "what was this trained with?" with "the part I recognise".
    Those simply arrive without a default or a tooltip.
    """
    if config is None:
        return []
    known = {(GROUP_OF.get(field.owner, ""), field.name): field for field in fields}
    settings: list[Setting] = []
    for group, values in config.payload.items():
        for name, value in values.items():
            field = known.get((group, name))
            text = runconfig.format_value(value)
            settings.append(
                Setting(
                    group=group,
                    name=name,
                    value=text,
                    default=field.default if field else "",
                    changed=field is not None and not same_value(text, field.default),
                    help=field.help if field else "",
                )
            )
    return settings


def same_value(stored: str, default: str) -> bool:
    """Whether two spellings of a setting mean the same thing.

    Textually where they are text, numerically where they are numbers: the
    trainer's source says ``3.0e-4`` and its own `config.json` says ``0.0003``,
    and a view that called that a change would mark every run as having altered
    the learning rate.
    """
    if stored == default:
        return True
    try:
        return float(stored) == float(default)
    except ValueError:
        return False


def flag_for(name: str) -> str:
    """The command-line flag for a config field, prefix and all.

    Derived from `SOURCES` rather than from the field name alone, so the one
    place that knows a group is prefixed is the same place that declares it.
    """
    prefix = PREFIXES["Shaping"] if name in REWARD_FIELDS else ""
    return f"--{prefix}{name.replace('_', '-')}"


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
        # `flag_for`, not a local `--` + name: `Shaping`'s flags are prefixed
        # (`--reward-city-weight`), and rebuilding them here would emit
        # `--city-weight`, which the trainer rejects — a Start button that
        # produces an unparseable command line.
        command += [flag_for(name), value]
    command += ["--out-dir", str(out_dir)]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command
