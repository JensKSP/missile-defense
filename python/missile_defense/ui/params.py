# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The training loop's knobs, read out of missile_defense.training's source. No Qt, no torch.

Every hyperparameter already has its reasoning written beside it in
``TrainConfig`` and ``PPOConfig`` — that is the project's whole documentation
style. The parameter form shows those same sentences as tooltips, so the UI
teaches rather than presenting twenty unexplained boxes, and so the two can never
drift: there is one copy of the text and the form reads it.

Reading the *source* rather than importing the dataclasses is not stubbornness —
``missile_defense.training`` imports torch, and this package must not (docs/ROADMAP.md, M8, risk
3). `ast` gives the fields, their annotations and their defaults; the reasoning
lives in ``#:`` comments, which are comments and therefore not in the tree at
all, so `tokenize` collects those separately and they are matched back by line.

The price of reading source is that a default can be a *name*: the trainer writes
``aim_trail: float = CANONICAL_AIM_TRAIL`` rather than repeating ``0.84`` in two
places that could drift. So a name is followed to what it stands for, across
modules, before it is offered to a form — see :func:`_constant`.

If missile_defense.training is not beside the trainer — an installed trainer watching a synced
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

from ..runs import runconfig

#: Where the trainer's dataclasses live, for the tooltips and the defaults —
#: ``python/missile_defense``, beside the ``missile_defense/ui`` this file is in.
#: Here rather than in the window, because the library screen reads them too and
#: importing the window
#: from a panel inside it is a circle.
TRAINER_SOURCES = Path(__file__).resolve().parents[1]

#: The four that change a run's character. Everything else is behind *Advanced*
#: (docs/ROADMAP.md, M8, phase 3) — defaults that are good and reasoned should
#: not have to be re-read every time a run is started.
HEADLINE = ("envs", "steps", "updates", "learning_rate")

#: Not for the form: the trainer supplies the run directory itself, and resuming
#: from a checkpoint is a checkpoint-browser feature, not a text box.
HIDDEN = ("out_dir", "resume")

#: Where each group of knobs lives, in the order the form shows them.
#:
#: `Shaping` is the third group and the one with the sharpest teeth: its two
#: non-potential terms genuinely change what the policy converges to, where
#: everything in `phi` provably does not. Its flags are prefixed (`--reward-…`)
#: because `Shaping.gamma` and `PPOConfig.gamma` are two different discounts.
SOURCES = (
    ("training/train.py", "TrainConfig"),
    ("training/ppo.py", "PPOConfig"),
    ("sim/env.py", "Shaping"),
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
#: dropdown cannot be misspelled. The values are the ones
#: `missile_defense.training.ppo.build_policy`
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
    # The same floor as its starting value, and for the same reason: the two are
    # one decision (anneal *from* here *to* there) and the form offers them as
    # one control, so a pair that could not represent the same numbers would be a
    # control whose two ends disagree. 1e-8 is zero for any optimizer's purposes.
    "learning_rate_final": (1e-8, 1.0),
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
    # Added when the form gained sliders. A slider cannot exist without a range,
    # and these seven had none — so they were the fields that stayed plain boxes
    # for no reason other than that nobody had written a bound down.
    "aim_trail": (0.0, 1.0),  # a fraction of the way to the aim point
    "reaction_delay": (0, 60),  # ticks; a second of lag is already absurd
    "eval_ramp_until": (0, 100_000),
    "record_ramp_until": (0, 100_000),
    "auxiliary_coef": (0.0, 10.0),
    "seed": (0, 2_147_483_647),
    # `int | None`: blank means "the run's own --updates". The low bound stays 0
    # rather than 1 so an untouched spin box still reads as unset — `values()`
    # compares against what the editor was built showing, and a minimum of 1
    # would make every dialog emit a schedule nobody asked for.
    "schedule_updates": (0, 1_000_000),
}

#: How a slider should travel across a field's range.
#:
#: `linear` is the default and is not listed. The other two exist because a
#: linear slider is useless over a range that spans decades: `learning_rate` runs
#: 1e-8 to 1, so at 1000 steps every value below 0.001 sits in the first single
#: pixel. `decade` maps position to the exponent; `log` is for ranges that start
#: at a real zero, which cannot have a logarithm — the reward weights run 0 to
#: 10,000 and `ammo_weight` defaults to 5.0, so linear would put the default in
#: the first thousandth of the travel and make it unpickable.
SCALE: dict[str, str] = {
    "learning_rate": "decade",
    "learning_rate_final": "decade",
    "city_weight": "log",
    "ammo_weight": "log",
    "base_weight": "log",
    "waste_penalty": "log",
    "multikill_bonus": "log",
    "max_ticks": "log",
    "updates": "log",
    "envs": "log",
}


# ---- how the dialog is laid out ----------------------------------------------

#: The three questions a run is configured by, in tab order. Named for the
#: *decision* rather than for the dataclass each came from: someone starting a
#: run picks between "what is it paid for", "how does it learn" and "how big and
#: how long", and only afterwards cares that those happen to be `Shaping`,
#: `PPOConfig` and `TrainConfig`.
#:
#: The split is not quite by owner, and deliberately: the annealing schedule is
#: `learning_rate` from `PPOConfig` paired with `learning_rate_final` from
#: `TrainConfig`, one decision spread over two classes. It belongs where the
#: decision is made, which is Learning.
DOMAINS: tuple[tuple[str, str, str], ...] = (
    ("reward", "Objective", "what the agent is paid for"),
    ("learn", "Learning", "how it learns"),
    ("run", "Run", "how big, how long, what it costs"),
)


@dataclass(frozen=True)
class Group:
    """A row of the dialog: some fields, under a name, in one domain.

    ``essential`` groups sit open on the panel; the rest fold, showing their
    values on the closed summary line so folding compresses rather than hides.
    Nothing here holds more than five fields — the point of the grouping is that
    opening one costs a glance, which a single "13 more" drawer did not.
    """

    name: str
    domain: str
    fields: tuple[str, ...]
    essential: bool = False


#: Every settable field, in exactly one group. `test_ui_params.py` holds that
#: claim in both directions — a field the trainer gained but nobody placed would
#: otherwise simply not appear in the dialog, which is the failure this list is
#: most likely to produce and the hardest to notice.
GROUPS: tuple[Group, ...] = (
    # --- Objective: all seven, no folds. `enabled` is the master switch drawn
    # above the terms it governs rather than a row among them.
    Group("Shaping", "reward", ("enabled",), essential=True),
    Group(
        "Potential terms", "reward", ("base_weight", "city_weight", "ammo_weight"), essential=True
    ),
    Group("Priced events", "reward", ("waste_penalty", "multikill_bonus"), essential=True),
    # `Shaping.gamma` is not here: it is derived from `PPOConfig.gamma`, which
    # the two dataclasses require to be equal (see SHARED_FLAGS). One control,
    # in Learning, writes both.
    # --- Learning
    Group(
        "Learning",
        "learn",
        ("learning_rate", "gamma", "clip", "entropy_coef", "architecture", "minibatches"),
        essential=True,
    ),
    Group("Network size", "learn", ("hidden", "auxiliary_coef")),
    Group("Credit & value", "learn", ("gae_lambda", "value_coef", "value_clip")),
    Group("Update stability", "learn", ("epochs", "max_grad_norm")),
    Group(
        "Annealing",
        "learn",
        ("learning_rate_final", "entropy_coef_final", "schedule_updates"),
    ),
    # --- Run
    Group("Scale", "run", ("envs", "steps", "updates"), essential=True),
    Group("Episode & pacing", "run", ("frame_skip", "max_ticks")),
    Group(
        "What gets written",
        "run",
        ("eval_every", "record_every", "checkpoint_every", "eval_ramp_until", "record_ramp_until"),
    ),
    Group("Human handicap", "run", ("aim_trail", "reaction_delay")),
    Group("Machine", "run", ("device", "seed")),
)

#: Fields the dialog derives rather than offers, mapped to the field they follow.
#: `Shaping.gamma` must equal `PPOConfig.gamma` — both dataclasses say so, and
#: the shaping-invariance proof assumes it — so offering two controls would be
#: offering a way to break it. `flags_for` already writes both flags from the one
#: value; this is what keeps the second out of the form.
DERIVED = {"Shaping": ("gamma",)}


def group_of(name: str, owner: str = "") -> Group | None:
    """The group a field belongs to, or ``None`` when it is derived or unknown."""
    if name in DERIVED.get(owner, ()):
        return None
    for group in GROUPS:
        if name in group.fields:
            return group
    return None


def domain_title(domain: str) -> str:
    return next((title for key, title, _ in DOMAINS if key == domain), domain)


#: The jargon the dialog uses, defined in the game's terms rather than the
#: literature's. The parameter form shows these where a word is marked, so an
#: abbreviation is never a dead end — and the same text is the whole glossary.
#: Every one of these has to be *used* somewhere, which a test checks: a
#: definition nobody can reach is a definition nobody maintains.
GLOSSARY: dict[str, str] = {
    "PPO": (
        "Proximal Policy Optimization — the training algorithm. It improves the "
        "policy in small, clipped steps, so one bad batch cannot wreck a network "
        "that was working."
    ),
    "GAE": (
        "Generalized Advantage Estimation — how credit for a score is spread back "
        "over the decisions that led to it. Lambda trades bias against variance."
    ),
    "gamma": (
        "Discount. How much a future point is worth against one now. 1/(1-gamma) "
        "is roughly how many steps ahead the agent cares about: 0.999 is about "
        "1000 steps, or 65 seconds of play."
    ),
    "potential": (
        "One number summarising how good the board looks, built from surviving "
        "batteries, cities and ammo. Shaping pays the agent the *change* in it, "
        "so progress is rewarded as it happens instead of at the end of a wave."
    ),
    "shaping": (
        "Extra reward paid on top of the game score, so progress is visible "
        "sooner. Switch it off and the agent is paid the game score and nothing "
        "else. Built as the difference of a potential, which Ng, Harada & Russell "
        "(1999) proved cannot change which policy is best — only how fast it is "
        "found — so runs differing only in these weights stay comparable."
    ),
    "entropy": (
        "How undecided the policy is. A bonus on it keeps the agent exploring; "
        "with none it commits early and stops improving. Missile Command punishes "
        "early commitment, so some is kept alive well into training."
    ),
    "clip": (
        "PPO's trust region. Caps how far one update may move the policy from the "
        "previous one; 0.2 is the standard starting point."
    ),
    "mlp": (
        "Multi-layer perceptron — a plain flat network. Cheap, and "
        "checkpoint-compatible with existing models."
    ),
    "entity": (
        "The relational network. It encodes each threat separately and attends "
        "across them, which reads the board far better and costs about twenty "
        "times the GPU memory per sample."
    ),
    "rollout": (
        "One batch of experience: every parallel environment stepped forward N "
        "times before the network is updated from what happened."
    ),
    "minibatches": (
        "A rollout is split into this many pieces, each updating the network in "
        "turn. More pieces means less memory at once — same data, smaller bites."
    ),
    "checkpoint": (
        "A saved snapshot — weights, optimizer state and iteration — that a run "
        "can be continued from."
    ),
    "policy": (
        "The trained network itself: the thing that looks at the board and decides where to shoot."
    ),
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

    @property
    def scale(self) -> str:
        """How a slider should travel this field's range: linear, log or decade."""
        return SCALE.get(self.name, "linear")

    @property
    def group(self) -> Group | None:
        """Where the dialog puts this field, or ``None`` when it is derived."""
        return group_of(self.name, self.owner)

    @property
    def derived(self) -> bool:
        """Whether the dialog computes this field instead of offering it.

        True for `Shaping.gamma`, which follows `PPOConfig.gamma` because the two
        must be equal. A derived field still reaches the command line — see
        `flags_for` — it simply has no editor of its own to disagree with.
        """
        return self.name in DERIVED.get(self.owner, ())

    @property
    def key(self) -> tuple[str, str]:
        """What identifies this field to the dialog.

        The owner *and* the name, because `gamma` is a field of two config
        classes and the form used to key its editors by name alone — so the two
        shared one box by accident, and only the second was ever written.
        """
        return (self.owner, self.name)


def read_params(package_dir: Path) -> list[Param]:
    """Every settable field of the trainer's two config dataclasses.

    ``package_dir`` is the root of the ``missile_defense`` package; :data:`SOURCES`
    names the files inside it. They sit in different layers now — ``TrainConfig``
    under ``training/`` and ``Shaping`` under ``sim/`` — so what is passed down
    from here is each file's *own* directory, which is what a relative import in
    it resolves against.
    """
    found: list[Param] = []
    for filename, class_name in SOURCES:
        path = package_dir / filename
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue  # no missile_defense.training beside this trainer; the form says so
        found.extend(_fields_of(source, class_name, path.parent))
    return found


def _fields_of(source: str, class_name: str, here: Path) -> list[Param]:
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
            default = _default_of(statement.value, tree, here)
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


def _default_of(value: ast.expr | None, tree: ast.Module, here: Path) -> str:
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
        found = _constant(value.id, tree, here, frozenset())
        if found is not None:
            return found
    return ast.unparse(value)


def _constant(name: str, tree: ast.Module, here: Path, seen: frozenset[str]) -> str | None:
    """What ``name`` stands for, or ``None`` if it cannot be followed to a value.

    Two ways a module can answer: it binds the name itself, or it imported it
    from somewhere else — and the real chain uses both, since
    ``missile_defense.sim.benchmark`` binds ``CANONICAL_AIM_TRAIL`` to a name it
    imported from ``missile_defense.sim.protocol``.

    ``here`` is the directory of the module being read, because that is what a
    relative import resolves against. It used to be the package root and the two
    were the same thing, which stopped being true when the package gained layers:
    the chain now starts in ``training/train.py`` and reaches ``sim/benchmark.py``
    through a ``from ..sim.benchmark import`` — two levels up and back down.

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
    for level, module, original in _imports_of(tree, name):
        source = _relative_module(here, level, module)
        if str(source) in seen:
            continue  # a cycle between two modules is not a reason to stop working
        sibling = _parsed(source)
        if sibling is None:
            continue
        found = _constant(original, sibling, source.parent, seen | {str(source)})
        if found is not None:
            return found
    return None


def _relative_module(here: Path, level: int, module: str) -> Path:
    """The file a ``from <dots><module> import`` in ``here`` names.

    One dot is this directory, two is the one above it, and so on — the same
    arithmetic the interpreter does. Dotted module parts become directories, so
    ``from ..sim.benchmark import`` from inside ``training/`` lands on
    ``sim/benchmark.py``.
    """
    base = here
    for _ in range(level - 1):
        base = base.parent
    return base.joinpath(*module.split(".")).with_suffix(".py")


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


def _imports_of(tree: ast.Module, name: str) -> Iterator[tuple[int, str, str]]:
    """Where this module imports ``name`` from, as (level, module, its name there).

    Relative single-name imports only — ``from .benchmark import X`` or
    ``from ..sim.benchmark import X``. An absolute import is a package this
    trainer cannot assume is beside it, and a ``from . import benchmark`` binds a
    module rather than a value.

    Any level, not just one. It was level one only, which was right while the
    package was flat and silently stopped resolving anything the day it was not:
    every cross-layer default would have fallen back to showing the constant's
    *name* in the form instead of its value.
    """
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) == name:
                yield node.level, node.module, alias.name


def _parsed(path: Path) -> ast.Module | None:
    """``path`` as a syntax tree, or ``None`` when it is missing or unparseable.

    Neither is worth raising over: this runs to make a form's default prettier,
    and a trainer that refuses to open a dialog because a module it was only
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
    #: trainer has no missile_defense.training beside it to read, or the field is not one.
    default: str
    changed: bool
    help: str


def settings_of(config: runconfig.RunConfig | None, fields: Sequence[Param] = ()) -> list[Setting]:
    """Everything a run recorded about itself, in the order the trainer wrote it.

    Driven by the *stored* file rather than by the field list: a run trained by a
    newer missile_defense.training carries knobs this trainer has never heard of, and hiding them
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


#: The one field name that belongs to two config classes at once, and what a
#: form setting it has to write.
#:
#: `PPOConfig.gamma` discounts the return, `Shaping.gamma` discounts the
#: potential, and `Shaping`'s own docstring is explicit that the invariance
#: proof — the entire reason shaping is safe to add — assumes the two are equal.
#: So the form offers *one* discount and it reaches both. That is also what the
#: dialog has always physically done: its editors are keyed by field name, so
#: the two fields have only ever had one box between them.
SHARED_FLAGS = {"gamma": ("--gamma", "--reward-gamma")}


def flags_for(name: str) -> tuple[str, ...]:
    """Every command-line flag one form field has to write. Usually exactly one.

    Derived from `SOURCES` rather than from the field name alone, so the one
    place that knows a group is prefixed is the same place that declares it.

    This was `flag_for`, singular, and resolved a name through `REWARD_FIELDS`
    alone. `gamma` is in that set, so *both* discounts came back as
    `--reward-gamma` and `--gamma` could not be produced at all: the PPO
    discount was unreachable from the trainer, and a run whose discount was set
    here trained with the two halves disagreeing. Silently — neither side
    checks, and the only symptom is that a shaping term everybody believes is
    optimality-neutral quietly is not.

    `Param.flag` was right about this all along, because it asks the owner. The
    two answers disagreeing is what let the bug sit behind a passing test.
    """
    if name in SHARED_FLAGS:
        return SHARED_FLAGS[name]
    prefix = PREFIXES["Shaping"] if name in REWARD_FIELDS else ""
    return (f"--{prefix}{name.replace('_', '-')}",)


def command_line(
    python: str,
    values: dict[str, str],
    *,
    out_dir: Path,
    resume: Path | None = None,
    module: str = "missile_defense.training",
) -> list[str]:
    """The command a configured run is started with.

    Only the values that were *changed* appear: a defaulted field is left to the
    dataclass, so the command line reads as the diff from the defaults rather
    than as a wall of restated numbers. It is shown in the dialog for the same
    reason — the trainer should leave you able to start the same run without it.

    ``resume`` comes last, and from a picker rather than the generic form: it is
    a *file that exists*, so offering it as a text box would be offering a way
    to mistype a path.
    """
    command = [python, "-u", "-m", module]
    for name, value in values.items():
        # `flags_for`, not a local `--` + name: `Shaping`'s flags are prefixed
        # (`--reward-city-weight`), and rebuilding them here would emit
        # `--city-weight`, which the trainer rejects — a Start button that
        # produces an unparseable command line.
        #
        # Plural because one form field can owe two flags: the discount belongs
        # to both config classes and has to be written to both, or the run
        # trains with them disagreeing.
        for flag in flags_for(name):
            command += [flag, value]
    command += ["--out-dir", str(out_dir)]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command
