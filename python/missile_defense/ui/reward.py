# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The reward a run was actually trained against, written out as its formula.

`missile_defense.sim.env.Shaping` is seven numbers, and the config panel shows them as seven rows
of a table. That is faithful and nearly useless: the rows do not say that three
of them are summed into a potential, that one is a discount applied to that
potential and not to the return, or that two of them are switched off. Somebody
reading the table has to reconstruct the equation from the docstring of a class
they are not looking at.

So this renders the equation instead, with the run's own numbers substituted, and
marks the one distinction that changes how a result should be read:

* **potential terms** (`base_weight`, `city_weight`, `ammo_weight`, `gamma`) are
  potential-based shaping in the sense of Ng, Harada & Russell (1999), so they
  provably cannot change which policy is optimal — only how fast it is found. A
  run that changed these is comparable with one that did not.
* **priced events** (`waste_penalty`, `multikill_bonus`) are not. They change the
  objective, so two runs that differ here were trained for different things and
  their scores answer different questions.

No Qt in here on purpose: the arithmetic and the wording are what deserve tests,
and `test_ui_reward.py` gets at them without a display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .params import Setting

#: The field names this module knows how to place in the equation. A run trained
#: by a newer trainer may carry shaping knobs that are not here; they stay in the
#: table, which is the honest place for "recorded but not understood".
POTENTIAL_FIELDS = ("base_weight", "city_weight", "ammo_weight")
PRICED_FIELDS = ("waste_penalty", "multikill_bonus")

#: What each weight multiplies, in the words the game uses rather than the field
#: name. `base_weight * live_bases` reads as "200 x batteries" on screen.
COUNTED = {
    "base_weight": "batteries",
    "city_weight": "cities",
    "ammo_weight": "ammo",
    "waste_penalty": "wasted shots",
    "multikill_bonus": "multi-kills",
}

#: One clause per term, shown on the term's own line. Short enough to sit beside
#: the arithmetic without wrapping, because a line that repeats `200 × batteries`
#: and says nothing else is noise — the equation above it already said that.
GIST = {
    "base_weight": "a third of your firepower, until the next wave",
    "city_weight": "what the end-of-wave bonus already pays",
    "ammo_weight": "so an unspent interceptor is worth something now",
    "waste_penalty": "a blast that destroyed nothing",
    "multikill_bonus": "one blast, several warheads",
}

#: The whole reasoning, as a tooltip. Deliberately about consequence in the game
#: rather than about the code.
WHY = {
    "base_weight": (
        "Losing a battery costs a third of your firepower until the next wave. "
        "Priced above a city on purpose: protecting the guns is what protects "
        "the cities."
    ),
    "city_weight": (
        "The same weight the end-of-wave bonus already pays, so shaping only "
        "delivers it the moment it is earned instead of at the wave boundary."
    ),
    "ammo_weight": (
        "Also the end-of-wave rate. Makes an unspent interceptor worth "
        "something now, which is what stops the agent emptying its magazines."
    ),
    "waste_penalty": (
        "Charged when a blast expires having destroyed nothing. Changes the "
        "objective, so a run that sets it is not comparable with one that "
        "does not."
    ),
    "multikill_bonus": (
        "Paid when one blast catches several warheads. The score already pays "
        "for this, and it is the one term that rewards waiting — which an "
        "agent can overlearn into holding fire while the cities burn."
    ),
}

INVARIANT = (
    "Potential terms cannot change which policy is best (Ng, Harada & Russell 1999) "
    "— only how quickly it is found. Runs that differ only here stay comparable."
)
OBJECTIVE = (
    "Priced events do change what the agent is trying to do. Two runs that differ "
    "here were trained for different things."
)
UNSHAPED = "Shaping is off for this run: the agent was paid the game score and nothing else."


@dataclass(frozen=True)
class Term:
    """One weighted quantity in the reward, as it should appear on screen."""

    name: str  #: the config field, so the table row and this line can be matched up
    weight: str  #: the run's own value, spelled as it was stored
    counts: str  #: what the weight multiplies, in the game's vocabulary
    gist: str  #: the one clause shown beside it
    why: str  #: the whole reasoning, for the tooltip
    #: Zero-weighted terms are drawn muted rather than hidden. A reader asking
    #: "was this run penalised for wasted shots?" is asking about a term that is
    #: not there, and an absent line answers that with silence.
    active: bool

    @property
    def line(self) -> str:
        """The term as one readable row: the arithmetic, then why it is there."""
        text = f"{self.weight} × {self.counts}"
        if not self.active:
            text += "  (off)"
        return f"{text} — {self.gist}" if self.gist else text


@dataclass(frozen=True)
class Formula:
    """The reward of one run: the equation, its terms, and what they imply."""

    shaped: bool
    gamma: str
    #: Required rather than defaulted: every construction site has both lists in
    #: hand, and an empty default would let a caller build a `Formula` that
    #: renders as a reward paying for nothing.
    potential: list[Term]
    priced: list[Term]

    @property
    def phi(self) -> str:
        """`phi(s)`, with numbers — the potential the shaping term differences."""
        if not self.potential:
            return ""
        return "φ(s) = " + "  +  ".join(f"{t.weight} × {t.counts}" for t in self.potential)

    @property
    def total(self) -> str:
        """The whole reward, priced events included when a run switched any on.

        Unshaped, that is the game score and nothing else — `VecEnv.step` nests
        the potential *and* both priced events inside `if shaping.enabled`. This
        used to write the shaped equation regardless and leave the contradiction
        to the note underneath, so a run that was paid the bare score was shown a
        formula full of terms it never saw, with "shaping is off" as a footnote.
        """
        if not self.shaped:
            return "r′ = score"
        line = "r′ = score  +  γ·φ(s′) − φ(s)"
        for term in self.priced:
            if not term.active:
                continue
            sign = "−" if term.name == "waste_penalty" else "+"
            line += f"  {sign} {term.weight} × {term.counts}"
        return line

    @property
    def notes(self) -> list[str]:
        if not self.shaped:
            return [UNSHAPED]
        notes = [INVARIANT]
        if any(term.active for term in self.priced):
            notes.append(OBJECTIVE)
        return notes


def formula_of(settings: Sequence[Setting]) -> Formula | None:
    """The reward formula of a run, or `None` when it recorded no shaping.

    Driven by the stored settings rather than by `Shaping`'s defaults, because
    the question this answers is what *this* run was paid for. A run from before
    a field existed simply has fewer terms.
    """
    stored = {s.name: s for s in settings if s.group == "shaping"}
    if not stored:
        return None

    enabled = stored.get("enabled")
    shaped = enabled is None or _truthy(enabled.value)

    def terms(names: Sequence[str]) -> list[Term]:
        out: list[Term] = []
        for name in names:
            setting = stored.get(name)
            if setting is None:
                continue
            out.append(
                Term(
                    name=name,
                    weight=_number(setting.value),
                    counts=COUNTED.get(name, name.replace("_", " ")),
                    gist=GIST.get(name, ""),
                    why=WHY.get(name, setting.help),
                    active=shaped and _nonzero(setting.value),
                )
            )
        return out

    gamma = stored.get("gamma")
    return Formula(
        shaped=shaped,
        gamma=_number(gamma.value) if gamma else "",
        potential=terms(POTENTIAL_FIELDS),
        priced=terms(PRICED_FIELDS),
    )


def formula_being_configured(values: Mapping[str, str]) -> Formula | None:
    """The same formula, for a run that has not started yet.

    :func:`formula_of` answers "what was this run paid for?" from a stored
    ``config.json``. This answers "what *would* it be paid?" from whatever the
    parameter dialog currently shows — the identical arithmetic and the identical
    wording, so the equation somebody reads while choosing the weights is the one
    they will read afterwards in the run's own panel.

    Worth having as its own entry point rather than a Qt-side rebuild: the terms,
    their order, which are potential and which are priced, and the sentence
    beside each are decisions this module already makes, and a second copy in the
    dialog would be a second place for them to drift.
    """
    settings = [
        Setting(group="shaping", name=name, value=value, default="", changed=False, help="")
        for name, value in values.items()
    ]
    return formula_of(settings)


def _number(text: str) -> str:
    """A stored value without the noise: `200.0` is `200`, `0.999` stays."""
    try:
        value = float(text)
    except ValueError:
        return text
    return str(int(value)) if value == int(value) else text


def _nonzero(text: str) -> bool:
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def _truthy(text: str) -> bool:
    return text.strip().lower() not in {"false", "0", "no", ""}
