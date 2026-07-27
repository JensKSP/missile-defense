# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The reward formula the console shows, checked without a display.

What is worth pinning here is not the wording but the distinctions: that a run
which switched shaping off does not get an equation implying otherwise, that a
zero-weighted term stays visible instead of vanishing, and that the two priced
events are marked as changing the objective while the potential terms are not.
Those are the claims a reader will act on.
"""

from __future__ import annotations

from md.ui.params import Setting
from md.ui.reward import INVARIANT, OBJECTIVE, UNSHAPED, formula_of


def _setting(name: str, value: str, group: str = "shaping") -> Setting:
    return Setting(group=group, name=name, value=value, default="", changed=False, help="")


def _defaults(**overrides: str) -> list[Setting]:
    values = {
        "city_weight": "100.0",
        "ammo_weight": "5.0",
        "base_weight": "200.0",
        "gamma": "0.999",
        "enabled": "True",
        "waste_penalty": "0.0",
        "multikill_bonus": "0.0",
    }
    values.update(overrides)
    return [_setting(name, value) for name, value in values.items()]


def test_a_run_without_shaping_settings_has_no_formula() -> None:
    # An older run, or a directory with no config.json at all. Nothing to draw,
    # and inventing defaults would be showing a formula that run never used.
    assert formula_of([]) is None
    assert formula_of([_setting("lr", "0.0003", group="ppo")]) is None


def test_the_potential_reads_as_the_equation_it_is() -> None:
    formula = formula_of(_defaults())
    assert formula is not None
    # Trailing `.0` is noise in an equation; `0.999` is not.
    assert formula.phi == "φ(s) = 200 × batteries  +  100 × cities  +  5 × ammo"
    assert formula.gamma == "0.999"
    assert formula.total == "r′ = score  +  γ·φ(s′) − φ(s)"


def test_terms_priced_at_nothing_stay_visible() -> None:
    """A missing line would answer "was this penalised?" with silence."""
    formula = formula_of(_defaults())
    assert formula is not None
    priced = {term.name: term for term in formula.priced}
    assert set(priced) == {"waste_penalty", "multikill_bonus"}
    assert not any(term.active for term in priced.values())
    # …and every term carries its reason, since that is the whole point of
    # showing the formula rather than the seven numbers.
    assert all(term.why and term.gist for term in (*formula.potential, *formula.priced))


def test_each_term_says_why_it_is_there_rather_than_repeating_the_equation() -> None:
    """The line under φ(s) has to add something; φ(s) already gave the arithmetic."""
    formula = formula_of(_defaults())
    assert formula is not None
    lines = {term.name: term.line for term in formula.potential}
    assert (
        lines["base_weight"] == "200 × batteries — a third of your firepower, until the next wave"
    )
    # An off term keeps its row and says so on it.
    off = next(t for t in formula.priced if t.name == "multikill_bonus")
    assert off.line.startswith("0 × multi-kills  (off) — ")


def test_a_switched_on_penalty_enters_the_equation_with_its_sign() -> None:
    formula = formula_of(_defaults(waste_penalty="15.0"))
    assert formula is not None
    assert formula.total == "r′ = score  +  γ·φ(s′) − φ(s)  − 15 × wasted shots"
    assert OBJECTIVE in formula.notes


def test_a_multikill_bonus_is_added_rather_than_subtracted() -> None:
    formula = formula_of(_defaults(multikill_bonus="25.0"))
    assert formula is not None
    assert formula.total.endswith("+ 25 × multi-kills")


def test_defaults_claim_comparability_and_nothing_more() -> None:
    """With both priced events off, runs differ only in potential terms."""
    formula = formula_of(_defaults())
    assert formula is not None
    assert formula.notes == [INVARIANT]


def test_shaping_off_says_so_instead_of_showing_a_potential() -> None:
    formula = formula_of(_defaults(enabled="False"))
    assert formula is not None
    assert not formula.shaped
    assert formula.notes == [UNSHAPED]
    # The weights are still listed — they were stored — but none of them applied.
    assert not any(term.active for term in (*formula.potential, *formula.priced))


def test_a_run_missing_a_newer_weight_simply_has_fewer_terms() -> None:
    """`base_weight` was added late; runs from before it must still render."""
    stored = [s for s in _defaults() if s.name != "base_weight"]
    formula = formula_of(stored)
    assert formula is not None
    assert [term.name for term in formula.potential] == ["city_weight", "ammo_weight"]
    assert formula.phi == "φ(s) = 100 × cities  +  5 × ammo"
