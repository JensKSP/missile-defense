# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Two agents, one seed, one screen — end to end through the real binary.

The chain this covers is the whole point of a match and every link in it was
built separately: a league model, an episode recorded on a *named* seed, a
manifest that pairs two of them, and a game that plays the manifest back as one
synchronized screen. Any one of those working in isolation proves nothing —
the failure this suite exists to catch is a pair that loads but shows two
unrelated episodes, and it is invisible to a unit test of either half.

The refusals matter as much as the success. A match that quietly showed one
side, or two different seeds, would be a comparison the viewer had no way to
distrust — so the binary is expected to *exit* rather than open a window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from md import league, policy_format, tournament

from .harness import assert_clean, needs_app, needs_display, run_app

pytestmark = [pytest.mark.e2e, needs_app, needs_display]

#: Enough ticks for a few waves, few enough that two episodes are seconds rather
#: than minutes. A random policy dies early anyway; this is the backstop.
MAX_TICKS = 4_000

#: Small on purpose: the forward pass runs once per decision in NumPy, and what
#: is being tested is the plumbing, not the network.
HIDDEN = 8


def _shapes() -> tuple[int, int]:
    """The observation width and action count this build actually uses.

    Read from a live environment rather than hard-coded: both have changed
    during development (`blast_features` 4 -> 5 moved the observation from 1895
    to 1959), and a fixture that pins them turns a spec change into a confusing
    failure here instead of a clear one where the spec lives.
    """
    from md.env import VecEnv

    env = VecEnv(num_envs=1, max_ticks=64)
    return int(env.observations.shape[1]), int(env.action_masks().shape[1])


def _policy(seed: int, observation_size: int, action_count: int) -> policy_format.NativePolicy:
    rng = np.random.default_rng(seed)

    def normal(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    return policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=observation_size,
        action_count=action_count,
        architecture="mlp",
        tensors=(
            policy_format.Tensor(
                "trunk.0.weight", (HIDDEN, observation_size), normal(HIDDEN, observation_size)
            ),
            policy_format.Tensor("trunk.0.bias", (HIDDEN,), normal(HIDDEN)),
            policy_format.Tensor("trunk.2.weight", (HIDDEN, HIDDEN), normal(HIDDEN, HIDDEN)),
            policy_format.Tensor("trunk.2.bias", (HIDDEN,), normal(HIDDEN)),
            policy_format.Tensor(
                "policy_head.weight", (action_count, HIDDEN), normal(action_count, HIDDEN)
            ),
            policy_format.Tensor("policy_head.bias", (action_count,), normal(action_count)),
            policy_format.Tensor("value_head.weight", (1, HIDDEN), normal(1, HIDDEN)),
            policy_format.Tensor("value_head.bias", (1,), normal(1)),
        ),
        metadata={},
    )


def _model(root: Path, model_id: str, name: str, seed: int) -> league.Model:
    observation_size, action_count = _shapes()
    directory = root / model_id
    directory.mkdir(parents=True, exist_ok=True)
    policy_format.write(
        directory / league.POLICY_NAME, _policy(seed, observation_size, action_count)
    )
    (directory / league.CARD_NAME).write_text(json.dumps({"display_name": name}), encoding="utf-8")
    model = league.find(model_id, root)
    assert model is not None
    return model


@pytest.fixture(scope="module")
def paired(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two recordings of the same seed and the manifest that pairs them."""
    root = tmp_path_factory.mktemp("match")
    left = _model(root / "league", "aaaa", "Amber Anvil", 1)
    right = _model(root / "league", "bbbb", "Brisk Harbour", 2)

    seed = 4242
    recordings = {
        "left": tournament.record_episode(left, seed, root / "left.mdr", max_ticks=MAX_TICKS),
        "right": tournament.record_episode(right, seed, root / "right.mdr", max_ticks=MAX_TICKS),
    }
    manifest = root / "match.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "seeds": [seed],
                "left": {
                    "model_id": "aaaa",
                    "display_name": "Amber Anvil",
                    "mean_score": 51000.0,
                    # Relative, which is what `write_manifest` should also
                    # produce: a match directory has to survive being moved.
                    "recording": recordings["left"].name,
                },
                "right": {
                    "model_id": "bbbb",
                    "display_name": "Brisk Harbour",
                    "mean_score": 47000.0,
                    "recording": recordings["right"].name,
                },
                "ranked": True,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_a_manifest_plays_as_one_split_screen(paired: Path, tmp_path: Path) -> None:
    run = run_app("--match", str(paired), frames=180, sandbox=tmp_path)
    assert_clean(run)
    # Its own screen, and its own mode: a match is not a replay wearing two hats,
    # and `--report` is where an automated check can tell the difference.
    assert run.state == "match", run.report
    assert run.mode == "match", run.report


def test_two_recordings_pair_without_a_manifest(paired: Path, tmp_path: Path) -> None:
    # The ad-hoc path: two episodes exist and nothing wrote a tournament record.
    left = paired.parent / "left.mdr"
    right = paired.parent / "right.mdr"
    run = run_app(
        "--match-left", str(left), "--match-right", str(right), frames=180, sandbox=tmp_path
    )
    assert_clean(run)
    assert run.state == "match", run.report


def test_half_a_pairing_is_refused(paired: Path, tmp_path: Path) -> None:
    # Not a default, not a silent single-side view: the two flags are one option.
    run = run_app(
        "--match-left",
        str(paired.parent / "left.mdr"),
        frames=60,
        sandbox=tmp_path,
        expect_report=False,
    )
    assert run.exit_code == 2, run.stderr
    assert "both sides" in run.stderr


def test_two_seeds_are_refused_rather_than_shown(paired: Path, tmp_path: Path) -> None:
    """The failure this whole feature has to be safe against.

    Two agents on two different problems, drawn side by side, is not a
    comparison — and it looks exactly like one.
    """
    observation_size, action_count = _shapes()
    other = paired.parent / "league" / "cccc"
    other.mkdir(parents=True, exist_ok=True)
    policy_format.write(other / league.POLICY_NAME, _policy(3, observation_size, action_count))
    model = league.find("cccc", paired.parent / "league")
    assert model is not None
    elsewhere = tournament.record_episode(
        model, 9999, paired.parent / "other.mdr", max_ticks=MAX_TICKS
    )

    run = run_app(
        "--match-left",
        str(paired.parent / "left.mdr"),
        "--match-right",
        str(elsewhere),
        frames=60,
        sandbox=tmp_path,
        expect_report=False,
    )
    assert run.exit_code == 2, run.stdout
    assert "same" in run.stderr


def test_a_missing_side_says_which_one(paired: Path, tmp_path: Path) -> None:
    broken = paired.parent / "broken.json"
    payload = json.loads(paired.read_text(encoding="utf-8"))
    payload["right"]["recording"] = "gone.mdr"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    run = run_app("--match", str(broken), frames=60, sandbox=tmp_path, expect_report=False)
    assert run.exit_code == 2
    # With two files in play, "could not read the recording" leaves a person
    # with no idea which one is bad.
    assert "right" in run.stderr
    assert "gone.mdr" in run.stderr
