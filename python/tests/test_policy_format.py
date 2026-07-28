# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The `.mdp` file, from the side that writes it.

Two languages have to agree about these bytes, and only one of them is here —
`agent/tests/unit/test_policy.cpp` reads the same fixture from C++. So the tests
below are about the *container*: that a policy survives a round trip unchanged,
and that every way a file can be wrong is refused rather than half-read.

**The rejections are the point.** A learned policy is a file a person may have
downloaded, and the format's whole reason for existing is that `.pt` is a pickle
and running one is running its author's code. A reader that trusts its input
gives that property back, so each malformed case here is a promise: a truncated
download, a flipped bit, a hand-edited manifest and a mismatched architecture
all raise, and none of them reads past the end of a buffer on the way.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from missile_defense import policy_format

OBS = 6
ACTIONS = 4
HIDDEN = 5


def fixture_policy() -> policy_format.NativePolicy:
    """A two-layer MLP with values that are distinctive rather than random.

    `arange`-derived, so a byte-order or stride mistake shows up as obviously
    wrong numbers instead of as plausible noise that still round-trips.
    """
    rng = np.random.default_rng(20260726)

    def weights(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.1).astype(np.float32)

    def bias(size: int) -> np.ndarray:
        return (np.arange(size, dtype=np.float32) * 0.01) - 0.02

    return policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=OBS,
        action_count=ACTIONS,
        architecture="mlp",
        tensors=(
            policy_format.Tensor("trunk.0.weight", (HIDDEN, OBS), weights(HIDDEN, OBS)),
            policy_format.Tensor("trunk.0.bias", (HIDDEN,), bias(HIDDEN)),
            policy_format.Tensor("trunk.2.weight", (HIDDEN, HIDDEN), weights(HIDDEN, HIDDEN)),
            policy_format.Tensor("trunk.2.bias", (HIDDEN,), bias(HIDDEN)),
            policy_format.Tensor("policy_head.weight", (ACTIONS, HIDDEN), weights(ACTIONS, HIDDEN)),
            policy_format.Tensor("policy_head.bias", (ACTIONS,), bias(ACTIONS)),
            policy_format.Tensor("value_head.weight", (1, HIDDEN), weights(1, HIDDEN)),
            policy_format.Tensor("value_head.bias", (1,), bias(1)),
        ),
        metadata={"display_name": "Fixture", "run": "test", "trained_updates": 800},
    )


#: A deliberately tiny `entity`, whose dimensions are all different from each
#: other. Equal extents are how a transposed weight or a swapped encoder slips
#: through a shape check, so nothing here shares a size with anything else it
#: could be confused for.
E_WIDTH = 3
E_HIDDEN = 4
E_THREAT_FEATURES = 2
E_INTERCEPTOR_FEATURES = 5
E_BLAST_FEATURES = 6
E_BATTERIES = 2
E_THREATS = 3
E_GLOBALS = 7
E_CONTEXT_INPUT = E_GLOBALS + (2 * E_WIDTH)
E_RELATION_INPUT = 4 * E_WIDTH
E_OBS = 40
E_ACTIONS = 1 + (E_BATTERIES * E_THREATS)


def entity_tensors() -> list[policy_format.Tensor]:
    """Every tensor `entity` names, shaped as `ARCHITECTURES` says it must be."""
    rng = np.random.default_rng(20260727)

    def make(name: str, *shape: int) -> policy_format.Tensor:
        values = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        return policy_format.Tensor(name, shape, values)

    def encoder(prefix: str, features: int) -> list[policy_format.Tensor]:
        return [
            make(f"{prefix}.0.weight", E_WIDTH, features),
            make(f"{prefix}.0.bias", E_WIDTH),
            make(f"{prefix}.2.weight", E_WIDTH, E_WIDTH),
            make(f"{prefix}.2.bias", E_WIDTH),
        ]

    def attention(prefix: str) -> list[policy_format.Tensor]:
        return [
            make(f"{prefix}.query.weight", E_WIDTH, E_WIDTH),
            make(f"{prefix}.key.weight", E_WIDTH, E_WIDTH),
            make(f"{prefix}.value.weight", E_WIDTH, E_WIDTH),
            make(f"{prefix}.output.weight", E_WIDTH, E_WIDTH),
            make(f"{prefix}.output.bias", E_WIDTH),
        ]

    return [
        *encoder("threat_encoder", E_THREAT_FEATURES),
        *encoder("interceptor_encoder", E_INTERCEPTOR_FEATURES),
        *encoder("blast_encoder", E_BLAST_FEATURES),
        *attention("interceptor_attention"),
        *attention("blast_attention"),
        make("actor_context.0.weight", E_HIDDEN, E_CONTEXT_INPUT),
        make("actor_context.0.bias", E_HIDDEN),
        make("actor_context.2.weight", E_HIDDEN, E_HIDDEN),
        make("actor_context.2.bias", E_HIDDEN),
        make("context_to_threat.weight", E_WIDTH, E_HIDDEN),
        make("context_to_threat.bias", E_WIDTH),
        make("relation.0.weight", E_WIDTH, E_RELATION_INPUT),
        make("relation.0.bias", E_WIDTH),
        make("relation.2.weight", E_WIDTH, E_WIDTH),
        make("relation.2.bias", E_WIDTH),
        make("fire_head.weight", E_BATTERIES, E_WIDTH),
        make("fire_head.bias", E_BATTERIES),
        make("noop_head.weight", 1, E_HIDDEN),
        make("noop_head.bias", 1),
        make("critic_trunk.0.weight", E_HIDDEN, E_OBS),
        make("critic_trunk.0.bias", E_HIDDEN),
        make("critic_trunk.2.weight", E_HIDDEN, E_HIDDEN),
        make("critic_trunk.2.bias", E_HIDDEN),
        make("value_head.weight", 1, E_HIDDEN),
        make("value_head.bias", 1),
    ]


def entity_policy(
    tensors: list[policy_format.Tensor] | None = None,
) -> policy_format.NativePolicy:
    return policy_format.NativePolicy(
        schema=policy_format.SCHEMA,
        observation_size=E_OBS,
        action_count=E_ACTIONS,
        architecture="entity",
        tensors=tuple(entity_tensors() if tensors is None else tensors),
        metadata={"display_name": "Entity Fixture"},
    )


def test_an_entity_policy_round_trips(tmp_path: Path) -> None:
    """The relational architecture survives the container unchanged."""
    policy = entity_policy()
    assert policy_format.read(policy_format.write(tmp_path / "e.mdp", policy)) == policy


def test_entity_does_not_carry_the_training_only_auxiliary_head() -> None:
    """It is never evaluated on a player's machine, so it is not in the file."""
    named = [name for name, _ in policy_format.ARCHITECTURES["entity"]]
    assert not [name for name in named if name.startswith("auxiliary_head")]


def test_entity_rejects_widths_that_chain_but_do_not_add_up() -> None:
    """`relation.0` takes four width-sized blocks; three would read the wrong ones.

    Every individual shape here is still self-consistent, which is exactly why
    name-equality alone cannot catch it and `_DERIVED` has to.
    """
    tensors = entity_tensors()
    swapped = [
        policy_format.Tensor(
            t.name,
            (E_WIDTH, 3 * E_WIDTH),
            np.zeros((E_WIDTH, 3 * E_WIDTH), dtype=np.float32),
        )
        if t.name == "relation.0.weight"
        else t
        for t in tensors
    ]
    with pytest.raises(policy_format.PolicyFormatError, match="do not add up"):
        policy_format.validate(entity_policy(swapped))


# ---- the round trip ----------------------------------------------------------


def test_policy_round_trip(tmp_path: Path) -> None:
    policy = fixture_policy()
    path = policy_format.write(tmp_path / "policy.mdp", policy)
    assert policy_format.read(path) == policy


def test_the_values_survive_exactly_and_not_approximately(tmp_path: Path) -> None:
    """float32 in, float32 out, bit for bit.

    Not a nicety: the C++ side asserts action-for-action parity with the Python
    policy on a fixed seed, and a format that rounded anywhere would make that
    assertion flaky in a way that took days to attribute.
    """
    policy = fixture_policy()
    back = policy_format.read(policy_format.write(tmp_path / "p.mdp", policy))
    for mine, theirs in zip(policy.tensors, back.tensors, strict=True):
        assert mine.values.dtype == theirs.values.dtype == np.float32
        assert np.array_equal(mine.values, theirs.values)


def test_the_file_starts_with_the_magic_and_is_not_a_pickle(tmp_path: Path) -> None:
    """Identifiable from its first bytes, and containing no executable anything.

    The reason this format exists at all: `.pt` is a pickle, and loading one is
    running whatever its author put in it. An `.mdp` is a header, some JSON and
    a block of little-endian floats — there is nothing in it to execute.
    """
    raw = (policy_format.write(tmp_path / "p.mdp", fixture_policy())).read_bytes()
    assert raw.startswith(policy_format.MAGIC)
    for pickled in (b"cnumpy", b"__reduce__", b"pickle", b"\x80\x04\x95"):
        assert pickled not in raw


def test_metadata_travels_with_the_weights(tmp_path: Path) -> None:
    """Which is what lets the game name the agent on screen (Task 3 Step 4b).

    A path is not a name and `policy-best.pt` says nothing about which run
    produced it, so the display name lives here, in the file.
    """
    back = policy_format.read(policy_format.write(tmp_path / "p.mdp", fixture_policy()))
    assert back.metadata["display_name"] == "Fixture"
    assert back.metadata["trained_updates"] == 800


# ---- the rejections ----------------------------------------------------------

Mutation = Callable[[bytes], bytes]


def _split(raw: bytes) -> tuple[dict[str, object], bytes]:
    """A written file back into its manifest and its payload."""
    (length,) = struct.unpack_from("<I", raw, len(policy_format.MAGIC) + 4)
    start = len(policy_format.MAGIC) + 8
    manifest = json.loads(raw[start : start + length].decode("utf-8"))
    return manifest, raw[start + length :]


def _rebuild(manifest: dict[str, object], payload: bytes) -> bytes:
    """...and back into a file, without re-deriving the checksum.

    Deliberately not going through :func:`policy_format.write`: these tests are
    about files that `write` would never produce, which is exactly the set a
    reader has to survive.
    """
    encoded = json.dumps(manifest).encode("utf-8")
    return (
        policy_format.MAGIC
        + struct.pack("<I", policy_format.CONTAINER_VERSION)
        + struct.pack("<I", len(encoded))
        + encoded
        + payload
    )


def unknown_schema(raw: bytes) -> bytes:
    manifest, payload = _split(raw)
    manifest["schema"] = policy_format.SCHEMA + 7
    return _rebuild(manifest, payload)


def truncate_payload(raw: bytes) -> bytes:
    return raw[:-64]


def duplicate_tensor(raw: bytes) -> bytes:
    manifest, payload = _split(raw)
    tensors = list(manifest["tensors"])  # type: ignore[arg-type]
    tensors[1] = {**tensors[1], "name": tensors[0]["name"]}
    manifest["tensors"] = tensors
    return _rebuild(manifest, payload)


def wrong_dimensions(raw: bytes) -> bytes:
    """A shape that no longer matches the observation size it claims."""
    manifest, payload = _split(raw)
    manifest["observation_size"] = OBS + 1
    return _rebuild(manifest, payload)


def corrupt_checksum(raw: bytes) -> bytes:
    manifest, payload = _split(raw)
    return _rebuild(manifest, bytes([payload[0] ^ 0xFF]) + payload[1:])


def non_finite_weight(raw: bytes) -> bytes:
    manifest, payload = _split(raw)
    values = bytearray(payload)
    values[0:4] = struct.pack("<f", float("nan"))
    manifest["checksum"] = policy_format.checksum(bytes(values))
    return _rebuild(manifest, bytes(values))


def offset_out_of_bounds(raw: bytes) -> bytes:
    """The one that would be a buffer overrun in the C++ reader."""
    manifest, payload = _split(raw)
    tensors = list(manifest["tensors"])  # type: ignore[arg-type]
    tensors[-1] = {**tensors[-1], "offset": len(payload) + 1024}
    manifest["tensors"] = tensors
    return _rebuild(manifest, payload)


def bad_magic(raw: bytes) -> bytes:
    return b"NOTAPOLI" + raw[8:]


def missing_tensor(raw: bytes) -> bytes:
    """An MLP without its value head is not an MLP."""
    manifest, payload = _split(raw)
    manifest["tensors"] = list(manifest["tensors"])[:-1]  # type: ignore[arg-type]
    return _rebuild(manifest, payload)


def unknown_architecture(raw: bytes) -> bytes:
    manifest, payload = _split(raw)
    manifest["architecture"] = "transformer"
    return _rebuild(manifest, payload)


@pytest.mark.parametrize(
    "mutation",
    [
        bad_magic,
        unknown_schema,
        truncate_payload,
        duplicate_tensor,
        wrong_dimensions,
        corrupt_checksum,
        non_finite_weight,
        offset_out_of_bounds,
        missing_tensor,
        unknown_architecture,
    ],
    ids=lambda m: m.__name__,
)
def test_policy_rejects_invalid_payload(tmp_path: Path, mutation: Mutation) -> None:
    written = policy_format.write(tmp_path / "good.mdp", fixture_policy())
    broken = tmp_path / "broken.mdp"
    broken.write_bytes(mutation(written.read_bytes()))
    with pytest.raises(policy_format.PolicyFormatError):
        policy_format.read(broken)


def test_a_rejection_says_which_file_and_what_was_wrong(tmp_path: Path) -> None:
    """An error a person meets when a download went wrong, so it has to explain.

    "PolicyFormatError" alone sends them to the source; the filename and the
    failed check together are usually the whole diagnosis.
    """
    written = policy_format.write(tmp_path / "good.mdp", fixture_policy())
    broken = tmp_path / "broken.mdp"
    broken.write_bytes(truncate_payload(written.read_bytes()))
    with pytest.raises(policy_format.PolicyFormatError) as raised:
        policy_format.read(broken)
    assert "broken.mdp" in str(raised.value)
    assert "checksum" in str(raised.value) or "truncated" in str(raised.value)


def test_an_empty_or_absent_file_is_a_format_error_not_an_os_error(tmp_path: Path) -> None:
    """Both are things a half-finished download leaves behind."""
    empty = tmp_path / "empty.mdp"
    empty.write_bytes(b"")
    with pytest.raises(policy_format.PolicyFormatError):
        policy_format.read(empty)
    with pytest.raises(policy_format.PolicyFormatError):
        policy_format.read(tmp_path / "nothing-here.mdp")


# ---- writing safely ----------------------------------------------------------


def test_writing_refuses_a_policy_it_could_not_read_back(tmp_path: Path) -> None:
    """Validation on the way *out*, so a bad file is never produced at all.

    Cheaper than every reader having to be the last line of defence, and it
    means a promotion that would have shipped a broken model fails at the point
    where someone can still do something about it.
    """
    policy = fixture_policy()
    broken = policy_format.NativePolicy(
        schema=policy.schema,
        observation_size=policy.observation_size,
        action_count=policy.action_count,
        architecture=policy.architecture,
        tensors=(
            *policy.tensors[:-1],
            policy_format.Tensor("value_head.bias", (1,), np.array([np.inf], np.float32)),
        ),
        metadata=policy.metadata,
    )
    with pytest.raises(policy_format.PolicyFormatError):
        policy_format.write(tmp_path / "never.mdp", broken)
    assert not (tmp_path / "never.mdp").exists()


def test_writing_replaces_the_destination_only_once_it_is_whole(tmp_path: Path) -> None:
    """A crash mid-write must not leave a half-file where a good one was.

    The league promotes by writing into place, so the failure this prevents is a
    model that was fine yesterday being unreadable because a disk filled up.
    """
    destination = tmp_path / "policy.mdp"
    original = policy_format.write(destination, fixture_policy())
    before = original.read_bytes()

    broken = policy_format.NativePolicy(
        schema=fixture_policy().schema,
        observation_size=OBS,
        action_count=ACTIONS,
        architecture="nonsense",
        tensors=fixture_policy().tensors,
        metadata={},
    )
    with pytest.raises(policy_format.PolicyFormatError):
        policy_format.write(destination, broken)
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp*")), "a temporary file was left behind"
