# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""``.mdp`` — a learned policy as data, readable from C++ without a Python in sight.

**Why this exists rather than shipping the `.pt`.** A PyTorch checkpoint is a
pickle, and loading one executes whatever its author put in it. That is tolerable
for a file you trained yourself on your own machine and unacceptable for one the
game loads out of an install directory, or one a person downloaded because
somebody said it was good. An `.mdp` is a magic number, a JSON manifest and a
block of little-endian float32: there is nothing in it to execute, and the reader
that refuses a malformed one is thirty lines rather than a sandbox.

The second reason is simpler. The game is C++ with no Python anywhere in it —
that is the packaging promise `debian/control` keeps — so the *only* format the
game could load is one that does not need torch to read. `.pt` is never an import
format here; it is what training writes and what :mod:`md.export_policy` converts.

## The layout

    magic            8 bytes, ``MDPOLICY``
    container        uint32 LE — how to parse the rest of *this* header
    manifest length  uint32 LE
    manifest         UTF-8 JSON, exactly that many bytes
    payload          the tensors, back to back, at the offsets the manifest gives

Two version numbers, because they answer different questions. ``container`` is
how to *parse the file* and has never changed; ``schema`` is what the numbers
*mean* — the observation encoding and action space the policy was trained
against — and moves whenever `md::encode` does. A reader that understands the
container can always read the manifest far enough to say "this policy is for a
different simulation", which is a much better failure than a parse error.

## The compatibility promise

* **Data only.** No code, no pickles, no references to Python types. A conforming
  reader never has to trust the file to avoid running something.
* **Little-endian float32**, explicitly, everywhere. Not native order: these files
  travel between machines, and "it worked on mine" is not a format.
* **Every tensor's offset and length are in the manifest and are bounds-checked
  against the payload before a byte is read.** The C++ reader has no other way to
  be safe, and the Python one does the same so that the two agree about which
  files are valid.
* **SHA-256 over the payload**, in the manifest. A flipped bit in a weight is
  otherwise a policy that plays slightly worse and nobody ever finds out.
* **Tensor order is fixed by the architecture, not by the file.** A reader looks
  tensors up by name and checks the set is exactly right; a file may not add,
  drop or rename one and still claim the architecture.
* **Unknown metadata keys are preserved and ignored.** That is the extension
  point — a display name, a provenance note, a canonical score — and it never
  affects how the weights are read.

Adding an architecture means adding an entry to :data:`ARCHITECTURES` here and
the matching forward pass in ``agent/src/policy.cpp``. Both sides then reject
what the other cannot run, which is the point of naming it in the file at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

#: First eight bytes of every `.mdp`. Long enough that `file` can be taught it
#: and that a truncated download is not mistaken for a short valid one.
MAGIC = b"MDPOLICY"

#: How to parse the header. Bumped only if the framing above changes, which is
#: not the same event as the numbers meaning something new.
CONTAINER_VERSION = 1

#: What the numbers *mean*: the observation encoding and the action space. This
#: moves when `md::encode` does, and a policy from a different schema is not
#: wrong — it is for a different game, and must be refused rather than run.
SCHEMA = 1

#: The one numeric type in the payload. Named as a numpy dtype string so the
#: byte order is in the file rather than in the reader's assumptions.
DTYPE = "<f4"
#: The same thing as a dtype object, for the calls that take one. Built once so
#: the string above stays the single statement of what is in the payload.
LITTLE_ENDIAN_F32 = np.dtype(np.float32).newbyteorder("<")
ITEMSIZE = 4

#: What each architecture's tensors are called and how their shapes chain, as
#: (name, (dimension names…)). The dimension names are resolved against each
#: other while reading, which is what turns "the shapes are plausible" into "the
#: shapes are the ones this observation size and action count imply".
#:
#: `mlp` is `md.ppo.Policy`: two tanh layers, then a policy head and a value
#: head off the shared trunk. `agent/src/policy.cpp` implements exactly this,
#: and `entity` beside it — deliberately those two and no more, because an
#: interpreter for arbitrary graphs is a much larger thing to get right than the
#: networks this project actually trains.
#:
#: `entity` is `md.ppo.EntityPolicy`: every threat through one narrow encoder,
#: cross-attention from each threat to the live interceptor and blast sets, a
#: pooled episode context, and a fire logit per battery — plus a critic that
#: shares nothing with any of it. `auxiliary_head` is deliberately absent: it is
#: a training-time signal that no forward pass on a player's machine evaluates,
#: so shipping it would be weight nobody reads.
ARCHITECTURES: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "mlp": (
        ("trunk.0.weight", ("hidden", "observation")),
        ("trunk.0.bias", ("hidden",)),
        ("trunk.2.weight", ("hidden", "hidden")),
        ("trunk.2.bias", ("hidden",)),
        ("policy_head.weight", ("actions", "hidden")),
        ("policy_head.bias", ("actions",)),
        ("value_head.weight", ("one", "hidden")),
        ("value_head.bias", ("one",)),
    ),
    "entity": (
        ("threat_encoder.0.weight", ("width", "threat_features")),
        ("threat_encoder.0.bias", ("width",)),
        ("threat_encoder.2.weight", ("width", "width")),
        ("threat_encoder.2.bias", ("width",)),
        ("interceptor_encoder.0.weight", ("width", "interceptor_features")),
        ("interceptor_encoder.0.bias", ("width",)),
        ("interceptor_encoder.2.weight", ("width", "width")),
        ("interceptor_encoder.2.bias", ("width",)),
        ("blast_encoder.0.weight", ("width", "blast_features")),
        ("blast_encoder.0.bias", ("width",)),
        ("blast_encoder.2.weight", ("width", "width")),
        ("blast_encoder.2.bias", ("width",)),
        ("interceptor_attention.query.weight", ("width", "width")),
        ("interceptor_attention.key.weight", ("width", "width")),
        ("interceptor_attention.value.weight", ("width", "width")),
        ("interceptor_attention.output.weight", ("width", "width")),
        ("interceptor_attention.output.bias", ("width",)),
        ("blast_attention.query.weight", ("width", "width")),
        ("blast_attention.key.weight", ("width", "width")),
        ("blast_attention.value.weight", ("width", "width")),
        ("blast_attention.output.weight", ("width", "width")),
        ("blast_attention.output.bias", ("width",)),
        ("actor_context.0.weight", ("hidden", "context_input")),
        ("actor_context.0.bias", ("hidden",)),
        ("actor_context.2.weight", ("hidden", "hidden")),
        ("actor_context.2.bias", ("hidden",)),
        ("context_to_threat.weight", ("width", "hidden")),
        ("context_to_threat.bias", ("width",)),
        ("relation.0.weight", ("width", "relation_input")),
        ("relation.0.bias", ("width",)),
        ("relation.2.weight", ("width", "width")),
        ("relation.2.bias", ("width",)),
        ("fire_head.weight", ("batteries", "width")),
        ("fire_head.bias", ("batteries",)),
        ("noop_head.weight", ("one", "hidden")),
        ("noop_head.bias", ("one",)),
        ("critic_trunk.0.weight", ("hidden", "observation")),
        ("critic_trunk.0.bias", ("hidden",)),
        ("critic_trunk.2.weight", ("hidden", "hidden")),
        ("critic_trunk.2.bias", ("hidden",)),
        ("value_head.weight", ("one", "hidden")),
        ("value_head.bias", ("one",)),
    ),
}

#: Dimensions whose size is fixed by the manifest rather than inferred.
_FIXED_DIMENSIONS = {"one": 1}

#: Relations between resolved dimensions that name-equality alone cannot state,
#: as (architecture, description, predicate). Two concatenations feed `entity`,
#: and a file whose widths chain individually but do not add up would otherwise
#: read a plausible network that slices its own inputs in the wrong places.
#:
#: What is *not* here is anything needing the observation layout — how many
#: threat slots there are, how wide the globals block is. Those are facts about
#: `md::encode`, so the reader that owns them checks them: `agent/src/policy.cpp`
#: against the simulation it is compiled with, which is the only place the
#: question "does this file match *this* game?" can honestly be answered.
_DERIVED: tuple[tuple[str, str, Callable[[Mapping[str, int]], bool]], ...] = (
    (
        "entity",
        "relation.0 takes the threat, both attention outputs and the episode context",
        lambda size: size["relation_input"] == 4 * size["width"],
    ),
    (
        "entity",
        "actor_context.0 takes the globals block and both pooled entity sets",
        lambda size: size["context_input"] > 2 * size["width"],
    ),
)


#: The one array type in this module. Named, because `np.ndarray` alone leaves
#: every operation on one partially unknown to a strict type checker — and the
#: dtype is not incidental here, it is half the format.
Weights = npt.NDArray[np.float32]

#: A parsed manifest. `object` rather than `Any`: every field is interrogated
#: before it is used, and typing it loosely is what would let a hand-edited file
#: reach the payload reader with a string where an offset belongs.
Manifest = dict[str, object]


class PolicyFormatError(Exception):
    """This file is not a policy this build can run, and why."""


@dataclass(frozen=True)
class Tensor:
    """One named array of weights. Always float32, always little-endian on disk."""

    name: str
    shape: tuple[int, ...]
    values: Weights

    def __eq__(self, other: object) -> bool:
        # The default dataclass comparison would compare arrays with `==` and
        # then try to take the truth of the resulting array.
        if not isinstance(other, Tensor):
            return NotImplemented
        return (
            self.name == other.name
            and self.shape == other.shape
            # numpy's own stubs type the buffer parameters of `array_equal` and
            # `frombuffer` below as Unknown, so strict mode cannot see through
            # either. Scoped rather than blanket: the arrays themselves are
            # typed, and only the stub's gap is waved through.
            and bool(np.array_equal(self.values, other.values))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        )

    def __hash__(self) -> int:
        return hash((self.name, self.shape))


@dataclass(frozen=True)
class NativePolicy:
    """Everything needed to run a policy, and nothing that needs interpreting."""

    schema: int
    observation_size: int
    action_count: int
    architecture: str
    tensors: tuple[Tensor, ...]
    #: Provenance and presentation: display name, run id, canonical score,
    #: simulator version. Never read by the forward pass — a reader that
    #: understands none of these keys still runs the policy correctly.
    metadata: Mapping[str, str | int | float]

    def tensor(self, name: str) -> Tensor:
        for found in self.tensors:
            if found.name == name:
                return found
        raise PolicyFormatError(f"no tensor named {name!r}")


def checksum(payload: bytes) -> str:
    """SHA-256 of the tensor block, hex. What the manifest carries."""
    return hashlib.sha256(payload).hexdigest()


# ---- validation --------------------------------------------------------------
# Run on the way *in* and on the way *out*. Writing a file that could not be read
# back is the failure worth preventing: a promotion that ships a broken model
# should fail where someone can still do something about it, not in the game.


def validate(policy: NativePolicy, *, what: str = "policy") -> None:
    """Raise :class:`PolicyFormatError` unless ``policy`` is runnable."""
    if policy.schema != SCHEMA:
        raise PolicyFormatError(
            f"{what}: schema {policy.schema} — this build reads schema {SCHEMA}. "
            "The observation encoding or action space has changed; re-export the "
            "checkpoint against this build."
        )
    if policy.architecture not in ARCHITECTURES:
        runnable = ", ".join(sorted(ARCHITECTURES))
        raise PolicyFormatError(
            f"{what}: architecture {policy.architecture!r} is not one this build "
            f"can run ({runnable})"
        )
    if policy.observation_size <= 0 or policy.action_count <= 0:
        raise PolicyFormatError(
            f"{what}: observation_size {policy.observation_size} and action_count "
            f"{policy.action_count} must both be positive"
        )

    names = [tensor.name for tensor in policy.tensors]
    if len(set(names)) != len(names):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise PolicyFormatError(f"{what}: duplicate tensor name(s) {duplicated}")

    expected = ARCHITECTURES[policy.architecture]
    if names != [name for name, _ in expected]:
        raise PolicyFormatError(
            f"{what}: {policy.architecture} expects exactly {[n for n, _ in expected]} "
            f"in that order, found {names}"
        )

    # Resolve the dimension names against each other. `hidden` is whatever the
    # first tensor says it is and must then be the same everywhere; the other
    # two are fixed by the manifest, which is what makes a manifest claiming a
    # different observation size than its weights have a rejection rather than a
    # policy that reads garbage off the end of a row.
    sizes: dict[str, int] = {
        "observation": policy.observation_size,
        "actions": policy.action_count,
        **_FIXED_DIMENSIONS,
    }
    for tensor, (_, dimensions) in zip(policy.tensors, expected, strict=True):
        if len(tensor.shape) != len(dimensions):
            raise PolicyFormatError(
                f"{what}: {tensor.name} has {len(tensor.shape)} dimensions, "
                f"expected {len(dimensions)}"
            )
        for extent, dimension in zip(tensor.shape, dimensions, strict=True):
            known = sizes.setdefault(dimension, extent)
            if known != extent:
                raise PolicyFormatError(
                    f"{what}: {tensor.name} has {dimension}={extent}, but "
                    f"{dimension} is {known} elsewhere in this policy"
                )
        if tensor.values.shape != tensor.shape:
            raise PolicyFormatError(
                f"{what}: {tensor.name} declares {tensor.shape} and holds {tensor.values.shape}"
            )
        if tensor.values.dtype != np.float32:
            raise PolicyFormatError(f"{what}: {tensor.name} is {tensor.values.dtype}, not float32")
        if not np.isfinite(tensor.values).all():
            # A NaN in a weight propagates to every logit, so the policy plays
            # uniformly at random and looks merely bad rather than broken.
            raise PolicyFormatError(f"{what}: {tensor.name} contains a non-finite value")

    for architecture, description, holds in _DERIVED:
        if architecture == policy.architecture and not holds(sizes):
            raise PolicyFormatError(
                f"{what}: {policy.architecture} dimensions do not add up — {description}"
            )


# ---- writing -----------------------------------------------------------------


def write(path: Path, policy: NativePolicy) -> Path:
    """Write ``policy`` to ``path`` atomically, or raise and leave it alone.

    Validated first, so a file this reader could not accept is never produced.
    Then written to a sibling temporary and renamed, because the league promotes
    by writing into place and the failure to prevent is a model that was fine
    yesterday being unreadable because a disk filled up mid-write.
    """
    validate(policy, what=str(path))

    payload = bytearray()
    described: list[dict[str, object]] = []
    for tensor in policy.tensors:
        # `astype` with an explicit little-endian dtype rather than `tobytes`:
        # the array may be big-endian or non-contiguous, and neither is visible
        # in a test that only ever runs on one machine.
        raw = np.ascontiguousarray(tensor.values, dtype=DTYPE).tobytes()
        described.append(
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": DTYPE,
                "offset": len(payload),
                "bytes": len(raw),
            }
        )
        payload += raw

    manifest = {
        "schema": policy.schema,
        "observation_size": policy.observation_size,
        "action_count": policy.action_count,
        "architecture": policy.architecture,
        "payload_size": len(payload),
        "checksum": checksum(bytes(payload)),
        "tensors": described,
        "metadata": dict(policy.metadata),
    }
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")

    temporary = path.with_name(f"{path.name}.tmp{id(policy):x}")
    try:
        with temporary.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<II", CONTAINER_VERSION, len(encoded)))
            handle.write(encoded)
            handle.write(payload)
            handle.flush()
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PolicyFormatError(f"{path}: could not be written ({error})") from error
    return path


# ---- reading -----------------------------------------------------------------


def read(path: Path) -> NativePolicy:
    """Parse ``path``, or raise :class:`PolicyFormatError` saying what was wrong.

    Every check is bounds-first: nothing indexes into the payload until the
    manifest's offsets have been shown to lie inside it. The C++ reader has the
    same order for the same reason, and the same fixtures prove both.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise PolicyFormatError(f"{path}: could not be read ({error})") from error

    header = len(MAGIC) + 8
    if len(raw) < header:
        raise PolicyFormatError(f"{path}: truncated — {len(raw)} bytes is shorter than the header")
    if not raw.startswith(MAGIC):
        raise PolicyFormatError(f"{path}: not a policy file (bad magic)")
    container, manifest_length = struct.unpack_from("<II", raw, len(MAGIC))
    if container != CONTAINER_VERSION:
        raise PolicyFormatError(
            f"{path}: container version {container}, this build reads {CONTAINER_VERSION}"
        )
    if len(raw) < header + manifest_length:
        raise PolicyFormatError(f"{path}: truncated — the manifest runs past the end of the file")

    try:
        # `object`, not `Any`: every field below is interrogated before it is
        # used, and a loose type here is what would let a hand-edited file reach
        # the payload reader with a string where an offset belongs.
        parsed: object = json.loads(raw[header : header + manifest_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyFormatError(f"{path}: the manifest is not readable JSON ({error})") from error
    if not isinstance(parsed, dict):
        raise PolicyFormatError(f"{path}: the manifest is not an object")
    manifest: Manifest = {str(key): value for key, value in parsed.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]

    payload = raw[header + manifest_length :]
    declared = _integer(manifest, "payload_size", path)
    if len(payload) != declared:
        raise PolicyFormatError(
            f"{path}: truncated — the manifest declares {declared} bytes of weights "
            f"and the file holds {len(payload)}"
        )
    if checksum(payload) != manifest.get("checksum"):
        raise PolicyFormatError(f"{path}: checksum mismatch — the weights are corrupt")

    tensors = tuple(_tensor(entry, payload, path) for entry in _entries(manifest, path))
    metadata: object = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PolicyFormatError(f"{path}: metadata is not an object")
    named: dict[str, str | int | float] = {}
    for key, value in metadata.items():  # pyright: ignore[reportUnknownVariableType]
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            named[str(key)] = value  # pyright: ignore[reportUnknownArgumentType]

    policy = NativePolicy(
        schema=_integer(manifest, "schema", path),
        observation_size=_integer(manifest, "observation_size", path),
        action_count=_integer(manifest, "action_count", path),
        architecture=str(manifest.get("architecture", "")),
        tensors=tensors,
        metadata=named,
    )
    validate(policy, what=str(path))
    return policy


def _entries(manifest: Manifest, path: Path) -> Sequence[Mapping[str, object]]:
    entries: object = manifest.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise PolicyFormatError(f"{path}: the manifest lists no tensors")
    found: list[Mapping[str, object]] = []
    for entry in entries:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(entry, dict):
            raise PolicyFormatError(f"{path}: a tensor entry is not an object")
        found.append({str(key): value for key, value in entry.items()})  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]
    return found


def _integer(manifest: Mapping[str, object], key: str, path: Path) -> int:
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PolicyFormatError(f"{path}: {key} is missing or not a whole number")
    return value


def _tensor(entry: Mapping[str, object], payload: bytes, path: Path) -> Tensor:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise PolicyFormatError(f"{path}: a tensor has no name")
    if entry.get("dtype") != DTYPE:
        raise PolicyFormatError(f"{path}: {name} is {entry.get('dtype')!r}, expected {DTYPE!r}")

    declared: object = entry.get("shape")
    if not isinstance(declared, list) or not declared:
        raise PolicyFormatError(f"{path}: {name} has no shape")
    shape: list[int] = []
    for extent in declared:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise PolicyFormatError(f"{path}: {name} has a non-positive extent in {declared!r}")
        shape.append(extent)

    offset = entry.get("offset")
    length = entry.get("bytes")
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length < 0:
        raise PolicyFormatError(f"{path}: {name} has no usable offset/length")
    # The check that keeps a hand-edited manifest from reading somebody else's
    # memory in the C++ implementation. Done before the slice, not after.
    if offset + length > len(payload):
        raise PolicyFormatError(
            f"{path}: {name} claims bytes {offset}..{offset + length} of a "
            f"{len(payload)}-byte payload"
        )
    if length != math.prod(shape) * ITEMSIZE:
        raise PolicyFormatError(
            f"{path}: {name} declares shape {shape} ({math.prod(shape) * ITEMSIZE} bytes) "
            f"and reserves {length}"
        )

    flat: Weights = np.frombuffer(  # pyright: ignore[reportUnknownMemberType]
        payload, dtype=LITTLE_ENDIAN_F32, count=math.prod(shape), offset=offset
    )
    # `astype` and not `reshape` alone: `frombuffer` gives a read-only view onto
    # the payload, and a policy handed to a caller should not be a window into a
    # buffer that goes away. The cast to native float32 is also where a
    # big-endian machine gets the byte swap it needs.
    values: Weights = flat.astype(np.float32).reshape(tuple(shape))
    return Tensor(name, tuple(shape), values)
