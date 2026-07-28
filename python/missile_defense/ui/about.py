# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""What the trainer says about itself: which build it is, and what it runs on.

Two things a shipped application owes its user, and neither is decoration.

**The version.** "It plateaus at 40k" is not a bug report without one, and the
trainer is the half most likely to be installed from a package rather than built
— so the person running it usually cannot answer the question from a checkout
they do not have. It comes from :data:`missile_defense.__version__`, which
``tools/version.py`` checks against the other three declarations so it cannot
quietly say last release's number.

**The notice.** The trainer runs on PySide6 and Qt Charts, both LGPL-3.0. The
project redistributes neither — they arrive from the distribution's packages,
the user's ``pip``, or the managed runtime — but a user is still owed the fact
that this MIT program stands on LGPL libraries, and a file in a repository they
may never open is not where they will meet it. THIRD_PARTY_LICENSES.md stays the
authority; this is the pointer to it that exists inside the running program.

No Qt in this module, deliberately: the dialog is a handful of lines in
``app.py``, while everything *decidable* about the text is here where pytest can
reach it without a display.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata

import missile_defense

HOMEPAGE = "https://github.com/JensKSP/missile-defense"

#: Where the full inventory lives. Named in the text rather than summarised away,
#: because About is allowed to be incomplete and that file is not.
INVENTORY = "THIRD_PARTY_LICENSES.md"


@dataclass(frozen=True)
class Component:
    """One package the trainer runs on, installed or not.

    ``version`` is ``None`` when the package is absent, which is a normal state
    and not an error: every optional half of this project — torch, psutil, both
    GPU probes — is optional on purpose, and a trainer that refuses to draw its
    About box on a machine without them would be reporting its own bug as theirs.
    """

    name: str
    version: str | None
    licence: str
    role: str

    @property
    def installed(self) -> bool:
        return self.version is not None


#: (display name, distribution name, licence, what it is doing here). The
#: distribution name is what `pip` knows it as, which is not always the import
#: name — `nvidia-ml-py` imports as `pynvml`, and asking metadata for the import
#: name would report every NVIDIA machine as having no probe.
_KNOWN: tuple[tuple[str, str, str, str], ...] = (
    ("PySide6", "PySide6", "LGPL-3.0-only", "the window, the widgets and Qt Charts"),
    ("NumPy", "numpy", "BSD-3-Clause", "the observation buffers"),
    ("PyTorch", "torch", "BSD-3-Clause", "training, in a separate process"),
    ("psutil", "psutil", "BSD-3-Clause", "the CPU and memory meters"),
    ("nvidia-ml-py", "nvidia-ml-py", "BSD-3-Clause", "the NVIDIA GPU probe"),
    ("amdsmi", "amdsmi", "MIT", "the AMD GPU probe"),
)

#: Compiled *into* `missile_defense._md_native`, which is the difference that
#: matters here. Everything in `_KNOWN` above is a separate package pip
#: installed and this program merely imports; these two are inside a binary this
#: project ships, so their notices travel with it — which is what their licences
#: ask for, and why `licenses/` carries both.
#:
#: No version: they are compiled in, so there is no metadata to ask, and a
#: number written here by hand would be one more thing to drift. The inventory
#: file records the versions the build actually used.
_COMPILED_IN: tuple[tuple[str, str, str], ...] = (
    ("nlohmann/json", "MIT", "reading the policy and match manifests"),
    ("nanobind", "BSD-3-Clause", "the C++/Python bridge itself"),
)


def version() -> str:
    """The version of the installed ``md`` package."""
    return missile_defense.__version__


def _installed_version(distribution: str) -> str | None:
    """The version of ``distribution``, or ``None`` if it is not installed.

    ``importlib.metadata`` reads what the installer wrote to disk, so this
    answers for torch without importing torch — which is the whole point.
    Importing it here would put a multi-gigabyte extension module and a CUDA
    context inside the trainer process and break the rule the trainer is built
    around (docs/ROADMAP.md, M8, risk 3).
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def components() -> tuple[Component, ...]:
    """Every package the trainer can use, with the version actually present."""
    return tuple(
        Component(name=name, version=_installed_version(distribution), licence=licence, role=role)
        for name, distribution, licence, role in _KNOWN
    )


def render_component(component: Component) -> str:
    """One line for the dialog: what it is, which version, under what licence."""
    state = component.version if component.installed else "not installed"
    return f"{component.name} {state} — {component.licence} — {component.role}"


def _lines() -> Iterator[str]:
    yield "Missile Defense Trainer"
    yield f"version {version()}"
    yield ""
    yield "© 2026 Jens Köhler. Released under the MIT License."
    yield "Developed with Claude Code (Anthropic)."
    yield HOMEPAGE
    yield ""
    yield "Missile Command is a trademark of Atari. This is an independent,"
    yield "non-commercial homage and is not affiliated with or endorsed by Atari."
    yield ""
    yield "This trainer is MIT-licensed and runs on libraries under their own terms:"
    for component in components():
        yield f"  {render_component(component)}"
    yield ""
    yield "None of those is redistributed here — pip installed them and this only"
    yield "imports them. These two are different: they are compiled into the"
    yield "simulation extension this project ships, so their notices travel with it."
    for name, licence, role in _COMPILED_IN:
        yield f"  {name} — {licence} — {role}"
    yield ""
    yield f"The full inventory, with versions, is in {INVENTORY}."


def summary() -> str:
    """The whole About text, as plain text."""
    return "\n".join(_lines())
