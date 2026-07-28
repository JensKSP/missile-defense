# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The documentation gate — §11 of the 1.0 completion plan, as a test.

Every check here runs against the **real tree**, like
`test_version.py::test_this_repository_declares_one_version` and for the same
reason: a documentation claim is only worth anything if something fails when it
stops being true. The plan's acceptance criterion is "no maintained document
contradicts the executable behavior or another maintained document", and until
this file existed nothing enforced it.

It earns its place: on the day it was written it caught a stale `poe` task name,
and the retired-score check below was written *because* three maintained
documents had three different answers for what the bundled policy scores.

**What this deliberately does not check.** Prose. There is no way to assert that
a paragraph is true, and a gate that tries becomes a gate people route around.
Everything here is mechanical — a path exists, a task exists, a number is
labelled — so a failure is always a fact, never an opinion.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tools._util import PROJECT_ROOT

#: Documents that record what was true when they were written and are **not**
#: maintained. They are exempt from every check here, because correcting them
#: would falsify the record — a superseded plan that quotes a superseded number
#: is not wrong, it is history. Anything not listed is maintained and is held to
#: all of it.
HISTORICAL = frozenset(
    {
        # Superseded by the 2026-07-27 completion plan, which says so in its own
        # first paragraph and keeps this as the implementation record.
        "docs/superpowers/plans/2026-07-26-ai-training-user-journey.md",
        "docs/superpowers/specs/2026-07-26-ai-training-user-journey-design.md",
    }
)

#: Scores from the **unhandicapped** protocol, retired on 2026-07-28 when every
#: contestant gained `md::agent::canonical_handicap`. They are not merely old:
#: quoting one as current inverts the project's headline finding, because the
#: scripted agent wins unhandicapped and loses under the handicap.
#:
#: The rule is not "never mention these" — `docs/FINDINGS.md` mentions all of
#: them and the comparison *is* the finding. The rule is that a file mentioning
#: one must also say somewhere that they are not current.
RETIRED_SCORES = {
    "98,542": "scripted HIGH, unhandicapped (now 13,687)",
    "90,866": "the learned policy, unhandicapped (now 23,067)",
    "90,865": "the learned policy, unhandicapped (now 23,067)",
    "19,585": "scripted LOW, unhandicapped (now 5,024)",
    "63,295": "scripted MEDIUM, unhandicapped (now 8,296)",
    "113,834": "an even older scripted figure",
}

#: Words that mark a number as no longer current. A file quoting a retired score
#: must contain at least one, which is a low bar on purpose: the failure this
#: guards against is a document presenting a retired number *as the answer*, and
#: any honest mention of one says why it is there.
RETIREMENT_MARKERS = (
    "retired",
    "unhandicapped",
    "predate",
    "superseded",
    "historical",
    "no longer",
    "used to",
    "stale",
)

#: Top-level directories a repo-relative path in a document can start with.
#: Anchoring on these is what keeps the path check free of false positives:
#: `md/observation.hpp` is an *include* path, `sim/export_policy.py` is
#: package-relative, and `runs/model.json` is a runtime artifact that only exists
#: once someone has trained something. None of the three is a broken reference,
#: and none of them starts with one of these.
SOURCE_ROOTS = (
    "agent/",
    "app/",
    "bench/",
    "bindings/",
    "core/",
    "debian/",
    "docs/",
    "packaging/",
    "python/",
    "replay/",
    "tools/",
)

_LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
_POE_TASK = re.compile(r"\bpoe ([a-z][a-z0-9-]*)")
_CODE_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|cpp|hpp|toml|json|yml|yaml|cmake|txt|md))`")


def _maintained_docs() -> list[Path]:
    """Every markdown file in the tree that anyone is expected to keep true."""
    found = sorted(PROJECT_ROOT.glob("*.md")) + sorted(PROJECT_ROOT.glob("docs/**/*.md"))
    docs = [p for p in found if p.relative_to(PROJECT_ROOT).as_posix() not in HISTORICAL]
    assert docs, "no documentation found — this test is looking in the wrong place"
    return docs


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def test_every_historical_document_still_exists() -> None:
    """Or the exemption above is silently excusing a file nobody can read.

    A stale entry here is worse than no entry: it would exempt a path that a
    future document might occupy, and that document would then be unchecked
    without anyone choosing that.
    """
    missing = [name for name in HISTORICAL if not (PROJECT_ROOT / name).is_file()]
    assert not missing, f"HISTORICAL names files that no longer exist: {missing}"


def test_every_relative_link_resolves() -> None:
    """A dead link in a README is the cheapest possible broken promise."""
    broken: list[str] = []
    for doc in _maintained_docs():
        text = doc.read_text(encoding="utf-8")
        for match in _LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).resolve().exists():
                line = text[: match.start()].count("\n") + 1
                broken.append(f"{_rel(doc)}:{line} -> {target}")
    assert not broken, "documentation links to files that do not exist:\n  " + "\n  ".join(broken)


def test_every_screenshot_is_still_referenced() -> None:
    """The other direction: an image nothing points at is a feature nothing does.

    `docs/images/replays.png` outlived the menu entry it depicted by exactly as
    long as it took someone to notice. Deleting the last reference to a
    screenshot is the normal way a withdrawn feature leaves the prose, and the
    picture of it is what stays behind — so the orphan *is* the signal.

    A screenshot is also the one part of the documentation no text check can
    read. `menu.png` went on showing a REPLAYS entry through every edit that
    removed the word, because the word was not in it.
    """
    images = sorted((PROJECT_ROOT / "docs" / "images").glob("*.png"))
    assert images, "no screenshots found — this test is looking in the wrong place"
    text = "\n".join(doc.read_text(encoding="utf-8") for doc in _maintained_docs())
    orphans = [_rel(image) for image in images if image.name not in text]
    assert not orphans, (
        "screenshots no maintained document references:\n  "
        + "\n  ".join(orphans)
        + "\nDelete them, or reference them. An unreferenced screenshot is usually "
        "a picture of something that no longer exists."
    )


def test_every_poe_task_named_in_the_docs_exists() -> None:
    """The commands are the part a reader will actually paste into a shell.

    Renaming a task and updating four of its five mentions is the ordinary way
    this breaks, and the reader meets it as `poe: no such task`.
    """
    tasks = set(
        tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["poe"][
            "tasks"
        ]
    )
    unknown: list[str] = []
    for doc in _maintained_docs():
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in _POE_TASK.finditer(line):
                if match.group(1) not in tasks:
                    unknown.append(f"{_rel(doc)}:{number} -> poe {match.group(1)}")
    assert not unknown, "documentation names poe tasks that do not exist:\n  " + "\n  ".join(
        unknown
    )


def test_every_repository_path_cited_in_the_docs_exists() -> None:
    """Catches the rename that moved the code and left the prose behind.

    Scoped to paths starting with a real top-level directory — see
    `SOURCE_ROOTS` for why that scoping is the whole design of this check.
    """
    missing: list[str] = []
    for doc in _maintained_docs():
        for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in _CODE_PATH.finditer(line):
                cited = match.group(1)
                if not cited.startswith(SOURCE_ROOTS):
                    continue
                if not (PROJECT_ROOT / cited).exists():
                    missing.append(f"{_rel(doc)}:{number} -> {cited}")
    assert not missing, "documentation cites paths that do not exist:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("score", sorted(RETIRED_SCORES))
def test_a_retired_score_is_never_quoted_as_a_current_one(score: str) -> None:
    """The check that exists because this went wrong in three documents at once.

    On 2026-07-28 the completion plan put the ladder at 19,585 / 63,295 / 98,542
    with the learned policy at 90,866, the roadmap said the learned policy was
    still retraining, and FINDINGS and the README had the measured 13,687 and
    23,067. All three were maintained documents. The plan then turned its numbers
    into a product instruction — "1.0 must not claim that the learned policy
    already beats HIGH" — which the measured result contradicts.

    Mentioning a retired number is fine and sometimes necessary. Mentioning one
    without saying it is retired is what this fails on.
    """
    pattern = re.compile(rf"(?<![\d.,]){re.escape(score)}(?![\d])")
    offenders: list[str] = []
    for doc in _maintained_docs():
        text = doc.read_text(encoding="utf-8")
        if not pattern.search(text):
            continue
        if any(marker in text.lower() for marker in RETIREMENT_MARKERS):
            continue
        line = text[: pattern.search(text).start()].count("\n") + 1  # type: ignore[union-attr]
        offenders.append(f"{_rel(doc)}:{line}")
    assert not offenders, (
        f"{score} ({RETIRED_SCORES[score]}) is quoted without being marked retired in:\n  "
        + "\n  ".join(offenders)
        + f"\nSay so, or use the current figure. Markers: {', '.join(RETIREMENT_MARKERS)}"
    )
