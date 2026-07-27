# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Run the README's own quick start, so it cannot quietly stop working.

The quick start is the first thing a stranger types and the one place they have
no context to debug from — and it is the hardest page to keep true, because
everyone who could notice it is broken already has the packages installed.
Writing it surfaced exactly that class of error once already (a dropped
``apt update``, which makes ``apt install clang-21`` fail outright on a stale
index, and is invisible from a machine that has the index).

So: extract the commands *out of the README* and run them on a clean copy of the
current Ubuntu LTS.
Extracted rather than copied into a script, because a copy is one more thing to
keep in step and the drift would be silent again.

Two commands cannot run verbatim in CI, and both are rewritten rather than
skipped — with the substitution printed, so a reader of the log knows precisely
what was and was not verified:

* the ``git clone`` fetches ``master``, and CI has to build *this* commit;
* the last line starts the game, which has no display to open and would never
  exit. Its executable being where the README says it is is the claim that
  matters, and that is asserted instead.

Everything else — the package names, the ``apt update`` before them, the preset
names, the build directory — runs exactly as written.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence

from ._util import PROJECT_ROOT

#: Where the quick start lives, and how a fenced block starts and ends.
README = "README.md"
HEADING = "## Quick start"
FENCE = re.compile(r"^```(\w*)\s*$")

#: The path the README tells you to run, and which this asserts the existence of.
GAME_PATH = "./build/release/app/md_app"


def extract(markdown: str, *, heading: str = HEADING) -> list[str]:
    """The lines of the first shell block under ``heading``.

    Deliberately strict: no block under that heading is an error rather than an
    empty run, because "the quick start passed" must never be what a *missing*
    quick start reports.
    """
    lines = markdown.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        raise LookupError(f"no '{heading}' heading in the README") from None

    for index in range(start + 1, len(lines)):
        fence = FENCE.match(lines[index])
        if not fence:
            continue
        if fence.group(1) not in ("bash", "sh", "console", ""):
            continue
        body: list[str] = []
        for line in lines[index + 1 :]:
            if FENCE.match(line):
                return body
            body.append(line)
        raise LookupError("the quick start's code block is never closed")
    raise LookupError(f"no shell block under '{heading}'")


def rewrite(lines: Sequence[str], *, source: str, drop_sudo: bool) -> tuple[list[str], list[str]]:
    """The runnable script, and the list of substitutions made.

    The notes are returned rather than printed so the caller decides where they
    go — and so this stays a pure function with a test.
    """
    script: list[str] = []
    notes: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("git clone "):
            script.append(f"git clone --depth 1 {source} missile-defense")
            notes.append(
                f"cloned {source} instead of the public URL — CI has to build "
                "the commit under test, not master"
            )
            continue
        if stripped == GAME_PATH:
            script.append(f'test -x "{GAME_PATH}" || {{ echo "no {GAME_PATH}"; exit 1; }}')
            notes.append(
                f"checked that {GAME_PATH} exists and is executable instead of "
                "starting it — there is no display, and the game does not exit"
            )
            continue
        if drop_sudo and "sudo " in line:
            # A root container has no `sudo` binary, so every occurrence goes —
            # not only a leading one but one after a pipe too, as in
            # `echo ... | sudo tee /etc/apt/sources.list.d/backports.list`.
            script.append(line.replace("sudo ", ""))
            continue
        script.append(line)
    if drop_sudo:
        notes.append("dropped `sudo` — this container is already root")
    return script, notes


def running_as_root() -> bool:
    """Whether ``sudo`` would be both unnecessary and probably absent.

    A minimal distro container runs as root and does not ship ``sudo``, so the
    README's lines fail there for a reason that has nothing to do with the README.
    ``os.geteuid`` is POSIX-only and this module is imported by the tests on
    Windows too, hence the lookup rather than the call.
    """
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the README's quick start.")
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT),
        help="what to clone instead of the public URL (default: this checkout)",
    )
    parser.add_argument(
        "--print", action="store_true", help="write the script to stdout and do not run it"
    )
    args = parser.parse_args(argv)

    block = extract((PROJECT_ROOT / README).read_text(encoding="utf-8"))
    script, notes = rewrite(block, source=args.source, drop_sudo=running_as_root())
    body = "\n".join(script)

    if args.print:
        print(body)
        return 0

    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    print("--- running the README's quick start ---", file=sys.stderr)
    print(body, file=sys.stderr)
    print("---", file=sys.stderr)

    # A fresh directory, so the clone in the block has somewhere to land and
    # nothing already-built can make a broken instruction look like it worked.
    with tempfile.TemporaryDirectory() as work:
        done = subprocess.run(["bash", "-e", "-x", "-c", body], cwd=work)
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
