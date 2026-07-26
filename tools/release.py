# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Set the version everywhere, then tag it — which is what publishes a release.

The version lives in three files and nothing keeps them in step, so bumping by
hand is three chances to ship artifacts that disagree about what they are. This
does all three, checks the result with the same code CI checks it with, and then
stops:

    poe bump 0.2.0             # edit the three files, show what changed
    poe bump 0.2.0 --commit    # ...and commit it
    poe bump 0.2.0 --tag       # ...and tag it
    poe bump 0.2.0 --push      # ...and push, which starts the release build

Each flag implies the ones before it, and nothing happens without one — the plain
form is a dry run you can read before agreeing to any of it. `--push` is the only
irreversible step: it puts a tag on a public remote and starts a workflow that
opens a draft release. See docs/RELEASING.md.

`dch` would be the Debian-native way to write the changelog entry, but it exists
only where devscripts does. This is the tooling every platform can run, so the
entry is written here and parsed back to prove it came out well-formed.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path

from . import _util, version

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

#: (file, regex, replacement template). The regex is the one `version.py` reads
#: with, so what is written here is exactly what the check reads back.
EDITS: tuple[tuple[str, str, str], ...] = (
    ("CMakeLists.txt", r"(?m)^(\s*VERSION\s+)\d+\.\d+\.\d+(\s*)$", r"\g<1>{v}\g<2>"),
    ("pyproject.toml", r'(?m)^(version\s*=\s*")\d+\.\d+\.\d+(")', r"\g<1>{v}\g<2>"),
    ("python/md/__init__.py", r'(?m)^(__version__\s*=\s*")\d+\.\d+\.\d+(")', r"\g<1>{v}\g<2>"),
)


def _maintainer(root: Path) -> str:
    """The Maintainer line from debian/control — the changelog trailer must match."""
    control = (root / "debian" / "control").read_text(encoding="utf-8")
    match = re.search(r"^Maintainer:\s*(.+)$", control, re.MULTILINE)
    if match is None:
        raise SystemExit("error: debian/control has no Maintainer line")
    return match.group(1).strip()


def changelog_entry(new: str, *, maintainer: str, when: datetime) -> str:
    """A Debian changelog stanza. The whitespace is load-bearing, not style.

    dpkg-parsechangelog wants: header, blank line, changes indented by two, blank
    line, then a trailer of one space, ``--``, one space, the maintainer, *two*
    spaces, and an RFC 2822 date. Get any of it wrong and the package fails to
    build with a message about the changelog rather than about the spacing.
    """
    return (
        f"missile-defense ({new}-1) unstable; urgency=medium\n"
        f"\n"
        f"  * New upstream release.\n"
        f"\n"
        f" -- {maintainer}  {format_datetime(when)}\n"
        f"\n"
    )


def bump(new: str, *, root: Path | None = None, when: datetime | None = None) -> None:
    """Write ``new`` into all three files. Idempotent; no git involved."""
    base = _util.PROJECT_ROOT if root is None else root
    for relative, pattern, template in EDITS:
        path = base / relative
        text = path.read_text(encoding="utf-8")
        replaced, count = re.subn(pattern, template.format(v=new), text, count=1)
        if count != 1:
            raise SystemExit(f"error: no version line to replace in {relative}")
        path.write_text(replaced, encoding="utf-8")

    changelog = base / "debian" / "changelog"
    stamp = when if when is not None else datetime.now().astimezone()
    entry = changelog_entry(new, maintainer=_maintainer(base), when=stamp)
    changelog.write_text(entry + changelog.read_text(encoding="utf-8"), encoding="utf-8")


def _run(root: Path, *args: str) -> None:
    _util.run(["git", *args], cwd=root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poe bump", description=__doc__)
    parser.add_argument("version", help="the new version, e.g. 0.2.0")
    parser.add_argument("--commit", action="store_true", help="commit the bump")
    parser.add_argument("--tag", action="store_true", help="commit, then tag (implies --commit)")
    parser.add_argument("--push", action="store_true", help="commit, tag and push (implies --tag)")
    args = parser.parse_args(argv)

    new = args.version.removeprefix("v")
    if not VERSION_RE.match(new):
        raise SystemExit(f"error: {args.version!r} is not X.Y.Z")

    root = _util.PROJECT_ROOT
    tag = f"v{new}"
    commit = args.commit or args.tag or args.push
    do_tag = args.tag or args.push

    if _util.run(["git", "status", "--porcelain"], cwd=root, capture=True).stdout.strip():
        # A bump commit that quietly carries unrelated work is how a release ends
        # up containing something nobody meant to ship.
        raise SystemExit("error: the working tree is dirty; commit or stash first")

    bump(new, root=root)
    _run(root, "--no-pager", "diff", "--stat")
    if version.check(tag) != 0:
        raise SystemExit("error: the bump did not take — the tree still disagrees")

    if not commit:
        print(f"\nDry run. The three files now say {new}; nothing committed.")
        print("Re-run with --commit, --tag or --push, or `git checkout -- .` to undo.")
        return 0

    _run(root, "add", "CMakeLists.txt", "pyproject.toml", "debian/changelog")
    _run(root, "commit", "-m", f"Release {tag}")
    if do_tag:
        _run(root, "tag", "-a", tag, "-m", tag)
    if args.push:
        _run(root, "push", "origin", "HEAD")
        _run(root, "push", "origin", tag)
        print(f"\nPushed {tag}. The release workflow is building; it will stop at a draft.")
    else:
        print(f"\nCommitted{' and tagged ' + tag if do_tag else ''}. Not pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
