# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""That the package imports, and that the version it reports is this tree's.

This used to assert a literal `"0.1.0"`. A hardcoded version in a test is a
fifth place the number is written — `poe bump` edits the four that *declare* it
and cannot know about one that merely asserts it — so the first release after it
was written turned it red, in a commit whose only job was to change the version.

Asserting against `tools.version` instead makes it a check that cannot rot, and
a more useful one than the literal ever was: `read_versions` reads the source
file's text, while `__version__` here is whatever *imported*. Those differ
exactly when a stale copy in site-packages shadows the checkout, which is a real
failure mode of this repo and one nothing else notices.
"""

import missile_defense

from tools.version import read_versions


def test_package_imports_and_reports_the_version_this_tree_declares() -> None:
    declared = read_versions()["python/missile_defense/__init__.py"]
    assert missile_defense.__version__ == declared
