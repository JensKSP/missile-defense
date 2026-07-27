# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Tests for running the README's quick start — without running it.

The job itself needs a clean Ubuntu container and several minutes; what needs
checking on every commit is cheaper and more fragile: that the block is still
*found*, and that the two lines CI cannot run verbatim are still recognised. A
substitution that silently stops matching turns the job into one that clones
master, or one that hangs waiting for a window.

The last test is the real point of the whole tool: it reads the actual README,
so a quick start that loses its code block fails here rather than in CI.
"""

from __future__ import annotations

import pytest

from tools import quickstart
from tools.quickstart import GAME_PATH, extract, rewrite

MARKDOWN = """\
# Project

## Quick start

Words before the block.

```bash
# 1 — dependencies
sudo apt update
sudo apt install clang-21

# 2 — build
git clone https://github.com/JensKSP/missile-defense.git
cd missile-defense
cmake --preset release && cmake --build --preset release

# 3 — play
./build/release/app/md_app
```

## Something else
"""


def test_the_block_under_the_heading_is_what_comes_back() -> None:
    lines = extract(MARKDOWN)
    assert lines[0] == "# 1 — dependencies"
    assert lines[-1] == GAME_PATH
    assert "## Something else" not in lines


def test_a_readme_with_no_quick_start_is_an_error_not_an_empty_run() -> None:
    # "the quick start passed" must never be what a missing one reports.
    with pytest.raises(LookupError):
        extract("# Project\n\nNo quick start here.\n")


def test_an_unclosed_block_is_an_error() -> None:
    with pytest.raises(LookupError):
        extract("## Quick start\n\n```bash\nsudo apt update\n")


def test_the_clone_is_redirected_at_the_commit_under_test() -> None:
    """Cloning the public URL would build master, not the change being reviewed."""
    script, notes = rewrite(extract(MARKDOWN), source="/checkout", drop_sudo=False)
    clone = next(line for line in script if line.startswith("git clone"))
    assert "/checkout" in clone
    assert "github.com" not in clone
    assert any("master" in note for note in notes)


def test_starting_the_game_becomes_checking_that_it_is_there() -> None:
    """There is no display in CI, and the game does not exit on its own."""
    script, notes = rewrite(extract(MARKDOWN), source="/checkout", drop_sudo=False)
    assert GAME_PATH not in script  # not the bare command any more
    assert any(line.startswith(f'test -x "{GAME_PATH}"') for line in script)
    assert any("display" in note for note in notes)


def test_everything_else_runs_exactly_as_written() -> None:
    """The package names and the presets are the whole point of the exercise."""
    script, _ = rewrite(extract(MARKDOWN), source="/checkout", drop_sudo=False)
    assert "sudo apt update" in script
    assert "sudo apt install clang-21" in script
    assert "cmake --preset release && cmake --build --preset release" in script


def test_sudo_goes_away_inside_a_root_container() -> None:
    script, notes = rewrite(extract(MARKDOWN), source="/checkout", drop_sudo=True)
    assert "apt update" in script and "sudo apt update" not in script
    assert any("sudo" in note for note in notes)


def test_a_piped_sudo_goes_away_too() -> None:
    # Adding an apt source is `echo ... | sudo tee <file>` — the sudo is after a
    # pipe, not leading, and a root container still has no sudo to run it.
    block = ["echo 'deb ... trixie-backports main' | sudo tee /etc/apt/x.list"]
    script, _ = rewrite(block, source="/checkout", drop_sudo=True)
    assert script == ["echo 'deb ... trixie-backports main' | tee /etc/apt/x.list"]


def test_the_real_readme_still_has_a_quick_start_this_can_run() -> None:
    readme = (quickstart.PROJECT_ROOT / quickstart.README).read_text(encoding="utf-8")
    lines = extract(readme)
    assert any(line.strip().startswith("git clone") for line in lines)
    # `in`, not `startswith`: the real line may carry a `CXX=…` prefix.
    assert any("cmake --preset" in line for line in lines)
    # And both substitutions still find something to substitute.
    _, notes = rewrite(lines, source="/checkout", drop_sudo=False)
    assert len(notes) == 2, notes
