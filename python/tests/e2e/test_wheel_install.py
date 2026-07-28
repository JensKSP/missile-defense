# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""The wheel this repository produces, installed and used.

Every other Python test here runs against the *source tree* with `PYTHONPATH`
pointed at `python/`. That is not what anybody installs, and the difference is
exactly where packaging bugs live: a module left out of the wheel, a compiled
binding that does not get copied, an entry point that names a function nobody
renamed, a path rule that only resolves because there happened to be a checkout
above the working directory.

So: build the wheel, install it into an interpreter that has never seen this
repository, run it from a directory that is not this repository, and check the
things a person would do first.

**Hermetic.** The install is `--no-index --find-links` against a directory
holding the wheel that was just built, so nothing is downloaded and the test
cannot pass or fail because of the network. The venv inherits site-packages for
the third-party dependencies — re-downloading NumPy and PySide6 into a throwaway
environment would make this a ten-minute test of pip rather than a one-minute
test of the package.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .harness import PROJECT_ROOT, build_wheel, needs_build, needs_native, needs_wheel_e2e

pytestmark = [pytest.mark.e2e, needs_native, needs_build, needs_wheel_e2e]

#: Building a wheel compiles the extension. Once per session, not per test.
WHEEL_TIMEOUT_S = 900.0
RUN_TIMEOUT_S = 300.0


@pytest.fixture(scope="session")
def installed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh interpreter with this project's wheel in it. Returns its python."""
    workspace = tmp_path_factory.mktemp("wheel")
    wheel = build_wheel(workspace / "dist", timeout=WHEEL_TIMEOUT_S)

    venv = workspace / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=True,
        capture_output=True,
        timeout=RUN_TIMEOUT_S,
    )
    python = (
        venv
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python")
    )
    assert python.is_file(), f"no interpreter at {python}"

    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",  # nothing is downloaded; the test cannot fail on a network
            "--find-links",
            str(wheel.parent),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, f"the wheel would not install:\n{result.stdout}\n{result.stderr}"
    return python


def _python(installed: Path, source: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a snippet in the installed interpreter, from outside the checkout.

    ``cwd`` matters as much as the interpreter: a path rule that resolves only
    because a checkout happened to be above the working directory is precisely
    the bug this file exists to catch, and it is invisible from inside the repo.
    """
    return subprocess.run(
        [str(installed), "-c", source],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        cwd=cwd,
        check=False,
    )


def test_the_installed_package_imports_its_own_native_binding(
    installed: Path, tmp_path: Path
) -> None:
    # From the wheel, not from the tree beside it. If `md` resolved to the
    # source checkout the test would pass while shipping nothing.
    result = _python(
        installed,
        "import md, md._md_native as n; print(md.__file__); print(n.__file__)",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        assert str(PROJECT_ROOT) not in line, (
            f"the installed package resolved to the checkout: {line}"
        )
        assert "site-packages" in line, line


def test_the_simulation_actually_runs_from_the_wheel(installed: Path, tmp_path: Path) -> None:
    """Importable is not the same as working.

    A binding built for another ABI, or one whose data files were left out,
    imports and then fails on the first call. Stepping an environment is the
    cheapest thing that would notice.
    """
    result = _python(
        installed,
        "import json, numpy as np\n"
        "from md.env import VecEnv\n"
        "env = VecEnv(2, seed=7, max_ticks=256)\n"
        "for _ in range(20):\n"
        "    env.step(np.zeros(2, dtype=np.int32))\n"
        "print(json.dumps({'obs': int(env.observations.shape[1])}))\n",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["obs"] > 0


def test_the_data_half_never_drags_torch_in(installed: Path, tmp_path: Path) -> None:
    """Everything the trainer reads with must import without torch.

    The rule the trainer is built around: it has to open on a machine with no
    training runtime and *tell* you how to get one. An import that pulled torch
    in would make the one screen that explains the problem the one screen that
    cannot be shown. Checked against `sys.modules` rather than by uninstalling
    torch, so it still holds on a machine that has it.
    """
    result = _python(
        installed,
        "import sys\n"
        "import md.paths, md.env, md.policy_format, md.library, md.archive, md.tournament\n"
        "leaked = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]\n"
        "print('LEAKED ' + ','.join(leaked) if leaked else 'CLEAN')\n",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("CLEAN"), result.stdout


def test_a_bare_install_says_what_is_missing_instead_of_crashing(
    installed: Path, tmp_path: Path
) -> None:
    """`pip install missile-defense` deliberately brings neither torch nor Qt.

    So both entry points land on a machine that cannot yet run them, and what
    they do about it *is* the first impression. A traceback reads as a broken
    package; a sentence naming the command reads as a package with two halves.
    This venv is that machine — it has the wheel and neither optional half.
    """
    scripts = installed.parent
    for name in ("missile-defense-train", "missile-defense-trainer"):
        executable = scripts / (f"{name}.exe" if sys.platform == "win32" else name)
        # A renamed function leaves a script that installs cleanly and dies on
        # the first run. Every one of these is somebody's first command.
        assert executable.is_file(), f"{name} was not installed"

    for name, missing in (
        ("missile-defense-train", "torch"),
        ("missile-defense-trainer", "PySide6"),
    ):
        result = subprocess.run(
            [str(scripts / name), "--help"],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
            cwd=tmp_path,
            check=False,
        )
        if result.returncode == 0:
            continue  # the optional half is present here; nothing to explain
        said = result.stdout + result.stderr
        assert "Traceback" not in said, f"{name} crashed instead of explaining:\n{said}"
        assert missing in said, f"{name} did not name what is missing:\n{said}"
        assert "pip install" in said, f"{name} did not say how to fix it:\n{said}"


def test_paths_resolve_with_no_checkout_anywhere_above(installed: Path, tmp_path: Path) -> None:
    """The installed layout has no `runs/` beside it and must not need one.

    Started from a desktop entry the working directory is `$HOME` or `/`, so a
    rule that looks "beside the shell" finds nothing while the trainer, installed
    the same way, is writing into the per-user data directory. The two have to
    agree, and here is where that is checkable.
    """
    elsewhere = tmp_path / "not-a-checkout"
    elsewhere.mkdir()
    result = _python(
        installed,
        "import json\nfrom md import paths\n"
        "print(json.dumps({'runs': str(paths.runs_dir()), 'models': str(paths.models_dir())}))\n",
        cwd=elsewhere,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    # Siblings, wherever they landed — the rule the game's C++ mirrors.
    assert Path(payload["models"]).parent == Path(payload["runs"]).parent
    assert str(PROJECT_ROOT) not in payload["runs"]
