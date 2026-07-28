# Working in this repository

Notes for anyone — human or agent — changing this tree. Not a style guide
([docs/TESTING.md](docs/TESTING.md) and the comments in the code cover that);
this is the set of things that have actually cost days here, written down so
they cost minutes next time.

## Other people are editing this tree at the same time

Several agents commit to the same branch concurrently.

* **Stage explicit paths. Never `git add -A` or `git add <directory>`.** Run
  `git status` again immediately before committing — files appear between your
  last check and your commit.
* **Never revert someone else's uncommitted work**, and if their changes end up
  inside your commit, say so in the message rather than quietly shipping it
  under your name.
* A failing test in your working tree is not automatically yours. Check whether
  it fails on `HEAD` before you debug it — twice this week the answer was
  "another agent is mid-edit", and once the file was not even tracked yet.

## CI runs in the one environment nobody develops in

The gate installs **no torch, no PySide6, no nanobind**. Every developer machine
has all three. That asymmetry produces a whole class of bug that is *invisible
locally and fatal in CI*:

* subclassing a Qt class resolves to `Any` only when PySide6 is missing, so
  mypy's `missile_defense.ui` override list must be complete or the gate fails with errors
  nobody can reproduce;
* pyright has no per-module `ignore_missing_imports`, so any module importing
  torch has to be in its `ignore` list;
* `find_package(Python COMPONENTS Development.Module)` needs `python3-dev`, and
  without it `Python_EXECUTABLE` comes back **empty** — the error says
  "install nanobind for *this* interpreter ()", and the empty parentheses are
  the whole message.

Those skip lists were hand-maintained and drifted four modules behind.
`python/tests/test_typing_policy.py` now checks them against the tree in both
directions, so adding a Qt or torch module fails locally with the exact line to
add. **Do not fix a gate failure by adding to a list; fix the list's test.**

### pyright lies unless you tell it which interpreter

`pyright` resolves imports from whatever `python3` is on `PATH` — *not* the venv
you invoked it through. On a dev box that is `.venv`, which has torch, so a
"CI-equivalent" check passes while CI reports thirteen errors. Always:

```bash
<civenv>/bin/python -m pyright --pythonpath <civenv>/bin/python
```

## `xvfb-run` merges stderr into stdout

It runs its command as `"$@" 2>&1`. Under CI — which is every e2e run — a
subprocess's `stderr` is therefore **always empty**.

This is not a test-ergonomics detail. `validation_errors()` grepped `stderr` for
Vulkan `VUID` lines, found none, and reported every run's renderer clean *for as
long as this suite has run under Xvfb*. The zero-validation-error gate was inert
and nobody could have known from the green ticks.

Assert on `AppRun.output` (both streams), never `AppRun.stderr`.

## A step that never ran is not a step that passed

When a job dies early, everything after it is **unverified, not verified**. Five
of the failures fixed this week were CI correctly reporting real defects that
had simply never been reached:

| What CI said | What was actually wrong |
|---|---|
| `no --report line on stdout` | mesa renamed lavapipe's ICD; the game could not open Vulkan at all |
| `staging failed` | staged from a *configured* tree with no game built in it |
| `nanobind not installed` | `python3-dev` missing, so Python was not found at all |
| `the staged trainer failed` | the test redirected `HOME`, which is where `pip --user` puts PySide6 |
| `dh_missing` | a shipped file no `.install` file claimed |

**The surface error almost never names the cause.** Read the log. Do not infer
from step names — every inference made that way this week was wrong, and every
log actually read named its cause in one line.

## Reproduce in the environment that failed

* `gh` is installed and authenticated. `gh run view --job=<id> --log-failed`.
  It refuses while a run is in progress; step *conclusions* are available
  immediately via `gh api .../actions/jobs/<id>`.
* **Filter runs by `workflowName`.** There are four workflows. Reading the
  Nightly run's job list and concluding that jobs it does not contain have
  started passing is a mistake that has been made here.
* Docker reproduces the `container:` jobs exactly — both Debian legs and
  `quick-start`. Stage the source with `git archive HEAD`, and copy
  `models/pretrained.mdp` in by hand; it is tracked but does not appear in the
  archive.
* **Windows now reproduces locally**, on a Windows box: it is MSVC throughout —
  the game, the bindings and the tests in one tree — so `cmake --preset windows`
  and `ctest --preset windows` are the same commands CI runs. It used to be
  MSYS2/CLANG64 for the game and MSVC for the extension, two ABIs that could
  only be assembled by CI, and that is why this line used to say otherwise.
* macOS still cannot be reproduced by any means available here. For that one,
  the log is the only instrument.

## Express the rule, not the inventory

A shim listing "the five commands `xvfb-run` calls" was wrong the moment it met
CI, which wanted a sixth (`getopt`). The replacement mirrors `/usr/bin` and
excludes what must *not* be reachable — anything starting with `python`,
`missile-defense-trainer`, `missile-defense-train` — which is the promise the test actually makes and
stays true however the wrapper changes.

The same instinct applies to fatal errors. Requiring the native extension
whenever the Python package is installed was too broad: Debian splits the
sources and the extension into two binary packages, and the sources-only build
is legitimate. The real fault is narrower — *bindings were asked for and
silently not produced* — and that is what the check now says.

## Optional halves must degrade, not crash

The game never depends on Python. `missile_defense.ui` must never import torch. Both entry
points land on machines without their optional half, and what they do there is
the first impression: name what is missing and the command that fixes it, never
a traceback. `tests/e2e/test_wheel_install.py` holds that line against a real
`pip install` into a bare interpreter.
