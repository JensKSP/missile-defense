# bindings/ — the Python extension (`_md_native`)

A [nanobind](https://nanobind.readthedocs.io/) module exposing `md::core` to
Python as a **vectorised** RL environment. Built only when asked:

```bash
cmake --preset release -DMD_BUILD_BINDINGS=ON -DPython_EXECUTABLE=$(which python)
cmake --build --preset release          # -> build/release/python/md/_md_native*.so
```

It is optional on purpose: a plain C++ checkout never needs a Python toolchain,
and CMake skips this directory with a message if Python or nanobind is absent.

## What is exposed

| Python | C++ | Notes |
|---|---|---|
| `Config`, `ObsSpec` | same | tunables and observation shape |
| `VecEnv` | `md::rl::VecEnv` | the batch environment |

`md::rl::VecEnv` lives here rather than in the core because *none of it is a rule
of the game*: frame-skip, auto-reset and the terminated/truncated split are how a
trainer chooses to consume the simulation. The core stays a pure simulation that
knows nothing about episodes-as-training-data.

## Two decisions that carry the throughput

**Zero copy.** Every method takes caller-owned NumPy arrays and writes into them
in place. A rollout never copies an observation batch and never allocates per
step. `bool` is one byte and NumPy's `bool_` is too, so even the flags and the
action mask are written directly.

**GIL released.** `step`, `reset` and `action_masks` drop the GIL around the C++
work, so the worker pool genuinely runs in parallel instead of taking turns. This
is why the environment does not need Python multiprocessing the way a pure-Python
env would — no pickling, no pipes, no subprocesses.

Measured on a 16-thread Ryzen laptop (WSL Debian, g++ `-O2`):

| batch | agent-steps/s | sim ticks/s | ms per batch |
|---|---|---|---|
| 256 | 314 k | 1.3 M | 0.82 |
| 1 024 | 1.14 M | 4.6 M | 0.90 |
| 4 096 | **2.35 M** | 9.4 M | 1.74 |

Larger batches amortise the fixed per-call cost, which is why a training loop
should prefer a few thousand environments over a few hundred.
