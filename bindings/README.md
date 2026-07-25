# bindings/ — the RL environment (`md::rl`) and the Python extension

Two targets live here:

| Target | What | Built |
|---|---|---|
| `md_rl` (`md::rl`) | `VecEnv` — the batch environment, pure C++ | **always** |
| `_md_native` | [nanobind](https://nanobind.readthedocs.io/) module wrapping it | opt-in |

`md_rl` is unconditional on purpose: the training loop sits on it, so it carries
the same test gate as the core (`bindings/tests`, CTest label `unit`). It needs
no Python — only `md::core` — so a plain C++ checkout still builds and tests it.

The extension is the optional half:

```bash
cmake --preset release -DMD_BUILD_BINDINGS=ON -DPython_EXECUTABLE=$(which python)
cmake --build --preset release          # -> build/release/python/md/_md_native*.so
```

CMake skips just that target, with a message, if Python or nanobind is absent.

## What is exposed

| Python | C++ | Notes |
|---|---|---|
| `Config`, `ObsSpec` | same | tunables and observation shape |
| `VecEnv` | `md::rl::VecEnv` | the batch environment |

`md::rl::VecEnv` lives here rather than in the core because *none of it is a rule
of the game*: frame-skip, auto-reset and the terminated/truncated split are how a
trainer chooses to consume the simulation. The core stays a pure simulation that
knows nothing about episodes-as-training-data.

## What the tests pin

`bindings/tests/unit/test_vec_env.cpp` covers the properties whose failure a
training run would otherwise blame on the learner: env *i* is seeded `seed + i`;
a rollout is identical to the same `Sim` stepped by hand (so frame-skip really
does sum the window's reward and re-decode the action each tick); the tick cap
truncates rather than terminates, and never both; `final_obs` holds the finished
episode while `obs` already holds the next one; and a batch is **bit-identical
whichever worker count runs it** — without that, reproducibility would not
survive moving to another machine.

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
