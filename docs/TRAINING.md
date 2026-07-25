# Training (M6)

`poe train` runs PPO against the vectorised environment. Everything worth tuning
is in `TrainConfig` in [`python/md/train.py`](../python/md/train.py), and each
field says why it defaults where it does.

```bash
poe train                                # sensible defaults
poe train -- --updates 2000
poe train -- --envs 2048 --record-every 25
```

Three things are wired in because they are what make a run interpretable:

* the policy is scored every `eval_every` updates on the **M4 protocol** — the
  same 32 seeds, aggregated by the same function as the scripted baseline — and
  printed against its 18,036;
* an episode is written to `runs/` every `record_every` updates, to open from the
  app's **REPLAYS** menu. A reward curve will not tell you the policy has learned
  to ignore MIRVs; watching it will;
* checkpoints land in `runs/checkpoints`.

## Getting PyTorch, on each platform

The simulation, the bindings and the app all build with the Clang presets. The
only awkward part is **PyTorch**, and only on Windows.

### Linux — nothing special

`pip install torch` and go. This is the primary target.

### Windows — the MSYS2 build cannot import torch

The Clang presets use MSYS2/MinGW, and **torch publishes no MinGW wheel** — not a
version conflict, there is no distribution for that platform tag at all:

```
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
```

Only the *extension module* has to share an ABI with the interpreter that imports
it, and `md_core`/`md_agent`/`md_replay`/`md_rl` need neither Qt nor Vulkan. So
build the headless half natively and leave the Qt app on MSYS2:

1. Install **VS Build Tools** (or LLVM for `clang-cl`) and a native **CPython**
   from python.org.
2. Build the module against that interpreter:

   ```bash
   poe bindings -- win-native --python "$LOCALAPPDATA/Programs/Python/Python312/python.exe"
   ```

3. `pip install torch` into that interpreter, then `poe train`.

The `win-native` preset is headless (`MD_BUILD_APP=OFF`) on purpose: the app
keeps building under MSYS2 with the Clang presets, exactly as before. Two
toolchains, separated at the one boundary that actually requires it.

> **Determinism note.** `-ffp-contract=off` is what makes replays bit-identical,
> and `cl.exe` has no exact equivalent (`clang-cl` does — the flag is passed
> through). An MSVC-built module falls back to `/fp:precise` and CMake warns. The
> golden checksum test is the authority: run `ctest --preset win-native -L e2e`
> before trusting a recording produced by an MSVC build.

## GPU acceleration

Largely not worth chasing for this project, and on some hardware not available:

* **NVIDIA** — the usual CUDA wheels, nothing special.
* **AMD** — ROCm PyTorch on Windows is a preview limited to **RX 7000/9000**
  series and some Ryzen AI APUs. Older cards (RDNA 2 and earlier) are not
  supported there, and consumer RDNA 2 on Linux needs `HSA_OVERRIDE_GFX_VERSION`
  workarounds. `torch-directml` runs on any DX12 GPU but pins an older torch.
* **CPU is a reasonable default here.** The policy is a two-layer 512-wide MLP
  over a ~1,900-float observation — not a transformer — and the environment
  already sustains ~1.7M agent-steps/s on CPU, so a run is often environment-
  bound rather than network-bound.
