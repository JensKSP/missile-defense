# Training on an NVIDIA GPU (Debian)

Training this project is **optimizer-bound, not simulation-bound** — the batched
C++ environment feeds samples faster than PyTorch can learn from them
([TRAINING.md](TRAINING.md#known-rough-edges)). That is the unusual case where a
GPU is the whole win: the wall the training loop hits is the forward/backward
pass through the policy, and that is exactly what the card does.

This page is the Debian recipe for getting there, written against the machine it
was developed on: **Debian 13 (trixie), an RTX 5090 (Blackwell, `sm_120`), driver
610.43.02**. The shape of it holds for any recent NVIDIA card; only the version
numbers move.

## The one thing to get right first

**You do not need the CUDA toolkit.** No `nvcc`, no `/usr/local/cuda`, no
matching a system CUDA install to anything. PyTorch's Linux wheels bundle their
own CUDA runtime (the `nvidia-*` packages pip pulls in beside `torch`), so the
only thing that has to come from the system is the **driver**. This trips people
up constantly — they install a multi-gigabyte toolkit they never use. This
project compiles no CUDA of its own (the simulation is CPU C++), so the toolkit
buys you nothing.

So the whole job is two moving parts that have to agree:

```
NVIDIA driver  ──exposes──▶  a max CUDA version  ──must be ≥──  the torch wheel's CUDA
```

## 1. The driver

Blackwell (RTX 50-series) needs a **recent** driver. Debian trixie's own
`nvidia-driver` from `non-free` can lag well behind what a just-released card
needs, so the reliable source is **NVIDIA's official CUDA repository for
Debian 13**, which is where this box's 610.43.02 came from. These are the exact
steps that produced the working setup:

```bash
# 1. Add NVIDIA's CUDA repo for Debian 13 and its signing key.
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2. Install the driver metapackage (open kernel module) + firmware.
#    This does NOT pull the CUDA toolkit — that is a separate `cuda-toolkit`
#    package you deliberately skip.
sudo apt install nvidia-driver nvidia-driver-cuda firmware-nvidia-graphics

# 3. Reboot so the kernel module loads.
sudo reboot
```

Then confirm the card is seen:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
# NVIDIA GeForce RTX 5090, 12.0, 610.43.02, 32607 MiB
```

`compute_cap 12.0` is `sm_120` — remember it; step 2 hinges on the torch wheel
having a kernel for it. The **CUDA Version** in the top-right of plain
`nvidia-smi` (here **13.3**) is the *highest* CUDA runtime this driver supports;
any torch wheel built for that or lower will run.

Things worth knowing, so you do not chase them as bugs:

* **Secure Boot.** If it is on, an unsigned kernel module will not load and
  `nvidia-smi` reports *"driver/library version mismatch"* or nothing. Either
  enrol a MOK to sign the module, or disable Secure Boot in firmware.
* **Open vs. proprietary module.** 610.x is the open-kernel-module driver, which
  is the correct and recommended one for Blackwell — the older proprietary module
  does not support these cards.
* **Nouveau.** The NVIDIA packaging blacklists the in-tree `nouveau` driver; a
  reboot is what makes that take effect.

## 2. PyTorch that matches the driver

Pick a wheel whose CUDA version is **≤ the driver's max** (from step 1) and
**≥ 12.8**, which is the first CUDA to ship a Blackwell `sm_120` kernel. This
machine's driver supports 13.3, so `cu130` is the natural, newest match:

```bash
# Into the project's venv (or the interpreter `poe` uses).
.venv/bin/python -m pip install "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/cu130
```

> **The `--index-url` is the whole point.** A plain `pip install torch` from
> PyPI may hand you a wheel built for an older CUDA with no `sm_120` kernel, and
> then everything imports fine and dies the first time it touches the GPU with
> *"no kernel image is available for execution on the device."* Name the CUDA
> build explicitly. `cu128` and `cu129` also work here; `cu130` is chosen only
> because the driver is new enough to allow the newest.

Verify it end to end — not just that it imports, but that it can actually compute
on the card:

```bash
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))   # (12, 0) on a 5090
x = torch.randn(8192, 8192, device="cuda")
print("matmul ok:", (x @ x).sum().is_floating_point())
PY
```

If `capability` prints `(12, 0)` and the matmul runs, the hard part is done.

## 3. Train on it

The environment is C++ and has to be built once for your interpreter; the trainer
then finds the GPU on its own — `train.py` selects `cuda` whenever
`torch.cuda.is_available()`, so there is no flag to remember.

```bash
.venv/bin/python -m pip install numpy nanobind   # runtime + build deps
poe bindings                                     # build _md_native for this venv
poe train                                        # auto-selects the GPU
```

The first line of a run tells you which device won:

```
training on cuda | 1024 envs x 128 steps = 131,072 samples/update | validation 32 seeds
```

If it says `training on cpu`, torch cannot see the card — go back and re-run the
verification in step 2; the answer is always there.

CUDA is for training and routine validation. After validation has selected
`policy-best.pt`, the one final held-out benchmark defaults to CPU so backend
rounding cannot change an argmax and quietly make the published score
device-dependent.

### What the GPU buys you here

Measured on this box (RTX 5090, Ryzen 9 9950X3D), steady-state throughput once
CUDA warm-up is amortised. **These are the flat `mlp` network** — see the next
section for the relational one, which is an order of magnitude slower per sample
and is what the presets `good` and `best` train:

| Config (`mlp`) | steps/s | vs CPU | VRAM | ~time for 1000 updates |
|---|---|---|---|---|
| CPU, 1024 envs (the [TRAINING.md](TRAINING.md) baseline) | ~7,600 | 1× | — | ~5 hours |
| GPU, 1024 envs | ~330,000 | **~43×** | 4.1 GB | **~7 min** |
| GPU, 4096 envs | ~433,000 | ~57× | 8.6 GB | — |
| GPU, 8192 envs | ~452,000 | ~60× | 14.2 GB | — |
| GPU, 16384 envs | ~456,000 | ~60× | 25.9 GB | — |

Two things fall out of that table:

* **Throughput plateaus around 4096–8192 envs.** The card's compute saturates
  there; going to 16384 nearly doubles VRAM for a rounding-error gain. "Use the
  whole 32 GB" is the wrong instinct — a bigger batch past saturation costs
  memory and sample efficiency (fewer gradient steps per sample) without going
  faster. **4096–8192 envs is the sweet spot.**
* **At the CPU's own 1024-env batch, the GPU is ~43× faster** — the five-hour run
  becomes a coffee break. That is the number to quote when someone asks whether
  the card is worth it for this workload.

### The relational architecture is a different machine

`--architecture entity` is **about ten times the compute per sample** — its
per-threat cross-attention and the training-only [auxiliary
targets](TRAINING.md#what-the-agent-is-paid-for) build tensors the flat network
never touches. Measured the same way, four updates each, sampling per-process SM
utilisation:

| Config | steps/s | GPU busy | Peak VRAM |
|---|---|---|---|
| `mlp`, 4096 × 128 (the `fast` preset) | ~446,000 | 50% | 5.6 GiB |
| `entity`, 1024 × 256 (the `good` preset) | ~42,000 | **93%** | 18.9 GiB |
| `entity`, 4096 × 256, 64 minibatches | ~42,000 | **95%** | 16.6 GiB |
| `entity`, 2048 × 512, 64 minibatches (the `best` preset) | ~42,000 | **94%** | 16.6 GiB |

**42k steps/s on this card is a saturated GPU, not an idle one.** That number
looks alarming next to the 446k above it, and the instinct — "the card is not
being used" — is exactly backwards: the flat network is the one that leaves the
GPU half idle, because it is small enough that the C++ simulation becomes the
bottleneck. The relational network is GPU-bound, and its rate barely moves with
`--envs` for that reason.

What that costs in wall clock, for the shipped presets:

| Preset | Samples | At ~42k / ~446k steps/s |
|---|---|---|
| `fast` | 100 × 524,288 = 52M | ~2 minutes |
| `good` | 1000 × 262,144 = 262M | ~1 h 45 (plus evaluations) |
| `best` | 4000 × 1,048,576 = 4.2B | **~30 hours** (plus evaluations) |

The console shows the observed rate and the remaining time on its update tile,
from the run's own `metrics.csv` rather than from this table — your card is not
this card.

### Getting the most out of this hardware

The knobs that matter (all in [TRAINING.md](TRAINING.md#the-knobs)):

* **`--envs 4096`** (or `8192`) — the batch is `envs × steps`, and a batch this
  size is what keeps the card busy rather than starved, while staying under the
  saturation point where sample efficiency starts to suffer.
* **TF32 tensor cores** — letting the `float32` matmuls use Blackwell's TF32 path
  is a measured **~10%** on top of the above (8192 envs: 452k → 499k steps/s), for
  a little matmul mantissa the policy does not miss. It affects only the training
  math, never the C++ simulation's determinism. Turn it on before training:

  ```python
  import torch
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.set_float32_matmul_precision("high")
  ```

* **`--device cuda`** — only needed to *force* the GPU if autodetection is ever
  wrong; normally leave it off.
* **`--minibatches`** — the VRAM figures in the table above are the flat `mlp`
  network. The relational `entity` architecture is a different order of
  magnitude: 1,024 × 256 peaks at **18.9 GiB**, because its per-sample
  threat×entity tensors make peak memory follow the *minibatch* rather than the
  batch. Raising `--minibatches` in step with `--envs` is what keeps a bigger
  batch affordable — the arithmetic, and the measurements behind it, are in
  [TRAINING.md](TRAINING.md#how-much-gpu-memory-a-run-needs).

Watch the card work in another terminal while a run is going:

```bash
nvidia-smi dmon -s um        # utilisation and memory, one line per second
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` | torch wheel has no `sm_120` kernel | reinstall from a `cu128`+ index (step 2) |
| `torch.cuda.is_available()` is `False` | driver not loaded, or wheel is a CPU build | check `nvidia-smi`; reinstall torch from the CUDA index |
| `CUDA driver version is insufficient for CUDA runtime version` | wheel's CUDA is newer than the driver | use a lower `cuXXX` index, or update the driver |
| `nvidia-smi`: *driver/library version mismatch* | module didn't reload after an update | reboot; if Secure Boot is on, sign or disable it |
| run prints `training on cpu` | torch can't see the card | the step-2 verification will say why |
