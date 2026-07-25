# Training an agent

This is the fun part. You have a game, a scripted opponent that plays it
reasonably well, and a way to watch anything either of them does. Now you teach a
neural network to beat the scripted one.

If you have never trained an RL agent before, this page is meant to be read top
to bottom. If you have, skip to [The knobs](#the-knobs).

## What you are actually training

The agent sees the same thing you do — positions, velocities, ammo, which cities
are still standing — as ~1,900 floats. It picks one of 385 discrete actions per
step: *do nothing*, or *send battery B at threat T*. That is it.

Two details shape everything else:

**It is bound by the same hands you are.** The crosshair has a top speed and the
trigger has a minimum interval, and both live in `Sim::step` rather than in any
driver. The agent cannot out-click you; it can only out-*think* you. That is what
makes "the AI beat the baseline" an interesting claim.

**Illegal actions are masked, not punished.** Firing an empty battery does
nothing, so those actions are removed before the network's softmax. The policy
never has to spend gradient discovering that wasted moves are wasted.

## The yardstick

The scripted agent (`poe eval`) plays 32 fixed seeds and scores:

| Metric | Baseline |
|---|---|
| Mean score | **18,036** |
| Mean wave reached | **16.0** |
| Cities surviving | **0.00** of 6 |
| Kills per interceptor | **1.10** |

That last row is where the headroom is. A blast that catches two warheads scores
twice for one interceptor, and the scripted agent manages that only occasionally.
Ammunition — not aim — is what runs out. **A perfect-marksmanship agent still
loses every game around wave 16**, which is exactly why this is worth learning
on: the problem is allocation under a budget, not reflexes.

Your policy is scored on those same 32 seeds, aggregated by the same C++
function, so the numbers sit next to each other honestly.

## Your first run

```bash
poe bindings          # build the C++ environment for Python
poe train             # 1024 envs, defaults, ~131k samples per update
```

You will see a line per update:

```
training on cpu | 1024 envs x 128 steps = 131,072 samples/update | baseline 18,036
update     1 | return        - | entropy 1.619 | value 0.040 | 210k steps/s
update     2 | return     4.87 | entropy 1.602 | value 0.349 | 214k steps/s
```

* `return` is `-` until the first episodes finish — episodes are thousands of
  ticks long, so this is normal, not a hang.
* `entropy` is how undecided the policy is. It should fall *slowly*. A crash
  toward zero in the first few dozen updates means it has committed early, and
  the usual fix is more `entropy_coef`.
* `value` is how badly the critic is predicting returns. Expect it to spike when
  the policy changes behaviour, then settle.

Every 25 updates an episode is written to `runs/`. **Go and watch one.** Open the
app, choose **REPLAYS**, pick the newest. This is the single most useful habit in
the whole loop: a return curve cannot tell you that your agent has learned to
ignore MIRVs, or that it is dumping three interceptors into one warhead. Watching
it for thirty seconds will.

## What a run leaves behind

Everything under `runs/` (`--out-dir` to change it):

| Path | What |
|---|---|
| `runs/checkpoints/policy-<n>.pt` | weights + optimizer + iteration, every `--checkpoint-every` |
| `runs/checkpoints/policy-final.pt` | always written at the end |
| `runs/update-<n>.mdr` | a watchable episode, ~80 kB |
| `runs/metrics.csv` | one row per update, for plotting afterwards |

## Picking up where you left off

```bash
poe train -- --resume runs/checkpoints/policy-00400.pt
```

Checkpoints carry the optimizer, not just the weights. That matters: Adam keeps
momentum estimates, and resuming without them makes the next few updates behave
unlike the ones before — which looks like a mysterious kink in your curve rather
than the artefact it is. `metrics.csv` is appended, so the history stays whole.

## Comparing what you have trained

```bash
poe train -- --load runs/checkpoints/policy-00400.pt
poe train -- --load runs/checkpoints/policy-00400.pt --record-to runs/at-400.mdr
```

Scores a saved policy on the canonical seeds without training anything, and
optionally records an episode of it playing. This is how you tell whether update
800 is really better than update 400, instead of trusting what scrolled past.

## The knobs

All in `TrainConfig` and `PPOConfig` in
[`python/md/train.py`](../python/md/train.py) and
[`python/md/ppo.py`](../python/md/ppo.py), each with its reasoning written next
to it. The ones actually worth touching first:

| Flag | Default | Try changing it when |
|---|---|---|
| `--envs` | 1024 | more is usually better until you run out of RAM |
| `--steps` | 128 | longer rollouts help credit assignment on slow outcomes |
| `--updates` | 1000 | the return is still climbing when it stops |
| `--eval-every` | 50 | you want the yardstick more or less often |
| `--record-every` | 25 | you want more episodes to watch |
| `--max-ticks` | 120000 | you are smoke-testing and want episodes to end fast |

`entropy_coef` (in `PPOConfig`) is the first thing to reach for if the policy
collapses onto one tactic early. `gamma` is already 0.997 because an episode is
tens of thousands of ticks and the payoff for saving a city arrives late.

## Known rough edges

Honest list, so you do not chase these as bugs:

* **GAE treats truncation as termination.** A run cut off by the tick cap is
  valued as if the world ended, which under-values long survivals. The
  environment already returns `final_observation` for a proper bootstrap — wiring
  it in is the obvious first improvement.
* **No curriculum.** M6 calls for one; training currently starts at full
  difficulty.
* **CPU by default.** Fine here: the policy is a two-layer MLP and the simulation
  sustains ~1.7M agent-steps/s, so runs are often environment-bound. See
  [Getting PyTorch](#getting-pytorch) if you want a GPU.

## Getting PyTorch

### Linux

`pip install torch`. This is the primary target.

### Windows

The Clang presets build under MSYS2/MinGW, and **torch publishes no MinGW
wheel** — there is no distribution for that platform tag at all. Only the
*extension module* has to share an ABI with the interpreter importing it, and the
simulation needs neither Qt nor Vulkan, so build the headless half natively and
leave the game on MSYS2:

1. Install **VS Build Tools** (C++ workload) and a **python.org CPython**.
2. From a Developer Command Prompt:

   ```bash
   poe bindings -- win-native --python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
   ```

3. `pip install torch` into that interpreter, then `poe train`.

Both module ABIs can sit beside the package at once — each interpreter loads its
own — so the MSYS2 tooling and the training environment coexist.

> **Determinism.** `-ffp-contract=off` is what makes replays bit-identical, and
> `cl.exe` has no exact equivalent (`clang-cl` passes it through). The MSVC build
> has been checked against the golden trajectory checksum and matches, but if you
> change compilers, run `ctest --preset win-native -L e2e` before trusting a
> recording it produced.

### GPU

* **NVIDIA** — the usual CUDA wheels.
* **AMD** — ROCm PyTorch on Windows is a preview limited to RX 7000/9000 series
  and some Ryzen AI APUs; older cards are not supported, and consumer RDNA 2 on
  Linux needs `HSA_OVERRIDE_GFX_VERSION` workarounds. `torch-directml` runs on
  any DX12 GPU but pins an older torch.
