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
| Mean score | **113,834** |
| Mean wave reached | **17.09** |
| Cities surviving | **0.00** of 6 |
| Kills per interceptor | **1.10** |

That last row is where the headroom is. A blast that catches two warheads scores
twice for one interceptor, and the scripted agent manages that only occasionally.
Ammunition — not aim — is what runs out. **A perfect-marksmanship agent still
loses every game around wave 17**, which is exactly why this is worth learning
on: the problem is allocation under a budget, not reflexes.

Your policy is scored on those same 32 seeds, aggregated by the same C++
function, so the numbers sit next to each other honestly.

## Your first run

```bash
poe bindings          # build the C++ environment for Python
poe train             # 1024 envs, defaults, ~131k samples per update
```

You will see a line per update — this is a real run, on a 16-thread CPU:

```
training on cpu | 1024 envs x 128 steps = 131,072 samples/update | baseline 113,834
update     1 | return        - | entropy 1.263 | value 1.503 | 8k steps/s
update     2 | return        - | entropy 1.176 | value 1.040 | 8k steps/s
update     5 | return     8.58 | entropy 0.548 | value 0.236 | 7k steps/s
update    20 | return    13.46 | entropy 0.549 | value 0.485 | 8k steps/s
```

**Budget about five hours for the 1000-update default**, ~17 s per update at
these settings. Pass `--updates 20` if you only want to see the loop turn over.

* `return` is `-` until the first episodes finish — episodes are thousands of
  ticks long, so this is normal, not a hang.
* **`return` is not the game score**, and the two are not comparable. It is the
  sum of *shaped* reward over an episode divided by `Shaping.scale` (100), so it
  reads in the tens while the baseline's score is 113,834. The eval block every
  `--eval-every` updates is what puts the policy and the baseline on one ruler.
* `entropy` is how undecided the policy is. It starts near **1.2, not ln(385) =
  5.9**, because action masking means only a handful of actions are ever legal —
  so read it as "about `exp(entropy)` real choices". A quick early fall as it
  learns which of those are worth taking is expected; what you are watching for
  is it continuing toward zero over the first few dozen updates, which means it
  has committed early, and the usual fix is more `entropy_coef`.
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
| `runs/checkpoints/policy-best.pt` | the highest-scoring evaluation so far — **usually not the final one** |
| `runs/update-<n>.mdr` | a watchable episode, ~80 kB |
| `runs/metrics.csv` | one row per update, for plotting afterwards |
| `runs/evals.csv` | one row per `--eval-every` scoring, in the baseline's units |
| `runs/config.json` | every setting the run was started with |
| `runs/model.json` | the network it is training — layers, shapes, parameter count |
| `runs/train.log` | a copy of everything it printed, flushed line by line |

`runs/` means the directory beside you in a checkout, and the per-user data
directory (`~/.local/share/MissileDefense/runs`) once this is installed from a
package — `--out-dir` and `$MD_RUNS_DIR` override, and the game's REPLAYS browser
follows the same rule so it finds what the trainer wrote. The order is in
[PACKAGING.md](PACKAGING.md#where-a-runs-files-go).

Those last two are deliberately separate files. `metrics.csv` is the training
return, which as above is *not* a score; `evals.csv` is the 32-seed summary that
is, so it is the one a "beat 113,834" line can honestly be drawn across. Keeping
them apart also keeps the sparse rows out of the dense file.

**Take `policy-best.pt`, not `policy-final.pt`.** PPO does not improve
monotonically — a moving target destabilises the critic and entropy collapses —
so a run that peaked at update 800 can quite normally finish worse at 1000. The
best checkpoint is kept separately, by canonical eval score rather than by shaped
return, and the trainer says which update it came from when it exits. The final
one is what `--resume` continues; the best one is what you score or ship.

## What a run reports about itself

A score says a policy plateaued. It does not say *why*. The full per-episode
statistics do, and every one of them is counted off the same deterministic event
stream the agent itself observes — no privileged look inside the simulation:

| Stat | Reads as |
|---|---|
| `ticks` (÷ 60 = seconds) | how long it survived |
| `wave_reached`, `waves_cleared` | how far it got, and how much it finished |
| `cities_left` / `cities_lost` / `bonus_cities` | what it was defending, and what it rebuilt |
| `bases_left` / `bases_lost` / `ammo_left` | what it defended *with*, and what it never spent |
| `shots`, `kills`, `hits`, `mirv_splits` | how the ammunition went |
| **`kills_per_shot[]`** — bins 0, 1, 2, 3, 4+ | the distribution behind the average |

The histogram is the one worth learning to read. "1.10 kills per interceptor" is
a mean, and a mean cannot distinguish an agent that reliably takes one threat per
shot from one that wastes half its ammunition and catches pairs with the rest.
The distribution separates them at a glance: bin 0 is wasted shots, and weight in
bins 2+ is the only evidence of *catching clusters*, which is where a score above
the baseline has to come from. The scripted baseline sits at 2% wasted, 86% single
kills and 12% multiples — a learned policy that beats it will not look like that.

Two places show them. `poe eval` prints the block for the scripted baseline:

```
mean score          15592.5   [14895 .. 16470]
survived               4000 ticks (66.7 s)   4 / 4 reached the cap
last wave              7.00   (6.00 cleared)
cities                 6.00 left   0.00 lost   0.00 rebuilt   (of 6)
bases                  3.00 left   0.00 lost   (of 3)
ammo unfired          19.75   (interceptors still loaded at the end)
targets killed        88.00   (0.00 MIRV splits)
shots fired           80.75   78.00 hit (98%)   1.09 kills/shot
kills per shot   0:5 (2%)  1:274 (86%)  2:36 (11%)  3:2 (1%)  4+:0 (0%)
```

and a training run prints the same block at every `--eval-every`, having written
it to `evals.csv` first. That file's original nine columns keep their names and
their order, so anything that read it before still finds them; the rest are
appended — the per-episode means, then the histogram as `shots_0kill` …
`shots_4plus`. Both printouts come from one C++ `Summary` and one `summarize`,
which is what makes the learned policy and the scripted baseline comparable at
all: the numbers are not merely alike, they are produced by the same code.

`poe eval --frame-skip 4` throttles the scripted agent to the neural policy's
own decision rate, if what you want to compare is tactics rather than reflexes.

## Stopping a run without losing it

A run is hours long, and `Ctrl-C` throws away everything since the last
checkpoint. Two files in the run directory, checked once per update, avoid that:

```bash
touch runs/STOP      # finish this update, write a final checkpoint, flush, exit
touch runs/PAUSE     # block between updates
rm runs/PAUSE        # carry on, exactly where it was
```

Pausing blocks the loop *between* updates rather than suspending the process, so
it keeps its allocations and its place — and a paused run still answers `STOP`.
Both files are cleared when a run starts and when one finishes, so a stale `STOP`
cannot kill tomorrow's run. This is the whole mechanism; the console's buttons
write these same files, which is why they also work on a run you started in a
terminal.

## Watching and driving a run from a window

```bash
poe ui                    # attach to ./runs
poe ui -- path/to/run     # or to a run directory synced from another machine
```

The eval score against the baseline as the big curve, return / entropy / value
loss underneath, and the recordings listed newest-first — select one and press
**▶ Play** (or double-click it) and it opens in the game; **Delete** removes one
you are done with. Under them is the network itself: architecture, parameter
count, the observation and action sizes and a line per layer, read out of
`runs/model.json` rather than out of a checkpoint — opening one of those needs
torch, and the console deliberately cannot. Beside it, which checkpoint is newest
and what it scored.

Each headline number carries the **peak** it has reached and the update it
reached it on, under the value it is at now. A run is not monotone — PPO peaks
and then falls back, which is exactly why the trainer keeps a `policy-best.pt`
separate from the final one — so "is this the best it has managed?" is a real
question, and the eval score's peak is the score that checkpoint holds. Entropy's
peak is almost always its first update, which is its own kind of useful: it is
the number the collapse is measured from.

In the corner of each chart is the same curve as statistics: `μ50 4.61 ±0.31 ·
Δ +0.42` — the mean and spread of the last 50 points, and how that mean compares
with the 50 before it. The window is the curve's last half up to 50 points and is
named in the text, because the charts are sampled at different rates: one point
per update for the diagnostics, one per `--eval-every` for the score. σ is what
says whether a rise is real, and Δ answers the question a training curve is
actually asked — *is it still going up?* — which neither the newest point nor the
peak can. **Hover any chart** and a chip reports the point under the pointer
(`update 400 · 1.32`), the nearest recorded one rather than an interpolation,
with the compared run's value beside it when there is one.

The dropdown beside the title lists the runs inside and beside the directory you
opened, so `poe ui` on a `runs/` full of experiments is enough: pick one and every
panel follows. Which is also how you flip between an experiment and the run it is
meant to beat.

The **vs** dropdown next to it holds one run against another: every curve gains
the second run's line in the same colour, faintly, and each headline number gains
its value underneath. That is the "did that change help?" view — one plot, two
lines, no arithmetic between two y-scales.

The bar across the top is deliberately small: one button that changes meaning
(**Start** → **Pause** → **Resume**), **Stop**, and **Reset**, which attaches to a
fresh run directory and never deletes the old one. **Start** opens the parameter
form — the four fields that change a run's character, everything else behind
*Advanced*, each carrying as its tooltip the reasoning written beside it in
`TrainConfig` and `PPOConfig`. Only what you change is passed, and the resulting
command line is shown, so nothing here is a thing only the UI can do.

**Start** also offers to *continue from* any checkpoint already in the run
directory — a picker rather than a path you type, since the file has to exist —
which passes `--resume` and carries the optimizer state with it.

Training runs as a separate process throughout, so closing the console (or
crashing it) leaves the run alone. **Log** shows what it has printed, whichever
way it was started: the trainer writes `runs/train.log` itself, so a run you
started in a terminal has one too.

Down the right-hand side, under the recordings, is what the machine is doing:
CPU, memory, and the GPU. That last row appears only when a vendor backend
works — `nvidia-ml-py` (imported as `pynvml`) for CUDA cards, `amdsmi` (or
`pyrsmi`) for ROCm — and says which one would fill it in when neither is there.
If a binding imports but its driver does not answer, the row shows that error
instead of claiming the binding is missing. Adding a vendor is one file in
`md/ui/probes/`.

It needs **PySide6** (`pip install PySide6`, Qt Charts included), and **psutil**
for the CPU and memory rows. The `console` extra also installs NVIDIA's small
`nvidia-ml-py` telemetry binding and, on Linux, AMD's `amdsmi` binding. Running
`python3 -m tools.bootstrap` from a checkout installs all of them into `.venv`.
These are optional and none is ever a dependency of the game. On Windows install
them into the same native interpreter that has torch; see
[WINDOWS.md](WINDOWS.md#training-on-windows).

It does **not** need torch. Where there is none, the primary button offers to
install one instead of being a dead control — see
[Getting PyTorch](#from-the-console-without-a-terminal).

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

* **No curriculum.** M6 calls for one; training currently starts at full
  difficulty.
* **CPU by default, and the optimizer is the bottleneck — not the simulation.**
  Measured at the defaults on a 16-thread CPU: **~7.6k agent-steps/s**, ~17 s per
  update. It is tempting to blame the environment for that, and wrong. At 1024
  envs the batched simulation runs ~1.1M agent-steps/s
  ([`bindings/README.md`](../bindings/README.md)), so collecting an update's 131k
  samples costs well under a second — a few percent of the update at most. The
  rest is torch: PPO takes `epochs` × `minibatches` = 4 × 8 passes over the
  rollout, so the learning phase does **four times the forward passes of the
  rollout and a backward pass with each**, through a 1895 → 512 → 512 trunk.

  Two consequences worth knowing before you tune anything. Adding environments
  does not raise steps/s — the batch grows with them, so the update gets
  proportionally more expensive. And this is the part a GPU would actually
  accelerate, which is the opposite of the usual RL situation where the
  environment is the wall. See [Getting PyTorch](#getting-pytorch).

## Getting PyTorch

### From the console, without a terminal

If you installed a package rather than cloning this repository, you do not need
any of the sections below. Open the training console; where the primary button
would say **Start** it says **Set up training…** instead, and that dialog
installs a copy of PyTorch the console manages itself.

It tells you what it is about to do before doing it: which build it recommends
for your machine, which index it comes from, and roughly how large the download
is. Change the build if you know better — a driver too old for the CUDA wheel is
the usual reason — and press **Install**.

What it does is worth knowing, because it is deliberately boring:

* It creates a **virtual environment of its own** under your data directory
  (`~/.local/share/MissileDefense/runtime`, and the equivalents elsewhere;
  `MD_RUNTIME_DIR` moves it to a scratch disk). Your system Python is untouched.
* It installs **only from PyPI or `download.pytorch.org`, over https**. That
  allow-list is the whole trust decision, and it is not configurable: installing
  a package is running its code.
* It then **proves the result works** by importing torch *and* the simulation
  binding in the new interpreter. Only if that succeeds does the runtime become
  the current one — so a download that fails, a wheel with no kernel for your
  card, or a cancelled install leaves nothing behind and cannot break a runtime
  that was already working.

The same dialog repairs and removes it again. Everything else in the console —
attaching to a run, the curves, browsing and replaying recordings — has never
needed torch and still does not.

`MD_PYTHON` still wins over all of this, for the split-interpreter case on
Windows.

### Linux

`pip install torch`. This is the primary target.

### Windows

Torch publishes no MinGW wheel, so the headless half has to be built with a
native MSVC toolchain while the game stays on MSYS2. Both module ABIs coexist
beside the package. The steps, and the determinism caveat that comes with
swapping compilers, are in [WINDOWS.md](WINDOWS.md#training-on-windows).

### GPU

* **NVIDIA** — the usual CUDA wheels. The Debian recipe end to end, and why you
  do *not* need the CUDA toolkit, is in [NVIDIA.md](NVIDIA.md) — along with the
  measured numbers: ~43× the CPU's throughput at the same batch, and where the
  card saturates.
* **AMD** — ROCm PyTorch on Windows is a preview limited to RX 7000/9000 series
  and some Ryzen AI APUs; older cards are not supported, and consumer RDNA 2 on
  Linux needs `HSA_OVERRIDE_GFX_VERSION` workarounds. `torch-directml` runs on
  any DX12 GPU but pins an older torch.
