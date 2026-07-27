# Training an agent

This is the fun part. You have a game, a scripted opponent that plays it
reasonably well, and a way to watch anything either of them does. Now you teach a
neural network to beat the scripted one.

If you have never trained an RL agent before, this page is meant to be read top
to bottom. If you have, skip to [The knobs](#the-knobs).

## What you are actually training

The agent sees the same thing you do — positions, velocities, ammo, which cities
are still standing, and the visible phase of every blast — as 1,959 floats. It
picks one of 385 discrete actions per decision: *do nothing*, or *send battery B
at threat T*. That is it.

Two details shape everything else:

**It is bound by the same hands you are.** The crosshair has a top speed, the
trigger has a minimum interval, and the simulation samples every driver at 15 Hz.
Those limits live in `Sim::step` rather than in a policy driver; a human click is
kept pending until the next decision tick instead of being dropped between
samples. The agent cannot out-click you; it can only out-*think* you. That is
what makes "the AI beat the baseline" an interesting claim.

**Illegal actions are masked, not punished.** Firing an empty battery does
nothing, so those actions are removed before the network's softmax. The policy
never has to spend gradient discovering that wasted moves are wasted.

Each policy decision advances four simulation ticks. Its next observation sums
the event counts from that whole window, rather than showing only the last tick,
so frame skip cannot hide a launch, detonation or loss cue that the human hears.

## The yardstick

The scripted agent (`poe eval`) plays the 32 fixed held-out canonical seeds
under the published protocol and scores:

| Metric | Baseline |
|---|---|
| Mean score | **13,687.28125** (range 83,525–108,920) |
| Mean wave reached | **15.75** |
| Cities surviving | **0.00** of 6 |
| Kills per interceptor | **1.09** |

That last row is where the headroom is. A blast that catches two warheads scores
twice for one interceptor, and the scripted agent manages that only occasionally.
Ammunition — not aim — is what runs out. **A perfect-marksmanship agent still
loses every game around wave 16**, which is exactly why this is worth learning
on: the problem is allocation under a budget, not reflexes.

The budget is worth stating precisely, because it is the whole game. Wave *N*
sends `2N + 6` threats at three batteries holding ten interceptors each,
refilled every wave and never bankable. Those cross at **wave 12**: past it,
clearing a wave needs more than one kill per interceptor, which is why 1.09
kills/shot and "dies around wave 16" are the same fact stated twice.

`--skill low|medium|high` runs a deliberately weakened baseline; only `high` is
the published yardstick. The ladder is built by removing behaviours, so what
each is worth is measurable — and the answer is lopsided. Ammunition memory
(`Params::coverage_horizon`, how many seconds ahead the agent remembers its own
shots) is worth **~78,000 points**; the `cluster_bonus` that waits for MIRV
spreads to converge is worth **~1,500**. See the README's skill table.

The console draws that ladder as three reference lines. Because a score may only
be read against seeds it was actually played on, the same three agents are also
measured on the validation block below, and the console picks the ladder that
matches the curve:

| Skill | Canonical (offset 32) | Validation (offset 0) |
|---|---|---|
| LOW | 5,024.09375 | 4,929.375 |
| MEDIUM | 8,295.9375 | 8,018.125 |
| HIGH | **13,687.28125** | 13,823.59375 |

Only the canonical column is a claim about anything; the validation column is a
progress marker for a run in flight, and the trainer never selects on it either.
Both are reproducible with `md_agent_eval --skill <name> --seed-offset <0|32>`,
and a test fails if any of the six drifts from what the agent scores.

There are two fixed, disjoint 32-seed blocks from the same deterministic stream:

* **Validation, offset 0:** the historical first 32 seeds. Routine evaluations
  and `policy-best.pt` selection use these. They are not held out because earlier
  runs already selected checkpoints on them.
* **Canonical held-out, offset 32:** the next 32 seeds. `poe eval` and the final
  `--load` benchmark use these at frame skip 4 (15 Hz), an exact 120,000-tick
  cap, and CPU inference for the learned policy.

Both sides are aggregated by the same C++ function. Keep the canonical block
unseen until one checkpoint has been selected on validation; scoring many
checkpoints on it would simply turn it into another validation set.

## Your first run

```bash
poe bindings          # build the C++ environment for Python
poe train             # 1024 envs, defaults, ~262k samples per update
```

You will see a line per update — this is a real run, on a 16-thread CPU:

```
training on cpu | 1024 envs x 256 steps = 262,144 samples/update | validation 32 seeds
update     1 | shaped ret        - | entropy 1.263 | value 1.503 | lr 3.00e-04 | ent coef 0.0200 | 8k steps/s
update     2 | shaped ret        - | entropy 1.176 | value 1.040 | lr 3.00e-04 | ent coef 0.0200 | 8k steps/s
update     5 | shaped ret     8.58 | entropy 0.548 | value 0.236 | lr 2.99e-04 | ent coef 0.0199 | 7k steps/s
update    20 | shaped ret    13.46 | entropy 0.549 | value 0.485 | lr 2.94e-04 | ent coef 0.0197 | 8k steps/s
```

The 256-step rollout is intentionally twice the previous default: it carries
credit across more of a wave, at the cost of roughly twice the samples and
optimizer work per update. Pass `--updates 20` if you only want to see the loop
turn over.

* `return` is `-` until the first episodes finish — episodes are thousands of
  ticks long, so this is normal, not a hang.
* **`return` is not the game score**, and the two are not comparable. It is the
  sum of *shaped* reward over an episode divided by `Shaping.scale` (100), so it
  reads in the tens while the scripted score is 13,687. The validation block
  every `--eval-every` updates is a game score, but it is not a canonical
  ahead/behind claim because it uses a different seed block.
* `entropy` is how undecided the policy is. It starts near **1.2, not ln(385) =
  5.9**, because action masking means only a handful of actions are ever legal —
  so read it as "about `exp(entropy)` real choices". A quick early fall as it
  learns which of those are worth taking is expected; what you are watching for
  is it continuing toward zero over the first few dozen updates, which means it
  has committed early, and the usual fix is more `entropy_coef`.
* `value` is how badly the critic is predicting returns. Expect it to spike when
  the policy changes behaviour, then settle.
* `auxiliary` is the relational policy's error on three observation-derived
  tactical tasks. It is zero for the legacy MLP. For `--architecture entity`,
  it should fall as the actor learns time pressure, existing blast/interceptor
  coverage, and which threats form useful clusters.

Every 25 updates an episode is written to `runs/`. **Go and watch one.** The
REPLAYS menu entry is commented out for now, so name the newest one directly:
`md_app --replay runs/update-00100.mdr`. This is the single most useful habit in
the whole loop: a return curve cannot tell you that your agent has learned to
ignore MIRVs, or that it is dumping three interceptors into one warhead. Watching
it for thirty seconds will.

## What a run leaves behind

Everything under `runs/` (`--out-dir` to change it):

| Path | What |
|---|---|
| `runs/checkpoints/policy-<n>.pt` | weights + optimizer + iteration, every `--checkpoint-every` |
| `runs/checkpoints/policy-final.pt` | always written at the end |
| `runs/checkpoints/policy-best.pt` | the highest validation score so far — **usually not the final one** |
| `runs/update-<n>.mdr` | a watchable episode, ~80 kB |
| `runs/metrics.csv` | one row per update, for plotting afterwards |
| `runs/evals.csv` | validation rows, plus a final canonical row after `--load` |
| `runs/config.json` | every setting the run was started with (`--show-config` reads it back) |
| `runs/model.json` | the network it is training — layers, shapes, parameter count |
| `runs/train.log` | a copy of everything it printed, flushed line by line |

`runs/` means the directory beside you in a checkout, and the per-user data
directory (`~/.local/share/MissileDefense/runs`) once this is installed from a
package — `--out-dir` and `$MD_RUNS_DIR` override, and the game's REPLAYS browser
follows the same rule so it finds what the trainer wrote. The order is in
[PACKAGING.md](PACKAGING.md#where-a-runs-files-go).

Those last two are deliberately separate files. `metrics.csv` is the training
return, which as above is *not* a score; `evals.csv` contains sparse game-score
summaries. Each row records its seed split and offset, seed count, frame skip,
tick cap, and inference device. That metadata is what lets the console draw the
scripted ladder — LOW, MEDIUM and HIGH — from the row's own seed block, and draw
nothing when the agent was never measured under its protocol.

**Take `policy-best.pt`, not `policy-final.pt`.** PPO does not improve
monotonically — a moving target destabilises the critic and entropy collapses —
so a run that peaked at update 800 can quite normally finish worse at 1000. The
best checkpoint is kept separately, by validation score rather than by shaped
return, and the trainer says which update it came from when it exits. The final
one is what `--resume` continues; the validation-selected best one is what you
benchmark once and, if it succeeds, ship.

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

The histogram is the one worth learning to read. "1.09 kills per interceptor" is
a mean, and a mean cannot distinguish an agent that reliably takes one threat per
shot from one that wastes half its ammunition and catches pairs with the rest.
The distribution separates them at a glance: bin 0 is wasted shots, and weight in
bins 2+ is the only evidence of *catching clusters*, which is where a score above
the baseline has to come from. On the canonical block, the scripted baseline sits
at 4% wasted, 83% single kills and 13% multiples — a learned policy that beats it
will not look like that.

Two places show them. `poe eval` prints the canonical block for the scripted
baseline:

```
mean score          98542.3   [83525 .. 108920]
survived              15427 ticks (257.1 s)   0 / 32 reached the cap
last wave             15.75   (14.81 cleared)
cities                 0.00 left   14.03 lost   8.03 rebuilt   (of 6)
bases                  0.75 left   5.94 lost   (of 3)
ammo unfired           0.00   (interceptors still loaded at the end)
targets killed       342.78   (9.53 MIRV splits)
shots fired          315.81   301.69 hit (96%)   1.09 kills/shot
kills per shot   0:452 (4%)  1:8395 (83%)  2:1204 (12%)  3:54 (1%)  4+:1 (0%)
survived cap              0 / 32
```

and a training run prints the same block on the validation split at every
`--eval-every`, having written it to `evals.csv` first. That file's original nine
columns keep their names and their order, so anything that read it before still
finds them; the rest are appended — the per-episode means, the histogram as
`shots_0kill` … `shots_4plus`, and the full evaluation protocol. Old files are
atomically widened with blank protocol fields, which keeps their data without
pretending those rows are baseline-comparable.

Both printouts come from one C++ `Summary` and one `summarize`; matching protocol
metadata is what makes a direct comparison valid. Canonical `poe eval` already
uses frame skip 4. Passing another `--frame-skip` is useful for experiments, but
the result is no longer the published baseline.

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

## Changing how often a run scores itself, while it runs

How often you want the yardstick is not a decision you can make once. Early on
the policy is a different animal every few updates and you want a score
constantly; later an evaluation plays 32 full-length episodes to repeat what the
last one said. So the eval cadence is the one setting a run re-reads:

```bash
cat runs/TUNING.json                          # what the run is using now
echo '{"eval_every": 50}' > runs/TUNING.json  # back off, from the next update
```

A starting run writes this file from its own command line, which is what makes it
the answer to *"what is this run on?"* rather than a pile of overrides. The loop
reads it once per update and logs the change (`eval interval 10 -> 50 updates`);
an unreadable or missing file simply leaves the run on what it was started with,
because a typo must not kill something that is hours old. The console's **eval
every** box writes exactly this file, so it drives a terminal-started run too.

## Watching and driving a run from a window

```bash
poe ui                    # attach to ./runs
poe ui -- path/to/run     # or to a run directory synced from another machine
```

The validation score as the big curve, return / entropy / value loss underneath,
and the recordings listed newest-first — select one and press
**▶ Play** (or double-click it) and it opens in the game; **Delete** removes one
you are done with. Under them is the network itself: architecture, parameter
count, the observation and action sizes and a line per layer, read out of
`runs/model.json` rather than out of a checkpoint — opening one of those needs
torch, and the console deliberately cannot. Beside it, which checkpoint is newest
and what it scored.

**Three dashed lines cross the score chart** — the scripted agent at LOW, MEDIUM
and HIGH, faintest first, so the hardest target carries the most ink. The EVAL
SCORE tile says the same thing in words — *beats MEDIUM · 31,961 to HIGH ·
validation* — and is red under LOW, amber climbing, green once it is past HIGH.
One yardstick would answer the only question a learner has, *am I getting
anywhere*, with "no" for the whole middle of a run; three answer it honestly.
LOW is the first target, MEDIUM the normal one for a trained policy, and HIGH on
the canonical block alone is the **published** baseline that "beat the scripted
agent" means.

**The lines come from the same seed block as the curve** ([the two blocks
above](#the-yardstick)), and every legend entry says which. A run scoring itself
on validation gets the validation ladder, which is what you watch for the hours a
run takes; a canonical benchmark row gets the published one. The two are a few
hundred points apart, so they are never mixed: a segment carrying both blocks, a
changed cadence or cap, a seed count that is not 32, or a row from before
protocols were recorded gets **no** lines at all, and the tile says what makes it
incomparable.

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
its value underneath. Score curves are overlaid only when seed split and offset,
seed count, cadence, cap, and inference backend all match. A protocol change
starts a new score segment and peak instead of drawing a line between unlike
measurements.

The bar across the top is deliberately small: one button that changes meaning
(**Start** / **Continue** → **Pause** → **Resume**), **Stop**, and **Reset**,
which asks what to call a fresh run directory, attaches to it, and never deletes
the old one. Beside them sit the two read-only questions about a run —
**Parameters…**, what it was started with, and **Log**, what it printed.

The primary button opens the parameter form — the four fields that change a run's
character, everything else behind *Advanced*, each carrying as its tooltip the
reasoning written beside it in `TrainConfig` and `PPOConfig`. Only what you
change is passed, and the resulting command line is shown, so nothing here is a
thing only the UI can do.

It also offers to *continue from* any checkpoint already in the run directory — a
picker rather than a path you type, since the file has to exist — which passes
`--resume` and carries the optimizer state with it. In a directory that already
has checkpoints the button reads **Continue**, the newest one is pre-picked, and
the form opens filled in from that run's own `config.json`, so continuing a run
does not mean retyping it.

### What a run was started with

**Parameters…** opens it: every setting, grouped as the trainer wrote it, with
the ones this run *changed* from the defaults marked and counted — twenty-six
numbers are unreadable, and the four that differ are the run. Each row's tooltip
is the sentence written beside that field in the trainer's source, the same text
the Start dialog shows. The library has the same button, so the question can be
asked of a run without opening it, which is what comparing eleven of them
actually looks like.

It is read-only and always available, including while a run is going. In a
terminal the same answer is `poe train -- --show-config runs/amber-anvil`, and
every run prints the block into its own `train.log` as it starts — which is why
the console's **Log** pane shows it too, even for a run started from a terminal.

### Presets: naming a set of options

At the top of that form is a **preset** picker. Three ship with the console, and
what each is for matters more than what each contains:

| Preset | What it is | On a 5090 | VRAM |
|---|---|---|---|
| `fast` | Throughput first — 4,096 envs (the [saturation point](NVIDIA.md#getting-the-most-out-of-this-hardware)), a 128-step rollout, 100 updates, the flat `mlp`. For checking the loop turns and the machine is set up. | ~2 min | ~6 GiB |
| `good` | **The recipe that produced the bundled model**: `entity`, 1,024 × 256, 1,000 updates — 90,866 on the held-out block, beating MEDIUM and approaching HIGH. | ~2 h | ~19 GiB |
| `best` | Four times the samples an update — `entity`, 2,048 × 512 for the late-wave credit [The knobs](#the-knobs) recommends — over 4,000 updates annealed across the whole run, in **64** minibatches rather than 8. | **~30 h** | ~17 GiB |

Those times are measured, and the two `entity` presets are **GPU-bound at about
42,000 steps/s** — a tenth of what `fast` reports, because the relational network
is ten times the compute per sample. A slow-looking steps/s on `good` or `best`
is a saturated card, not an idle one; the [comparison is in
NVIDIA.md](NVIDIA.md#the-relational-architecture-is-a-different-machine), and the
console shows your own card's rate and the time remaining on the update tile.

Only `good` has a measured result behind it. `best` is a considered bet, and its
description in the console says so — a preset promising a number nobody has
measured would be the expensive kind of wrong, paid for in GPU hours. Note that
`best` needs *less* memory than `good` despite training on four times the data:
see below, because that is the part everyone gets wrong.

**Save as…** keeps whatever is in the form under a name of your own; **Update**
and **Delete** work on your own presets only. The three above are read-only
because their names are quoted here and in the console's help, so `good` has to
keep meaning what this table says — take a copy and change that instead.

Saved presets live in `presets.json` beside `runs/` (`$MD_PRESETS_FILE`
overrides; see [PACKAGING.md](PACKAGING.md#where-a-runs-files-go)). They store
only the values that differ from the trainer's own defaults, which is also why
picking `good` shows a command line of just `--architecture entity`: the other
three of its four values *are* the defaults today. The preset pins all four
regardless, so the recipe survives a default changing under it.

### How much GPU memory a run needs

**Peak memory follows the minibatch, not the batch.** This is the one piece of
arithmetic worth doing before starting a long run, and the intuition everybody
brings to it is wrong.

The rollout buffer is `envs × steps` samples at 8,245 bytes each — an
observation, a mask and five small columns. That part scales with the batch and
is what people expect. But the *update* pushes `batch ÷ minibatches` samples
through the network at a time, and on the relational `entity` architecture each
of those samples costs about **547 KiB** of working memory: its entity encoders
and the [auxiliary targets](#what-the-agent-is-paid-for) build per-sample
threat×entity tensors. That is sixty times the buffer's cost per sample, so the
minibatch dominates completely:

```
peak ≈ envs × steps × 8 KiB   +   (envs × steps ÷ minibatches) × 547 KiB
```

Measured on an RTX 5090 with `torch.cuda.max_memory_allocated()`, two updates
each — the model in `python/md/footprint.py` is a straight line through these to
within 1%, and a test holds it there:

| Architecture | envs × steps | minibatches | minibatch | Peak |
|---|---|---|---|---|
| `entity` | 1,024 × 256 | 8 | 32,768 | 18.95 GiB |
| `entity` | 2,048 × 256 | 16 | 32,768 | 20.97 GiB |
| `entity` | 4,096 × 256 | 32 | 32,768 | **out of memory** |
| `entity` | 4,096 × 256 | 64 | 16,384 | 16.61 GiB |
| `entity` | 2,048 × 512 | 64 | 16,384 | 16.59 GiB |
| `entity` | 4,096 × 512 | 64 | 32,768 | **out of memory** |
| `mlp` | 1,024 × 256 | 8 | 32,768 | 2.81 GiB |
| `mlp` | 4,096 × 128 | 8 | 65,536 | 5.56 GiB |

So the fix for a run that will not fit is almost never fewer envs:

* **`--minibatches × 2` halves the peak** and costs nothing — the same samples,
  the same update, in smaller pieces. It does change the optimisation slightly
  (more, smaller gradient steps per epoch), which is a trade worth making
  knowingly rather than a free lunch.
* Fewer `--envs` or `--steps` shrinks the buffer *and* the minibatch, but it
  also shrinks the run: less data per update, for the same wall-clock cost per
  sample.
* PyTorch *reserves* 10–30% more than it allocates, and your desktop is holding a
  gigabyte or two before you start. Leave headroom.

The console shows this estimate under the command line whenever a card is
visible, and warns before you start a run that will not fit. If one runs out
anyway, the trainer prints the same arithmetic and names the knob.

Training runs as a separate process throughout, so closing the console (or
crashing it) leaves the run alone. **Log** shows what it has printed, whichever
way it was started: the trainer writes `runs/train.log` itself, so a run you
started in a terminal has one too.

Down the right-hand side, under the recordings, is what the machine is doing:
CPU, memory, GPU load and **VRAM**. That last one has a bar of its own rather
than a fragment of the caption line, because it is the number that ends a run
eight hours in — GPU utilisation only tells you the card is busy, which you knew
when you pressed Start. Read it against [what a run
needs](#how-much-gpu-memory-a-run-needs) before starting a second one alongside
the first. The two GPU rows appear only when a vendor backend
works — `nvidia-ml-py` (imported as `pynvml`) for CUDA cards, `amdsmi` (or
`pyrsmi`) for ROCm — and say which one would fill them in when neither is there.
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

## The library, and promoting a model

`poe ui` opens on the **run library** when the directory holds several runs, and
on the run itself when it holds one — both are what the path you gave meant. The
library is a level above the dashboard rather than a tab beside it: "which of
these eleven is worth my attention?" needs scores, sizes and states side by
side, which a dropdown cannot show. `‹ Library` goes back.

Every run has two names. The **directory** never changes — it is what a
`--resume`, a path and every file inside the run refer to. The **display name**
is yours, editable at any time including while the run is going, and it is what
the list shows. Notes are for the sentence you will want in a fortnight.

**New run…** asks for the name before it asks for anything else, prefilled with
a generated one so Enter is still a whole answer. What you type becomes the
directory, lowercased and hyphenated — `Entity policy, 3 seeds` writes into
`runs/entity-policy-3-seeds` — and is kept as the display name when the two
differ. A name already taken gains a `-2` rather than being refused. Nothing is
written until the run actually starts, so a parameter form you cancel leaves no
empty directory behind. From a terminal the same decision is `--out-dir`.

**Enter Model League** copies a checkpoint into `models/<id>/` as a
`policy.mdp` — the data-only format the game and the native evaluator read
(docs/API.md §7). It is in both places the decision gets made: on the **run
list**, where every run's best score is one column and the comparison actually
happens, and on the **run screen** under the model panel, where it is something
you do *to* the thing described right above it. Same dialog either way. Three
things about it:

* it defaults to the best *evaluated checkpoint that still exists*, which is
  often not the best score: checkpoints and evaluations happen on different
  cadences, so a run's peak evaluation frequently has no file behind it;
* it is a **copy**, so the model outlives the run being cleaned up or archived;
* it can **refuse** — a checkpoint that cannot be exported and read back is not
  promoted, and nothing is written. That is the intended outcome: a league entry
  the game cannot load would be worse.

The league sits beside the run list. **Watch it play** opens the game on that
model against a fresh seed, **Export…** writes the `.mdp` out for somebody else,
and **Import .mdp…** takes one in, validated before anything is written.

Promotion is also the **install step for the game**: `models/` is exactly where
the game looks, so a promoted model appears under **WATCH AI → MODELS** in the
menu without restarting anything.

**Names are unique, and that is enforced.** A model's name is the only thing
shown — in the table, in the game's menu, in a head-to-head result — so two
called `deadline-1330` are two rows nobody can tell apart and picking the wrong
one is silent. Promoting, importing or renaming onto a name already in the
league therefore stops and offers the two answers worth having:

* **Use another name**, which is the default and leaves everything alone;
* **Replace it**, which swaps the weights in place. The model keeps its **id**,
  so any path written down still resolves, and loses its **results**, because
  those were measured on the weights that just left.

The id — the directory name — stays unique by suffixing (`anvil`, `anvil-2`),
since a path must always resolve. A *name* cannot be fixed up that way, because
`anvil-2` is not a name anybody chose.

**Delete…** removes a model from the league, and that is the same act as
removing it from the game: the menu lists exactly this directory and nothing
else. It says what goes before it goes, and there is no second copy anywhere —
a promoted model is frequently the only surviving artifact of a run that has
since been cleaned up, so **Export…** first if there is any chance you want it
back. Recorded matches are deliberately kept: a match is evidence about the
*other* model too, which is why they live in `matches/` beside the league rather
than inside either contestant.

### Ranking models against each other

**Evaluate** scores one model over the canonical held-out seeds — the only
protocol the league ranks on, stated in the dialog before it starts, because a
number measured some other way does not belong in the table.

**Head-to-head…** plays two models over the *same* seeds, taken once and handed
to both. Both run on a worker thread with a progress bar and a cancel; nothing
is recorded until a contest finishes, so cancelling leaves the league exactly as
it was. The bar counts **seeds that are in**, and a canonical block is about a
minute and a half per contestant: inference runs through the same C++ policy the
game does, not the NumPy reference forward pass, which is a hundred times slower
and turned a contest into something that looked like a hang.

**Watch this seed** (in the same dialog) opens the game on the seed being played
*right now*. In a head-to-head that is both models **side by side**: a contest
plays one contestant's whole block and then the other's, so only one side of the
seed has been computed when you ask — both are recorded first, which takes a
couple of seconds and happens off the event loop so the contest's own progress
keeps arriving. An `Evaluate` has one contestant, so it opens one window.

Either way the game plays its own copy — it and the evaluator are both
deterministic, so the same policy on the same seed is the same episode, tick for
tick — which means the contest never waits for it and closing the window changes
nothing. `[` and `]` fast-forward up to 8x. The single-model case from a
terminal:

```bash
./build/release/app/md_app --watch-model models/<id>/policy.mdp --seed <n>
```

Two mean scores answer *which* model is better and nothing at all about *how*.
So a finished head-to-head offers to record one shared seed from each side and
open them side by side in the game — one screen, one clock, both agents solving
the same problem. That is the question the table provokes and cannot answer.

In that split screen, **wave sync** (`W`, on by default) holds whichever side
reaches a new wave first until the other arrives, and says so under its score.
Without it a stronger agent is fighting wave 7 beside the other's wave 5 within
a minute, and the two halves stop being comparable — with it, every wave is
played by both at once. Off gives the strict reading, where the tick number
alone says where both sides are and getting further in the same time is the
whole point.

## Getting the disk back

Training fills a disk faster than anything else here: a long run is gigabytes of
checkpoints, most of which nobody will load again. **Storage…** on a run shows
what it costs, split by what it would cost to lose, and offers three things.

**Clean up** removes checkpoints that are not the newest, not the best-evaluated
and not pinned. It shows the list first and then removes exactly that list — the
plan you agreed to, not a fresh one computed at the moment of the click.

**Archive…** writes the run to a ZIP, hashing every file on the way in so the
archive can be verified later without the original. **Archive and remove…**
writes, verifies, and only then deletes: the one outcome nobody recovers from is
a run archived, deleted and then found unreadable.

**Delete…** in the run list is the one with no copy of anything: it names what
goes — checkpoints, recordings, updates, total size — and then removes the
directory for good. It refuses a **live** run rather than asking, because
deleting a directory a trainer still has open loses the run in pieces instead of
all at once; stop it first. It also refuses to delete the library directory
itself when that directory *is* the run.

**Restore…** puts one back, verifying before a single file lands and refusing to
merge into an existing run — a half-merged run has `metrics.csv` from one and
checkpoints from another, and nothing downstream would notice.

Neither can be cancelled once started, on purpose. An interrupted write cleans
itself up; an interrupted *removal* leaves a run nobody can reason about, and a
stop that only sometimes stops is worse than none.

## Why a run stopped improving: the STATISTICS tab

The score curve tells you a run has plateaued. It cannot tell you why, and that
is the only question a plateau raises. **STATISTICS**, beside **TRAINING** above
the plots, is the whole per-episode stat block from the latest evaluation —
every column [`evals.csv`](#what-a-run-leaves-behind) carries, which until now
nothing read.

Fourteen tiles, in the order the questions get asked:

| Group | What it answers |
|---|---|
| score · survived · wave reached · waves cleared | how well it did, and *how long it lasted* — `survived` is `mean_ticks` as minutes and seconds, because 11,633 is not a duration anyone can feel |
| cities lost · cities left · bases lost · cities rebuilt | what it cost to get there |
| shots fired · kills · hit rate · wasted · ammo left · MIRV splits | how the ammunition was spent |

**Wasted** is the one to watch: shots that killed nothing at all. Two policies
with the same score can be spending three times the ammunition on it, and it is
the difference between one that survives wave 14 and one that runs dry in wave 9.

Under the tiles, the **kills-per-shot distribution** — every shot the evaluation
fired, binned by how many threats its blast destroyed (0, 1, 2, 3, 4+). This is
the clearest single read on whether the policy has learned to *wait for a
cluster*: a policy trading one interceptor for one warhead has almost everything
in the `1` bin, and one that has learned the game has mass at `2` and beyond. The
`0` bin is the wasted ammunition, and the footnote gives the totals and a floor
on the mean (a floor because the last bin is open-ended).

Beside it, four curves that are deliberately **not** the score: ticks survived,
waves cleared, cities lost, bases lost. A run whose score has flattened while its
survival time is still climbing is learning something; one where every line is
flat has stopped, and one where survival is up while cities lost is up too is
trading damage for time.

The **vs** picker at the top applies here as well: every tile gains a delta
against the other run — coloured by whether that direction is *better*, which is
not the same as *larger* — and both the distribution and the four curves gain the
other run's values faintly beside them.

Every number comes from the run's own `evals.csv`, so a run started before those
columns existed keeps its score curve and this tab says so rather than showing a
grid of zeroes. The arithmetic is in `md.ui.stats`, which has no Qt in it and is
tested against hand-written rows; `python/tests/e2e/test_analysis.py` then drives
the real window against a real run, because the failure worth catching is a
column renamed on one side of that join.

## Picking up where you left off

```bash
poe train -- --resume runs/amber-anvil                          # carry on
poe train -- --resume runs/amber-anvil/checkpoints/policy-00400.pt   # from there
```

**Point it at the run, and it works the rest out.** Which checkpoint is the
latest — by the update stored inside it, not by file name or timestamp, since
`policy-best.pt` is usually an *earlier* update and resuming that would silently
rewind the run. What the run was configured with, from its own `config.json`.
And how much of the annealing horizon is left, so a run stopped at update 400 of
1,000 does the remaining 600 rather than another thousand. It writes into the
same run directory, so `metrics.csv` is appended and the history stays one story.

Anything you type still wins — `--updates 200` shortens it, `--out-dir runs/fork`
forks it into a fresh directory instead of extending it — and a flag that
contradicts the checkpoint is still rejected by name rather than silently
changing the run.

Checkpoints carry the optimizer, not just the weights. That matters: Adam keeps
momentum estimates, and resuming without them makes the next few updates behave
unlike the ones before — which looks like a mysterious kink in your curve rather
than the artefact it is. They also carry the original learning-rate and entropy
schedule, so update 401 resumes with update 401's coefficients instead of
restarting the decay.

In the console this is the **Continue** button: a run directory that already has
checkpoints says *Continue* rather than *Start*, and the dialog behind it opens
with that run's own settings filled in and its newest checkpoint picked.

The blast lifetime phase enlarged the observation from 1,895 to 1,959 floats.
Older checkpoints use the previous input schema and are intentionally rejected
by both `--load` and `--resume`; retrain them rather than silently padding or
misaligning their inputs.

## Comparing what you have trained

```bash
poe train -- --load runs/checkpoints/policy-best.pt
poe train -- --load runs/checkpoints/policy-best.pt --record-to runs/best.mdr
```

Runs the held-out canonical benchmark without training anything, and optionally
records an episode. It defaults to CPU and pins the canonical seed offset, count,
frame skip, and tick cap. A different `--device` is allowed for diagnosis, but
the output explicitly disables the published ahead/behind comparison. For a
checkpoint under a run's `checkpoints/` directory, the result is appended to that
run's `evals.csv`, which is when the console can draw the *published* ladder
rather than the validation one it uses while a run is in flight.

Use validation scores to decide whether update 800 is better than update 400,
then run this command once for the selected `policy-best.pt`. Choosing among
checkpoints from repeated canonical results leaks the benchmark back into
training.

### Holding one run against another in the console

The compare picker on the dashboard overlays a second run's curves and fills the
STATISTICS tiles with deltas. It only does that when the two runs were evaluated
under the **same protocol** — the same seed set, decision cadence and tick cap —
because two numbers measured differently are not two numbers, and a delta
between them would be a claim the data does not support.

When it cannot compare, it says so and why, in the tab and on the score tile.
Reading nothing at all and having to guess whether the feature is broken was the
old behaviour and is the bug this replaced.

## Breaking an MLP plateau

The flat MLP is retained for old checkpoints and controlled comparisons, but the
plateau-breaking path is the relational policy:

```bash
poe train -- \
  --architecture entity \
  --out-dir runs/relational-seed-1000 \
  --seed 1000
```

Every threat uses the same encoder and separately attends to the live
interceptors and blasts. This makes “is this particular threat already covered?”
a direct relationship rather than something the network must reconstruct from
1,959 unrelated positions in a flat vector. Its critic is a separate network, so
a large value-loss update cannot overwrite the actor's tactical features.

During PPO updates, the actor also predicts time-to-impact, coverage, and local
cluster density for each live threat. Those labels are derived from the same raw
observation available to the deployed policy; they are never appended to the
observation and require no hidden target or scripted-agent knowledge. Set
`--auxiliary-coef 0` for an ablation.

One seed is evidence about one run, not an architecture. Use the
[multi-seed runner](MULTI_SEED.md) to compare several fresh seeds on validation,
then benchmark only its selected checkpoint once on the canonical split.

## The knobs

All in `TrainConfig` and `PPOConfig` in
[`python/md/train.py`](../python/md/train.py) and
[`python/md/ppo.py`](../python/md/ppo.py), each with its reasoning written next
to it. The ones actually worth touching first:

| Flag | Default | Try changing it when |
|---|---|---|
| `--envs` | 1024 | more is usually better until you run out of RAM |
| `--steps` | 256 | try 512 when later-wave resource decisions still receive weak credit |
| `--updates` | 1000 | the return is still climbing when it stops |
| `--learning-rate-final` | 1e-5 | the late policy still moves too much or freezes too early |
| `--entropy-coef-final` | 0.002 | late exploration is too costly or collapses too soon |
| `--schedule-updates` | same as `--updates` | decay should finish earlier or later than the run |
| `--architecture` | `mlp` | use `entity` for relational attention and a separate critic |
| `--auxiliary-coef` | 0.1 | ablate or tune the relational tactical prediction loss |
| `--eval-every` | 10 | you want the yardstick more or less often — also changeable mid-run, above |
| `--record-every` | 25 | you want more episodes to watch |
| `--max-ticks` | 120000 | you are smoke-testing and want episodes to end fast |

### What the agent is paid for

The reward weights are the third group, and the only one that can change *what
the policy converges to*. They are `--reward-*` on the command line and the last
block of the console's **Advanced** section:

| Flag | Default | What it prices |
|---|---|---|
| `--reward-city-weight` | 100 | a city still standing |
| `--reward-base-weight` | 200 | a battery still firing — worth a third of your guns for the rest of the wave |
| `--reward-ammo-weight` | 5 | an unspent interceptor |
| `--reward-waste-penalty` | 0 | **a shot that killed nothing** |
| `--reward-multikill-bonus` | 0 | **each kill beyond a blast's first** |
| `--reward-gamma` | 0.999 | the shaping discount — *not* `--gamma`, which is PPO's |
| `--reward-enabled` | true | shaping at all |

**The two in bold behave differently from the rest, and it matters.** City, base
and ammo are potential-based (Ng, Harada & Russell, 1999), so they provably
leave the optimal policy unchanged and only change how fast it is found — they
deliver the end-of-wave bonus at the moment it is earned instead of at the next
wave boundary, where 60 Hz discounting makes it nearly invisible. `waste_penalty`
and `multikill_bonus` are *not* potential terms: they genuinely change the
objective, which is the only way to change what the agent learns to do, and the
reason they must be judged on validation rather than on themselves.

None of them touches the game score, so the 13,687.28125 scripted yardstick is
unaffected whatever you set them to. A run records its weights in
`runs/config.json`, so a result you liked is reproducible.

`--reward-gamma` and `--gamma` are two different discounts, which is why the
reward ones are prefixed at all — argparse refused to have both otherwise.

`learning_rate` and `entropy_coef` in `PPOConfig` are the starting values. The
trainer linearly reduces them to the two final values above, reaching them on
the last scheduled update and clamping there. `gamma` is already 0.999 because
an episode is tens of thousands of ticks and the payoff for saving a city
arrives late. `--gae-lambda 0.97` is a reasonable next experiment if even a
512-step rollout is still too short-sighted, but it raises variance and is not
the default.

## Known rough edges

Honest list, so you do not chase these as bugs:

* **No curriculum.** M6 calls for one; training currently starts at full
  difficulty.
* **CPU by default, and the optimizer is the bottleneck — not the simulation.**
  The previous 128-step batch measured **~7.6k agent-steps/s**, ~17 s per update
  on a 16-thread CPU; the current 256-step batch does roughly twice the work per
  update. It is tempting to blame the environment for that, and wrong. At 1024
  envs the batched simulation runs ~1.1M agent-steps/s
  ([`bindings/README.md`](../bindings/README.md)), so collection is only a small
  part of an update. The rest is torch: PPO takes `epochs` × `minibatches` =
  4 × 8 passes over the rollout, so the learning phase does **four times the
  forward passes of the rollout and a backward pass with each**, through a
  1959 → 512 → 512 trunk.

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
