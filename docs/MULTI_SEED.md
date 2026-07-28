# Reproducible multi-seed training

A single PPO run can plateau because of its initialization or the particular
training trajectories it saw. Compare several genuinely independent runs before
changing the algorithm:

```bash
missile-defense-train --multiseed \
  --out-dir runs/entity-3seed \
  --num-seeds 3 \
  --seed-start 1000 \
  -- \
  --architecture entity \
  --updates 750 \
  --envs 4096 \
  --eval-every 25 \
  --record-every 50
```

Use a new `--out-dir`. The runner refuses a nonempty experiment directory and
owns `--seed`, `--out-dir`, `--resume`, and `--load`, so each child starts from
zero in its own directory:

```text
runs/entity-3seed/
  experiment.json
  summary.json
  summary.csv
  seed-001000/
  seed-001001/
  seed-001002/
```

`experiment.json` records the interpreter, seed schedule, complete commands, and
exit codes. Each seed directory is an ordinary training run and can be opened in
the trainer. The summary can be rebuilt without training:

```bash
missile-defense-train --multiseed \
  --out-dir runs/entity-3seed \
  --num-seeds 3 \
  --seed-start 1000 \
  --aggregate-only
```

## Selection boundary

The runner reads each run's `evals.csv`, ignores canonical rows, and selects the
highest-scoring `policy-best.pt` only after verifying that seed offset, seed
count, frame skip, tick cap, and inference device match across runs. A
canonical-only run is incomplete, not a candidate.

Do not run `--load` on every candidate. That would turn the held-out canonical
benchmark into another validation set. Make all architecture, hyperparameter,
and seed choices from `summary.json`, then benchmark the one selected checkpoint
once:

```bash
python -m md.train --load \
  runs/entity-3seed/seed-001001/checkpoints/policy-best.pt
```

For an honest comparison between architecture experiments, keep the same
validation protocol and training-seed schedule for each experiment. Use the
canonical score only for the final model chosen across all experiments.
