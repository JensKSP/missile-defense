# Performance — measuring and improving simulation throughput

The goal is simple: **run as many simulations per second as the machine allows**,
on whatever machine that happens to be. Everything here is committed and runnable
by hand; nothing depends on a particular host.

```bash
poe bench                    # release build — the numbers that mean anything
poe bench --threads 4        # pin the thread count instead of detecting it
poe bench --csv              # machine-readable
poe profile                  # steady workload to attach a sampling profiler to
```

## The workflow

1. **Benchmark** (`bench/`, target `md_bench`) — rates, not wall-clock totals, so
   results compare across machines. Every measured value feeds a checksum that is
   printed, so nothing can be optimised away.
2. **Profile from outside** — with a sampling profiler. See below for why there is
   no in-code timing.
3. **Measure** — never optimise before this step.
4. **Improve** — and re-measure to prove it.

## Reference numbers

Ryzen laptop, 16 hardware threads, clang 22, `release` preset (`-O3` + ThinLTO):

| Measurement | Rate | Per call |
|---|---|---|
| `Sim::step` (no policy) | ~59 M ticks/s | ~17 ns |
| `Sim::step` + scripted agent | ~9.9 M ticks/s | ~101 ns |
| `md::encode` (observation) | ~14 M obs/s | ~70 ns |
| Episodes, 1 thread | ~600 /s | |
| Episodes, 16 threads | ~5 000 /s | **8.5×** |

Absolute numbers drift with thermal state — treat them as a scale, and compare
variants back-to-back rather than across sessions.

Note that end-to-end **episode** throughput barely moved when the simulation got
47 % faster, because evaluation is ~75 % scripted agent. The simulation gains land
where they matter for training instead: the RL path is `step` + `encode` with a
neural policy, and no scripted agent in sight.

Same-trajectory split of a driven tick: **policy ~69 %, simulation ~31 %**. Note
this is measured on one trajectory with timers around both calls — comparing the
two throughput rows above instead would be wrong, because undefended episodes die
around wave 3 with an almost empty sky while agent-driven ones reach wave 16 with
far more entities per tick.

For **training** throughput the relevant path is `step` + `encode` ≈ 90 ns/tick;
the scripted agent is only used for the M4 baseline.

That path is not, however, what bounds a training run at the PPO defaults. A full
run measures **~7.6k agent-steps/s end to end** against ~1.1M for the environment
alone at the same batch size, so the simulation is a few percent of the wall clock
and PyTorch is the rest (see
[Known rough edges](TRAINING.md#known-rough-edges)). Worth knowing before
optimising anything here *for training's sake*: until the PPO update is cheaper, a
faster `Sim::step` does not make a run finish sooner. The numbers above still
govern evaluation, the scripted baseline, and any headless batch work.

## Parallelism

`md::VecSim` owns a batch of independent simulations and offers two modes:

- `step(actions, results)` — lock-step the whole batch, which is what a vectorised
  RL environment needs (one batched inference per tick).
- `run_episodes(policy, n)` — workers pull whole episodes from a shared counter.
  Episodes vary a lot in length, so a per-tick barrier would leave most cores
  waiting on the longest survivor; claiming work on demand keeps them busy.

**Thread count is detected at run time** (`VecSim::hardware_threads()`), never
baked in, because the same build runs on very different machines. Pass an explicit
count to override.

Scaling is near-linear to the physical core count (1.9× / 3.8× / 6.7× at 2/4/8),
then SMT adds about a third more (8.9× at 16). That is the expected shape: the
simulations share nothing, each `Sim` is ~12 KB — far wider than a cache line, so
neighbouring workers never share one — and `step()` neither allocates nor locks.

## Why there is no in-code profiling

**`Sim::step` reads no clock, and must not.** It is a pure function of
`(state, action)` on a fixed timestep — that is what lets drivers own the loop,
replays reproduce bit-exactly, and sixteen threads run it without coordinating.
Wall-clock time has no place inside it.

This was tried anyway, with per-phase scoped timers behind a compile flag, and the
result refuted the approach. A tick is ~17 ns spread over fourteen phases — a
couple of nanoseconds each — while a `steady_clock` read costs ~13 ns. With timers
on, a tick inflated to ~790 ns and every phase reported the same ~26 ns/call: the
instrument, not the code. At **46× distortion** you are no longer measuring a
slower version of the program, you are measuring a different one. The whole
mechanism was removed rather than kept behind a caveat, because a table that is
confidently wrong is worse than no table — somebody will believe it.

The right tool never touches the code: a sampling profiler interrupts the process
on a timer and reads the instruction pointer, so the simulation stays pure and
unaware. The `profile` preset builds for exactly that — `-O3 -g` with
`-fno-omit-frame-pointer` so stacks unwind — and `poe profile` runs a long steady
single-threaded workload to attach to.

| Platform | Tool | Notes |
|---|---|---|
| Linux | `perf record -g ./build/profile/bench/md_bench && perf report` | the default choice |
| Linux | `valgrind --tool=callgrind` | exact instruction counts, ~50× slower |
| Windows | `wpr -start CPU -filemode` … `wpr -stop md.etl`, view in WPA | ships with Windows; needs admin |
| Windows | Intel VTune / AMD uProf | vendor tools, best per-core detail |
| Any | `llvm-xray` | needs `-fxray-instrument`; Linux/BSD/macOS only |

## Codegen options

The `release` preset is **`-O3` + ThinLTO** (keeping `-g`, so it stays
debuggable). Measured back-to-back on the same machine, single-threaded:

| Variant | sim ticks/s | vs `-O2` | golden checksum |
|---|---|---|---|
| `-O2` (the old default) | 40.2 M | — | ✅ |
| `-O3` | 46.7 M | +16 % | ✅ |
| `-O3` + ThinLTO | **59.1 M** | **+47 %** | ✅ |
| + `-march=native` | 70.0 M | +74 % | ✅ |

LTO is the big one: the simulation is small functions across several translation
units, so cross-TU inlining pays off far more than instruction selection does.

**Every variant was checked against the golden determinism test**, not just for
speed. That check is not a formality — LTO and auto-vectorisation are both allowed
to reorder floating-point work, and reassociating a single sum would silently break
the "same seed ⇒ same trajectory" guarantee that replays and the Debug==Release
test depend on. It holds because the project never enables fast-math and pins
`-ffp-contract=off`.

| Option | Default | Notes |
|---|---|---|
| `MD_LTO` | ON in `release` | Portable, deterministic; costs build time |
| `MD_NATIVE` | OFF | `-march=native`: +74 %, but the binary stops being portable |
| `MD_PGO` | `off` | `generate` / `use`, see below |

Anything that would trade accuracy for speed — `-ffast-math`, `-freciprocal-math`,
FMA contraction — is **off limits**: it would break determinism, and with it
replays, the golden test, and reproducible training.

### Profile-guided optimisation

```bash
cmake --preset release -DMD_PGO=generate && cmake --build --preset release
LLVM_PROFILE_FILE=md-%p.profraw ./build/release/bench/md_bench
llvm-profdata merge -output=md.profdata md-*.profraw
cmake --preset release -DMD_PGO=use -DMD_PGO_FILE="$PWD/md.profdata"
cmake --build --preset release
```

Untried so far; typically worth 5–15 % on branch-heavy code. Re-run the
determinism test afterwards like any other codegen change.

## Findings so far

**Observation encoding was 8× above its memory floor.** `md::encode` writes 1895
floats, and cost 406 ns — while merely zeroing that buffer costs 51 ns (it stays in
L1). The gap was a per-element bounds check on every one of those writes. Clearing
the buffer once and then storing only live entities removed it: **406 ns → 65 ns, a
6.2× speedup**, and the padding contract ("empty slots read zero") is now true by
construction rather than by loop. This matters more than it looks: every RL step
needs an observation, so encoding was about to dominate the training loop at ~9×
the cost of the simulation step it describes.

## Known opportunities

- **The observation is mostly padding.** 1895 floats cover full simulation capacity
  (128 threat slots), but the busiest state observed under agent play holds 6
  threats. A smaller `ObsSpec` would cut the buffer sharply — at the cost of being
  able to hide a live threat from the policy, which the fairness rule in
  [API.md](API.md) forbids by default. A defensible middle path is to measure the
  true maximum concurrent threats across many seeds and cap just above it.
- **The scripted agent dominates evaluation** (~69 % of a driven tick), mostly in
  `solve_intercept`'s fixed-point iteration and the coverage test's path sampling.
  Worth attacking only if baseline evaluation becomes a bottleneck — it is not on
  the training path.
- **PGO and LTO are untried.** `llvm-profdata` is available; `MD_NATIVE=ON` enables
  `-march=native` for local runs where portability does not matter.
