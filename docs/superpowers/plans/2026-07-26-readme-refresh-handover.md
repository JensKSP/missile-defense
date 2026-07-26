# README refresh — handover

Session ran out of budget during the *screenshot capture* phase. **No file in the
repository was modified** — the working tree is exactly as it was at the start
(the pre-existing uncommitted Task 11 / PPO / NVIDIA work is untouched). The only
outputs are candidate images and scripts, preserved outside the tree.

## What was asked

Update `README.md`:

1. **Fresh screenshots**, with the in-game one showing *maximum action* — the
   current `docs/images/gameplay.png` is one MIRV and one fireball over an empty
   sky, and reads as boring.
2. **Add screenshots of the AI Training Console.**
3. **Add prose sections** covering, in order:
   - how to set your machine up for AI training,
   - how to run the scripted AI,
   - how to run the pre-trained, packed model,
   - how to train your own model,
   - how to run your own model in the game,
   - how to watch replays.

## The blocker to raise with Jens before writing (2 of the 6 sections)

**There is no pre-trained packed model, and no way to run a learned policy inside
the game.** Verified, not assumed:

* `app/main.cpp` accepts only `--play`, `--watch`, `--replay`, `--frames`,
  `--until-done`, `--silent`, `--report`. No `--policy` / model flag.
* `grep` for `.mdp` across the tree returns nothing — the portable policy format
  does not exist yet.
* Nothing ships a checkpoint; `runs/checkpoints/*.pt` are local artefacts only.

That is exactly **Tasks 1 → 2 → 3** of
`2026-07-26-ai-training-user-journey.md` (portable `.mdp`, native C++ inference
with cross-process parity, then a bundled pretrained model), all still open. The
in-game AI is the **scripted** agent only.

So those two sections cannot be written truthfully today. The honest substitute,
which *does* work now and should be written as such:

```bash
# score a saved policy on the canonical 32 seeds, and record it playing
poe train -- --load runs/checkpoints/policy-best.pt --record-to runs/mine.mdr
# then watch it: REPLAYS in the menu, or
md_app --replay runs/mine.mdr
```

Record → replay is the current path from "my checkpoint" to "watching it play in
the game". **Ask Jens** whether to (a) ship the README with that substitution and
a note that native in-game inference is coming, or (b) hold those two sections
until Tasks 1–3 land.

## Screenshots — the recipe that works

The hard-won part. Scripts and candidate frames are preserved in
`~/.cache/md-readme-shots/` (`burst.sh`, `burst2.sh`, `burst3.sh` plus six PNGs).

**Why the obvious approaches fail.** The scripted agent is *too good*: it
intercepts threats near their spawn point, so the sky is permanently swept clean.
Every frame of `--watch` — early wave, late wave, 1× or 8× — is two fireballs
near the top and an empty middle. That is why the existing screenshot is boring;
it is an accurate picture of a competent agent.

**What works — `burst3.sh`:** fast-forward at 8×, then press `T` to take over and
spray interceptors at random sky positions. A human's spread lets threats live
long enough to be *seen*, so incoming trails, MIRV splits, interceptors in flight
and several fireballs are all on screen at once. This produces genuinely dense,
interesting frames.

```bash
~/.cache/md-readme-shots/burst3.sh <outdir> <ff-seconds> <shot-count>
# e.g. burst3.sh shots4 24 130
```

Rank the output by action density in the sky (note: **`LC_ALL=C`** — a German
locale makes `printf`/`sort -g` reject `0.0378` as an invalid number, which
silently produced a garbage ranking twice):

```bash
export LC_ALL=C
for f in *.png; do
  m=$(convert "$f" -gravity center -crop 90%x70% +repage \
        -colorspace Gray -threshold 14% -format "%[fx:mean]" info:)
  echo "$m $f"
done | sort -g -r | head -12
```

Then **look at a `montage` contact sheet** — the metric ranks a fullscreen GAME
OVER banner above real action, so it shortlists, it does not choose.

**Best candidates so far** (all 1853×1043, in `~/.cache/md-readme-shots/`):
`f057.png`, `f051.png`, `f056.png`, `f088.png` — crossing red/green/purple
trails, two or three fireballs, a MIRV cluster, clean HUD with no "AI PLAYING"
overlay. Any of these replaces `docs/images/gameplay.png` well. Recapture at
1280×720 if a smaller asset is wanted; the window size is whatever the app opens
with.

Gotchas worth keeping:

* Send keys with `xdotool windowactivate --sync $WID` first, then
  `xdotool key --clearmodifiers`. Keys sent without activating are dropped —
  the first run reached only 4× because two `]` presses vanished.
* At **8× a full baseline run (17 waves, 109,655 points) ends in ~35 s**, so a
  35 s warm-up lands on the GAME OVER screen. ~24 s is mid/late game.
* A stray click or key during the spray can open the pause menu; discard those
  frames (several in the last batch have the menu overlaid).

## Screenshots still to take

* **Training console** — `poe ui`. Window title is
  `Missile Command — training console · <run dir>`, so capture with
  `poe shot -- --title "training console"`. PySide6 and psutil are already in
  `.venv`. **Real data is available** — see below — so the curves, the vs-run
  comparison, the recordings list and the system meters will all be populated.
  This is the one screenshot the README has none of and the user explicitly
  asked for.
* **`docs/images/menu.png`** — stale: it predates the **REPLAYS** entry.
* Optionally the **REPLAYS browser** for the replay section, and a refresh of
  `docs/images/intercept.png`.

## Facts established this session (do not re-derive)

* **Real training data exists** at `~/.local/share/MissileDefense/runs/` — a root
  run plus `mlp-measures` (1500 updates), `entity-1024`, `entity-4096`. Perfect
  for a console screenshot, including the **vs** comparison dropdown.
  `poe ui` with no argument resolves there (no `./runs` in the checkout, so
  `md.paths` falls through to the per-user data dir).
* **Best learned score so far: ~79,724** (root run, update 800) and ~70,206
  (`mlp-measures`, update 1400). The scripted baseline is **113,834** — so the
  learned policy has **not** beaten it. The README's existing "113,834 is the
  number a learned policy has to beat" framing stays accurate; do not write
  anything implying the target has been met.
* Release build is **green** (`cmake --build --preset release`, exit 0).
* `torch 2.13.0+cu130`, CUDA available, RTX 5090. `xdotool`, `import`, `ffmpeg`,
  `montage` all present.
* **README bug spotted:** the *Menu:* line under [How to play](../../../README.md)
  lists START, WATCH AI, HELP, OPTIONS, HIGHSCORES, ABOUT, EXIT — it is missing
  **REPLAYS**, which `app/game_window.cpp:95` does render. Fix while editing.
* Most of the requested prose already exists in `docs/TRAINING.md` and
  `docs/NVIDIA.md`. The README's job is a short, linked on-ramp per section, not
  a second copy — match the existing README voice (dense, opinionated, one
  concrete command per idea).

## Suggested order for the next session

1. Ask Jens the pre-trained-model question above.
2. Copy the chosen action frame into `docs/images/gameplay.png`.
3. Capture the training console (+ menu, + replays browser).
4. Write the README sections, fix the REPLAYS omission.
5. `poe check` is not needed for a docs-only change, but run `poe lint` if any
   tooling is touched. Stage **explicit paths** — other agents commit to this
   branch too.
