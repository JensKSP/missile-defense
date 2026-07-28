# Human acceptance for 1.0

Automation cannot sign any of this off. Every box here needs a person, a real
machine, and — for the platform rows — hardware that is not this one.

**Nothing in this file is pre-ticked, and nothing may be ticked from a green CI
run.** The rule the roadmap already states applies to all of it: *implementation
complete + gate green* means **ready for sign-off**, not passed. Record who,
when, and which build; a tick with no evidence is worse than an empty box,
because it stops anyone looking again.

Use the nightly artifacts (<https://github.com/JensKSP/missile-defense/releases/tag/nightly>)
or a release candidate — **not a source checkout**. Most of what this list is for
lives between the packaging and the program, and a checkout has neither.

---

## How to record a result

| Column | Means |
|---|---|
| **Build** | The version string from the artifact filename, e.g. `0.1.0-dev213-g86b122e8` |
| **Who / when** | The person who performed it and the date |
| **Evidence** | What convinced you: a screenshot, a recording, a log excerpt, or a sentence describing what you saw |
| **Result** | `pass`, `fail`, or `blocked` — never blank once attempted |

A `fail` is a finding, not a defeat: write what happened, not what should have.

---

## 1. Milestone sign-off

The roadmap has these implemented and awaiting a human. Each needs someone to
*use* it, not to read that it exists.

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 1.1 | **M2 — Polished game.** Play several full games. Menu, pause, help, restart, highscore entry, options (audio/music toggles), game over. Destructible bases, MIRV and smart-bomb threats, bonus cities all appear and behave | | | | |
| 1.2 | **M3 — Record & replay.** Record a run, load it, pause, scrub, change speed, and take over mid-replay and keep playing | | | | |
| 1.3 | **M4 — Scripted AI.** Watch LOW, MEDIUM and HIGH. The difference between them is visible as behaviour, not only as a score | | | | |

## 2. The first-time learning journey

The single most valuable route, and the one no unit test covers, because every
gap in it has been *between* features rather than inside one. Do it in order, on
a clean machine, without a terminal.

### The route, step by step

Budget **about 30 minutes**, excluding the training run's own wall time. Follow
it in order without skipping: the point is the joins between steps, and a step
taken out of order stops testing the join that precedes it.

Two rules while walking it. **Do not use a terminal** — if you reach for one, the
step has failed, and *that* is the finding. **Do not use knowledge you have from
building this project**; if a screen does not tell you what to do next, note it
rather than supplying the answer yourself.

| Step | Do this | Expect to see | Row |
|---|---|---|---|
| 1 | Install both packages. Launch **Missile Defense** from the desktop menu — not from a shell | The menu: START, WATCH AI, TRAIN AI, HELP, OPTIONS, HIGHSCORES, ABOUT, EXIT | 2.1 |
| 2 | **START**. Play one game to the end | Cities fall, the wave counter climbs, game over — and initials entry if you qualified | 2.2 |
| 3 | **WATCH AI → SCRIPTED LOW**. Then repeat with **SCRIPTED HIGH** | Two visibly different agents. LOW double-fires at warheads already doomed; HIGH does not. The difference must read as *behaviour*, not as a number | 2.3 |
| 4 | **WATCH AI → MODELS**, choose the bundled model | It plays, and it plays unlike either scripted rung — deeper, looser, more shots | 2.4 |
| 5 | Back to the menu, **TRAIN AI**. Then close the trainer and open it again from the desktop menu | The same window both ways. If a runtime must be installed first, the screen says so and offers **Set up training…** | 2.5 |
| 6 | Start a run from a preset, without opening **Advanced** | A run starts having asked you for nothing you could not answer | 2.6 |
| 7 | Read the estimate it gave you before starting, and note the clock | Whatever it promised about wall time and disk is what happens | 2.7 |
| 8 | Watch the curves for a few updates | Score, return, entropy and value move. What each one means is legible from the screen, not from this document | 2.8 |
| 9 | Let it reach a checkpoint. **Evaluate** it | A score *and* its protocol: which seed block, how many episodes, ranked or unranked | 2.9 |
| 10 | Promote the checkpoint, then **Head-to-head…** against the bundled model | Both on identical seeds, with the shared protocol named | 2.10 |
| 11 | **Watch the match?** → the split screen | Two viewports, one transport. Whose side is whose, both scores, the seed. Afterwards you can say *why* one side won | 2.11 |
| 12 | **Archive…** the run. Then **Restore a run** from the archive | The archive verifies before it offers to remove anything, and the restored run still has its curves, config and recordings | 2.12 |
| 13 | Look back over the whole walk | — | 2.13 |

Record the result of each step in the matching row below.

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 2.1 | Install the full package set and launch the game from the desktop menu | | | | |
| 2.2 | Play one game as a human | | | | |
| 2.3 | Watch the scripted AI at two skill levels | | | | |
| 2.4 | Watch the bundled learned model | | | | |
| 2.5 | Open the trainer — both directly and via **TRAIN AI** | | | | |
| 2.6 | Start the first preset run *without choosing an unexplained parameter* | | | | |
| 2.7 | The stated time and disk estimate matched what actually happened | | | | |
| 2.8 | Watch the curves move, and understand what they mean from the screen alone | | | | |
| 2.9 | Evaluate a checkpoint; the protocol (seeds, episodes, ranked or not) is stated | | | | |
| 2.10 | Compare it against the bundled model on identical seeds | | | | |
| 2.11 | Open the split-screen match and understand *why* one side won | | | | |
| 2.12 | Archive the run, then restore it | | | | |
| 2.13 | **No dead end, no empty panel without an explanation, and no lie about timing anywhere in 2.1–2.12** | | | | |

### Comprehension check

Ask someone who has not seen the project before to do §2, then to explain:

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 2.14 | …what an observation, an action and a reward are | | | | |
| 2.15 | …why training return and game score are different | | | | |
| 2.16 | …why evaluation uses held-out seeds and identical conditions | | | | |

If they cannot, the copy is wrong — not the person.

## 3. Packages: install → use → upgrade → uninstall

One row per supported package. **Upgrade** means from the previous public
artifact, not a fresh install.

| # | Package | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 3.1 | Debian 13 (trixie) — `missile-defense` | | | | |
| 3.2 | Debian 13 (trixie) — `missile-defense` + `missile-defense-trainer` | | | | |
| 3.3 | Ubuntu 26.04 — `missile-defense` | | | | |
| 3.4 | Ubuntu 26.04 — `missile-defense` + `missile-defense-trainer` | | | | |
| 3.5 | Ubuntu 24.04 — `missile-defense` **(game only; no training packages exist for it)** | | | | |
| 3.6 | Windows 10/11 x64 — NSIS installer | | | | |
| 3.7 | Windows 10/11 x64 — portable ZIP | | | | |
| 3.8 | macOS 14+ Apple silicon — `.dmg` | | | | |

For each: does uninstall leave the user's runs, recordings and highscores alone,
and say so?

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 3.9 | Ubuntu 24.04 says *in the product*, not only in a document, why TRAIN AI is unavailable there | | | | |
| 3.10 | macOS: the signing/notarisation status a user actually meets on first launch is what `docs/MACOS.md` says it is | | | | |

## 4. Real hardware

CI builds Windows and macOS; it does not launch a window on either, and it renders
on a software device with no GPU anywhere. Nothing below is covered by any green
tick that exists today.

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 4.1 | Windows: the game launches, renders and plays on real hardware | | | | |
| 4.2 | Windows: the trainer launches and can start a run | | | | |
| 4.3 | macOS 14+ Apple silicon: the game launches, renders and plays (MoltenVK) | | | | |
| 4.4 | macOS: the trainer launches and can start a run | | | | |
| 4.5 | An AMD GPU renders the game correctly | | | | |
| 4.6 | An Intel GPU renders the game correctly | | | | |
| 4.7 | An NVIDIA GPU renders the game correctly | | | | |
| 4.8 | **Frame pacing is smooth on the weakest supported GPU.** `Renderer::submit` serialises each frame to work around a `QVulkanWindow` defect; it measured free on an RTX 5090 and on lavapipe, both of which are bound elsewhere. A weak GPU is the case that measurement did not cover | | | | |

## 5. Look, sound, and readability

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 5.1 | The game looks good in motion — trails, blasts, terrain, starfield | | | | |
| 5.2 | Sound and music are pleasant, balanced, and the toggles work | | | | |
| 5.3 | A split-screen match is readable: whose side is whose, both scores, the seed, the protocol | | | | |
| 5.4 | Charts are readable, and meaning never depends on colour alone | | | | |
| 5.5 | Naming is consistent — **Missile Defense** everywhere, in both binaries, the installers, the desktop entries and the about text | | | | |

## 6. Accessibility

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 6.1 | Every screen of the **game** is fully operable from the keyboard | | | | |
| 6.2 | Every screen of the **trainer** is fully operable from the keyboard | | | | |
| 6.3 | Keyboard focus is always visible | | | | |
| 6.4 | Text scales with the desktop's font setting without clipping | | | | |
| 6.5 | Contrast is sufficient in both light and dark desktop themes | | | | |

## 7. Failure and recovery

Each of these must produce an explanation and a way forward — never a traceback,
a silent no-op, or a corrupted artifact.

| # | Item | Build | Who / when | Evidence | Result |
|---|---|---|---|---|---|
| 7.1 | Cancel a training run mid-update; restart it | | | | |
| 7.2 | Fill the disk during a run | | | | |
| 7.3 | Start the trainer with no torch installed | | | | |
| 7.4 | Start the trainer with no game installed | | | | |
| 7.5 | Open a recording or model from an incompatible build | | | | |
| 7.6 | Kill the training process outright, then reopen the trainer | | | | |
| 7.7 | Interrupt an archive or restore halfway through | | | | |
| 7.8 | Point the game at a corrupt `.mdr` recording | | | | |
| 7.9 | Run the game on a machine with no working Vulkan driver | | | | |

---

## What is already machine-checked

So that this list is not re-testing what CI covers. As of 2026-07-28:

- Documentation **links, `poe` task names and cited repository paths** all
  resolve, and no maintained document quotes a retired unhandicapped score as if
  it were current (`python/tests/test_docs.py`). That check cannot read prose, so
  §2 and §5 remain entirely a person's job.

- Debian trixie, Ubuntu 26.04 and Ubuntu 24.04 packages **build**, and the
  installed game starts under a virtual X server on a software Vulkan device.
- Windows and macOS artifacts **build**. Neither is launched anywhere.
- The renderer raises **no Vulkan validation message** — no allow-list — across
  menu, gameplay and both scripted skill levels, with synchronization validation
  enabled (`poe vulkan-runtime`). On lavapipe only, and **no GPU vendor's driver
  has ever been validated by any automated run**; §4 is where that gets checked.

  Read that claim with its history. Until 2026-07-27 the CI runner had no
  `vulkan-validationlayers` package, and `QVulkanInstance::setLayers` on a layer
  the loader does not have is silently ignored — so every validation assertion in
  CI passed by finding nothing to fail on. The gate now runs a canary first and
  refuses to report anything if the layer is not live.
- Shaders are `spirv-val`-clean against `vulkan1.0` (`poe vulkan-shaders`).
- The game package contains no Python; a game-only install stays game-only.

None of that is a substitute for a single row above.
