# What the two agents taught

This project ended up running the same experiment twice: a hand-written
algorithmic agent and a learned one, on the same simulation, under the same
protocol. The result is the most interesting thing in the repository, and it is
not the score — it is that **the answer reverses depending on one rule.**

Everything below is measured on the **held-out canonical block** under the
published handicap, unless a paragraph says otherwise. The handicap holds every
agent to the crosshair speed, trigger interval and 15 Hz decision rate a human
hand is held to; [docs/API.md](API.md) defines it and
[docs/TRAINING.md](TRAINING.md) the evaluation protocol.

## The two agents

The scripted agent is a few hundred lines of geometry. The learned one is PPO
with a relational attention network over threats, interceptors and blasts — a
1,959-float observation, 385 actions, an hour on an RTX 5090. On the same
held-out block, both handicapped, the scripted agent scores **13,687** and the
learned policy **23,067**. The interesting part is not the gap but its *shape*:

| | Scripted HIGH | Learned |
|---|---|---|
| Mean score | 13,687 | **23,067** |
| Mean wave reached | 7.16 | **8.91** |
| Kills per interceptor | **0.73** | 0.61 |
| Wasted shots | **36%** | 44% |

**It wins on depth while still losing on marksmanship.** Putting a blast where a
warhead is going to be is a closed-form intercept problem, and a human writes it
once, exactly; a network has to recover the same geometry from a scalar reward,
which is a spectacularly indirect way to learn ballistics, and it never quite
does. What it does instead is spend: it fires more, hits less often, and gets two
waves deeper for it.

## The reversal

**Without the handicap, the result flips.** Given an agent that never mis-clicks
and never waits, the geometry *is* the game: the same comparison ran **98,542 to
90,866 in the scripted agent's favour.**

> Those two numbers are from the **unhandicapped** protocol and are retired. They
> are recorded here because the comparison is the finding; they are not the
> project's benchmark, and anything quoting 98,542 as a current score is stale.
> The current numbers are 13,687 and 23,067, above.

So the conclusion is narrower than "write the algorithm" and more useful:
**where a good algorithmic solution exists it wins exactly as far as its
preconditions hold** — and a closed-form aimer's precondition is a perfect hand.
Learning earns its keep on the part with no closed form, allocation under a fixed
ammunition budget, and under the handicap that is enough to win.

## The one idea the scripted agent has

Both protocols agree on what carries the hand-written agent, and it is a single
idea: **do not shoot what is already dead.**

`Params::coverage_horizon` is the dial — how many seconds ahead the agent
remembers the shots it has already fired. At HIGH it tracks every interceptor in
flight and never fires twice at a warhead that is already doomed; at LOW it
tracks none and wastes over two thirds of its ammunition. That one behaviour is
the *entire* spread of the scripted ladder: worth about **78,000** points
unhandicapped, and **8,663** of HIGH's 13,687 under the handicap.

Two things fell out of measuring it, both from the unhandicapped runs — so the
absolute figures are retired and the shape is what carries over:

- **The response is a cliff, not a slope.** 0.30 s scored ~34k and 0.40 s ~85k,
  because that is where the dial crosses a typical interceptor's flight time and
  the agent either remembers a shot before it lands or does not.
- **The sophisticated-looking part is worth almost nothing.** Switching off
  `cluster_bonus`, which deliberately waits for MIRV spreads to converge, cost
  about **1,500** points against ammunition memory's ~78,000.

## What this does not settle

Two things stop the above from being the whole story.

**The learned policy got there with no game-specific knowledge at all.** Nobody
told it what a MIRV is, or that ammunition is scarce. It would retrain unchanged
against a game whose wave table or weapons you altered, where the scripted agent
would have to be rewritten by hand. That generality is not visible in either
score.

**The strongest version of this system is probably neither one alone** — a
scripted aimer under a learned allocator, each owning the half it is actually
good at. That is the experiment this repository is now set up to run and has not
run yet.
