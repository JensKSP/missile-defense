# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""When to evaluate and when to record — dense early, settled later.

A fixed interval samples a run's least interesting part most thoroughly. The
first hundred updates are where a policy changes shape; by update 800 it is
inching, and measuring every 20 there buys almost nothing while costing the same
as it did at the start. An evaluation is not free — early on it costs most of an
update — so the sampling ought to follow where the information is.

So the cadence *ramps*: the gap starts at one update and grows until it reaches
the configured interval at ``ramp_until``, and stays there.

**The schedule is a function of the update number alone.** That is deliberate,
and it is the difference from the obvious alternative of backing off once the
curve looks flat. Backing off on flatness means measuring less *because* nothing
is happening — and then being slow to notice when something starts happening
again. It would also make two runs with the same settings sample at different
points, so their curves could not be laid over each other. This way they can.
"""

from __future__ import annotations

#: Sampling every update forever is the honest reading of "no ramp".
NO_RAMP = 0


def gap_at(update: int, *, interval: int, ramp_until: int) -> int:
    """How many updates to wait after ``update`` before sampling again.

    Grows linearly in ``update`` — which spaces the samples geometrically, the
    "logarithmic" shape this is for — and is clamped so it never drops below one
    update or rises above the configured interval.
    """
    if interval <= 0:
        return 0
    if ramp_until <= NO_RAMP:
        return interval
    scaled = round(interval * update / ramp_until)
    return max(1, min(interval, scaled))


def schedule(*, interval: int, ramp_until: int, last: int) -> list[int]:
    """Every update at which to sample, from 1 to ``last`` inclusive.

    Walked rather than solved: the gap depends on where the previous sample
    landed, and a closed form for that is harder to read than the loop and no
    faster at these sizes.
    """
    if interval <= 0:
        return []
    points: list[int] = []
    update = 1
    while update <= last:
        points.append(update)
        update += gap_at(update, interval=interval, ramp_until=ramp_until)
    return points


def is_due(update: int, *, interval: int, ramp_until: int) -> bool:
    """Whether ``update`` is a sampling point.

    Derived from ``update`` and nothing else, so a resumed run picks the schedule
    back up where the original would have been rather than starting its ramp
    again at whatever update it resumed on. That bug is invisible in a fresh run
    and obvious in a chart with a second dense patch in the middle of it.
    """
    if interval <= 0:
        return False
    at = 1
    while at < update:
        at += gap_at(at, interval=interval, ramp_until=ramp_until)
    return at == update
