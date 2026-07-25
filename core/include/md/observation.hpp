// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/config.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace md {

class Sim;

/// How the simulation is presented to a policy — one flat `float` vector, written
/// into a caller-owned buffer so a `VecSim` can fill a batch row zero-copy.
///
/// **What goes in is raw state, never analysis.** The rule (DESIGN.md §6): a
/// feature is admissible only if a human perceives it directly on screen.
/// Positions, velocities, types, ammo, cooldowns and the audio event stream all
/// qualify. Derived tactical quantities — time-to-impact, which city a threat is
/// aimed at, danger rankings, intercept points — do not: computing those *is* the
/// job we want the policy to learn, and handing them over would be doing the
/// hard part and then claiming the network discovered it.
///
/// Entity slots are emitted in the simulation's own array order, deliberately
/// *not* sorted by urgency: any ranking is itself a triage heuristic, and sorting
/// would smuggle it back in through the ordering.
struct ObsSpec {
    // Slots exposed per entity kind. The defaults are the simulation's own
    // capacities, so the policy always sees every entity the human can see.
    // Lowering them speeds training but truncates by slot index, which can hide a
    // live threat — an information asymmetry. Do that knowingly.
    std::uint32_t threats = max_threats;
    std::uint32_t interceptors = max_interceptors;
    std::uint32_t blasts = max_blasts;

    // Per-slot feature widths.
    static constexpr std::size_t threat_features = 9;      // present, pos2, vel2, type4
    static constexpr std::size_t interceptor_features = 7; // present, pos2, vel2, target2
    static constexpr std::size_t blast_features = 4;       // present, pos2, radius
    static constexpr std::size_t base_features = 4;        // alive, x, ammo, cooldown
    static constexpr std::size_t city_features = 2;        // alive, x
    static constexpr std::size_t global_features = 5;      // crosshair2, trigger, wave, score
    static constexpr std::size_t event_features = 10;      // per-EventType count this tick

    /// Total float count — the exact number `encode` writes.
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return (static_cast<std::size_t>(threats) * threat_features) +
               (static_cast<std::size_t>(interceptors) * interceptor_features) +
               (static_cast<std::size_t>(blasts) * blast_features) +
               (static_cast<std::size_t>(base_count) * base_features) +
               (static_cast<std::size_t>(max_cities) * city_features) + global_features +
               event_features;
    }
};

/// Write the observation for `sim` into `out` (exactly `spec.size()` floats).
///
/// Features are normalised into roughly [-1, 1]: positions against the world box,
/// velocities against `interceptor_speed`, timers against their own intervals.
/// Unused slots are zero-padded, and their leading `present` flag is 0. Writes
/// nothing if `out` is too small; allocation-free and `noexcept` throughout.
void encode(const Sim& sim, const ObsSpec& spec, std::span<float> out) noexcept;

} // namespace md
