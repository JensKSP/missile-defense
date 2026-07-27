// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace md {

/// The landscape the field is played on: a ridge line the cities and batteries
/// stand *on*, rather than a flat bar they float above.
///
/// Two rules shape it. Every installation gets a level plateau wide enough for
/// its whole footprint — a row of towers planted across a slope would each start
/// at a different height, which reads as a bug and not as terrain. And the
/// batteries stand on higher ground than the towns, the way the arcade original
/// puts its three launchers on mounds.
///
/// Cosmetic only. `md::core` still resolves every landing at `y = 0` and knows
/// nothing about this, so no hill here can change what a shot hits — which is
/// also why it lives in `app/` and is a pure heightfield with no Vulkan in it.
class Terrain {
  public:
    // The shape, in world units. Towns sit low and the batteries on mounds; the
    // ground between them rises into hills, and rolls up to a shoulder at either
    // edge of the frame so the field reads as a valley rather than as a strip.
    static constexpr float town_level_min = 2.4f;
    static constexpr float town_level_span = 1.2f;
    static constexpr float battery_level_min = 4.3f;
    static constexpr float battery_level_span = 0.8f;
    /// Deliberately a wide range. Nine evenly spaced plateaus with a matching
    /// bump between each is a scallop, not a landscape — it is hills that differ
    /// from their neighbours, in height and in where they peak, that stop the eye
    /// from finding the repeat.
    static constexpr float crest_min = 1.5f;
    static constexpr float crest_span = 3.5f;
    static constexpr float peak_min = 0.38f; // where along a gap its hill tops out
    static constexpr float peak_span = 0.24f;
    static constexpr float shoulder = 4.0f;
    /// Half-width of a plateau, as a fraction of the spacing between two
    /// installations. 0.3 of the nine-slot spacing is ~10 world units either
    /// side, which clears the widest footprint (a city's 7) with room to spare.
    static constexpr float plateau_fraction = 0.30f;

    Terrain() = default;

    /// Build the field from where the installations actually stand, rather than
    /// from a second copy of the slot formula in `Sim::reset` — one layout, one
    /// source, and a test that plants a real `Sim` on it catches any drift.
    Terrain(float world_width, std::span<const float> city_x, std::span<const float> base_x)
        : width_{world_width} {
        sites_.reserve(city_x.size() + base_x.size());
        for (const float x : city_x) {
            sites_.push_back(Site{.x = x, .level = 0.0f, .battery = false});
        }
        for (const float x : base_x) {
            sites_.push_back(Site{.x = x, .level = 0.0f, .battery = true});
        }
        std::ranges::sort(sites_, {}, &Site::x);
        if (sites_.empty()) {
            return;
        }

        flat_ = (width_ / static_cast<float>(sites_.size())) * plateau_fraction;

        // Seeded rather than hand-placed: the hills only have to look unplanned,
        // and a fixed seed keeps the same landscape across runs — and across both
        // halves of a split-screen match, which build their backdrops separately.
        Pcg32 rng{20260727};
        for (Site& site : sites_) {
            site.level = site.battery ? battery_level_min + rng.uniform(0.0f, battery_level_span)
                                      : town_level_min + rng.uniform(0.0f, town_level_span);
        }
        hills_.reserve(sites_.size() - 1);
        for (std::size_t i = 1; i < sites_.size(); ++i) {
            hills_.push_back(Hill{.crest = crest_min + rng.uniform(0.0f, crest_span),
                                  .peak = peak_min + rng.uniform(0.0f, peak_span)});
        }
    }

    /// Surface height of the ground at world x — where anything standing there
    /// has its feet.
    [[nodiscard]] float height(float x) const noexcept {
        if (sites_.empty()) {
            return town_level_min;
        }
        const Site& first = sites_.front();
        const Site& last = sites_.back();
        // Outside the outermost plateaus the ground climbs to the frame's edge.
        if (x <= first.x - flat_) {
            return first.level +
                   (shoulder * smoothstep(ramp(first.x - flat_ - x, first.x - flat_)));
        }
        if (x >= last.x + flat_) {
            return last.level +
                   (shoulder * smoothstep(ramp(x - (last.x + flat_), width_ - (last.x + flat_))));
        }
        if (x <= first.x + flat_) {
            return first.level;
        }
        for (std::size_t i = 1; i < sites_.size(); ++i) {
            const Site& prev = sites_[i - 1];
            const Site& site = sites_[i];
            if (x < site.x - flat_) {
                // The hill between two plateaus: the levels blend across it, and a
                // crest rides on top. Both terms flatten out at either end, so the
                // hill meets the level ground without a crease.
                const float u = ramp(x - (prev.x + flat_), (site.x - flat_) - (prev.x + flat_));
                const Hill& hill = hills_[i - 1];
                return prev.level + ((site.level - prev.level) * smoothstep(u)) +
                       (hill.crest * hump(u, hill.peak));
            }
            if (x <= site.x + flat_) {
                return site.level; // the level ground an installation stands on
            }
        }
        return last.level; // unreachable: the shoulder above already caught this
    }

    /// A second, distant ridge — pure decoration, drawn behind the ground in a
    /// tone barely off the sky. It is what turns a silhouette into a landscape,
    /// and it is a plain sum of sines because nothing ever stands on it.
    [[nodiscard]] static float ridge(float x) noexcept {
        return 7.6f + (3.0f * std::sin((0.0215f * x) + 0.9f)) +
               (2.0f * std::sin((0.0471f * x) + 2.7f)) + (0.9f * std::sin((0.113f * x) + 5.1f));
    }

    /// How far level ground extends either side of an installation's centre.
    [[nodiscard]] float plateau_half_width() const noexcept { return flat_; }

  private:
    /// One installation and the level ground it stands on.
    struct Site {
        float x = 0.0f;
        float level = 0.0f;
        bool battery = false;
    };

    /// The hill filling one gap: how high it rises, and how far along it does so.
    struct Hill {
        float crest = 0.0f;
        float peak = 0.5f;
    };

    /// Position within a span, clamped to [0, 1] and safe on a zero-length one.
    [[nodiscard]] static float ramp(float offset, float span) noexcept {
        return std::clamp(offset / std::max(span, 1.0e-3f), 0.0f, 1.0f);
    }

    /// Smooth 0 -> 1 with zero slope at both ends.
    [[nodiscard]] static float smoothstep(float u) noexcept { return u * u * (3.0f - (2.0f * u)); }

    /// A hill: 0 at both ends, 1 at `peak`, flat where it meets the plateaus.
    /// Skewing where the summit falls is what gives each one its own profile —
    /// a long shallow climb into a short drop, or the reverse.
    [[nodiscard]] static float hump(float u, float peak) noexcept {
        const float skewed =
            (u < peak) ? (0.5f * ramp(u, peak)) : (0.5f + (0.5f * ramp(u - peak, 1.0f - peak)));
        const float s = std::sin(std::numbers::pi_v<float> * skewed);
        return s * s;
    }

    std::vector<Site> sites_;
    std::vector<Hill> hills_; // one hill per gap between neighbouring sites
    float width_ = 0.0f;
    float flat_ = 0.0f;
};

} // namespace md
