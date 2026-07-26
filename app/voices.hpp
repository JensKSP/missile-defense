// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace md {

/// A fixed bank of sound-effect playback slots, mixed additively.
///
/// Split out of AudioEngine so the one part of the audio path with decisions in
/// it can be tested without a sound card: what happens when more sounds arrive
/// than there are slots. That is not a corner case — the losing final wave emits
/// six CityLost, three BaseLost, a detonation per interceptor in flight and a
/// GameOver within a single tick, against sixteen slots.
///
/// Nothing here locks or allocates. The engine owns the mutex, because it is the
/// engine that knows one side is a real-time callback (see AudioEngine::Impl).
class VoiceBank {
  public:
    /// Sixteen simultaneous effects. Beyond this the mix is mud anyway, and the
    /// interesting question stops being "how many" and becomes "which to drop".
    static constexpr std::size_t capacity = 16;

    /// Begin playing `samples`, displacing another voice if all slots are busy.
    ///
    /// The displaced voice is the one **nearest to finishing**, which is the
    /// choice that loses the least audio: it has the fewest samples left to
    /// play. The obvious alternative — always taking slot 0 — is what this
    /// replaces, and it is worse than it looks. During a cascade it re-triggers
    /// whatever is in slot 0 once per event, so a sound that had barely started
    /// is restarted from zero dozens of times in a row instead of a full bank
    /// quietly dropping the excess.
    void start(std::span<const float> samples) noexcept {
        if (samples.empty()) {
            return; // a sound with no samples would occupy a slot forever
        }
        Voice* chosen = nullptr;
        float most_played = -1.0f;
        for (Voice& voice : voices_) {
            if (voice.samples.empty()) {
                chosen = &voice; // a free slot always wins; nothing is displaced
                break;
            }
            // Fraction rather than absolute position: a 4.2 s siren 90 % through
            // has less left than a 0.09 s pop that has just begun, and it is
            // what is *left* that a steal costs.
            const float played =
                static_cast<float>(voice.pos) / static_cast<float>(voice.samples.size());
            if (played > most_played) {
                most_played = played;
                chosen = &voice;
            }
        }
        if (chosen != nullptr) {
            chosen->samples = samples;
            chosen->pos = 0;
        }
    }

    /// Add every playing voice into `out`, releasing those that end.
    ///
    /// Additive: the caller has already written whatever else belongs in the
    /// buffer (the music bed), and clamping is its job too, since it is the one
    /// that knows what else went in.
    void mix(std::span<float> out) noexcept {
        for (Voice& voice : voices_) {
            if (voice.samples.empty()) {
                continue;
            }
            const std::size_t left = voice.samples.size() - voice.pos;
            const std::size_t n = left < out.size() ? left : out.size();
            for (std::size_t i = 0; i < n; ++i) {
                out[i] += voice.samples[voice.pos + i];
            }
            voice.pos += n;
            if (voice.pos >= voice.samples.size()) {
                voice.samples = {}; // done — free the slot for the next sound
            }
        }
    }

    /// Stop everything immediately (the audio switch being turned off).
    void silence() noexcept {
        for (Voice& voice : voices_) {
            voice.samples = {};
        }
    }

    [[nodiscard]] std::size_t active() const noexcept {
        std::size_t n = 0;
        for (const Voice& voice : voices_) {
            n += voice.samples.empty() ? 0U : 1U;
        }
        return n;
    }

    /// How far slot `index` has played, in samples. For tests and diagnostics.
    [[nodiscard]] std::size_t played_of(std::size_t index) const noexcept {
        return voices_[index].pos;
    }

  private:
    struct Voice {
        std::span<const float> samples; //< empty means the slot is free
        std::size_t pos = 0;
    };

    std::array<Voice, capacity> voices_{};
};

} // namespace md
