// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/event.hpp"

#include <memory>
#include <span>

namespace md {

/// Plays procedurally-generated retro sound effects for game events. Owns a
/// miniaudio playback device with a small software voice mixer. Degrades
/// silently (no crash) if no audio device is available.
class AudioEngine {
  public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    AudioEngine(AudioEngine&&) = delete;
    AudioEngine& operator=(AudioEngine&&) = delete;

    /// Play the sound effect for each event in the step's event stream.
    void handle_events(std::span<const Event> events) noexcept;

    /// Enable or mute all sound effects (silences any currently-playing voices).
    void set_enabled(bool on) noexcept;

    [[nodiscard]] bool enabled() const noexcept;

    /// Enable or mute the looping background music (independent of the SFX).
    void set_music_enabled(bool on) noexcept;

    [[nodiscard]] bool music_enabled() const noexcept;

    /// Has the audio callback *ever* been allowed to mix a sample?
    ///
    /// Sticky, and latched on the audio thread itself, which is what makes it
    /// worth reporting. The state at any one moment cannot catch the fault this
    /// exists after: a `--silent` run played music from inside `GameWindow`'s
    /// constructor until `main` got as far as parsing the flag, and by the time
    /// anything asked "is it audible?" the answer was honestly no. What
    /// `--silent` promises is that the answer was never yes.
    [[nodiscard]] bool ever_audible() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_; // pimpl keeps miniaudio out of this header
};

} // namespace md
