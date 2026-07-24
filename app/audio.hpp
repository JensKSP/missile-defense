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

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_; // pimpl keeps miniaudio out of this header
};

} // namespace md
