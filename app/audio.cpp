#include "audio.hpp"

#include "md/rng.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <miniaudio.h>
#include <mutex>
#include <vector>

namespace md {

namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kSampleRate = 48000.0f;
constexpr std::size_t kEventCount = 8; // EventType values, Fire..GameOver
constexpr std::size_t kVoiceCount = 16;

std::size_t ix(EventType type) noexcept {
    return static_cast<std::size_t>(type);
}

float sine(float freq, float t) noexcept {
    return std::sin(2.0f * kPi * freq * t);
}

float decay(float t, float rate) noexcept {
    return std::exp(-rate * t);
}

// Append `dur` seconds of samples produced by fn(t) to v.
template <class Fn> void add(std::vector<float>& v, float dur, Fn fn) {
    const std::size_t n = static_cast<std::size_t>(dur * kSampleRate);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(fn(static_cast<float>(i) / kSampleRate));
    }
}

std::array<std::vector<float>, kEventCount> build_sfx() {
    std::array<std::vector<float>, kEventCount> sfx;
    Pcg32 rng{7};
    const auto noise = [&rng] { return (rng.next_float() * 2.0f) - 1.0f; };

    // Fire — a short rising "pew".
    add(sfx[ix(EventType::Fire)], 0.12f,
        [](float t) { return 0.22f * sine(320.0f + (680.0f * (t / 0.12f)), t) * decay(t, 9.0f); });

    // Detonate — a soft pop.
    add(sfx[ix(EventType::Detonate)], 0.09f, [&noise](float t) {
        return ((0.22f * noise()) + (0.22f * sine(130.0f, t))) * decay(t, 26.0f);
    });

    // ThreatKilled — a satisfying crackle.
    add(sfx[ix(EventType::ThreatKilled)], 0.18f, [&noise](float t) {
        return ((0.32f * noise()) + (0.15f * sine(220.0f, t))) * decay(t, 13.0f);
    });

    // CityLost — a low, heavy boom.
    add(sfx[ix(EventType::CityLost)], 0.5f, [&noise](float t) {
        return ((0.35f * sine(70.0f, t)) + (0.25f * noise())) * decay(t, 6.0f);
    });

    // BaseLost — a deeper boom.
    add(sfx[ix(EventType::BaseLost)], 0.5f, [&noise](float t) {
        return ((0.35f * sine(52.0f, t)) + (0.22f * noise())) * decay(t, 6.0f);
    });

    // WaveCleared — a rising two-note chime.
    add(sfx[ix(EventType::WaveCleared)], 0.14f,
        [](float t) { return 0.24f * sine(523.0f, t) * decay(t, 6.0f); });
    add(sfx[ix(EventType::WaveCleared)], 0.20f,
        [](float t) { return 0.24f * sine(784.0f, t) * decay(t, 5.0f); });

    // BonusCity — a bright ding.
    add(sfx[ix(EventType::BonusCity)], 0.35f,
        [](float t) { return 0.24f * sine(1047.0f, t) * decay(t, 7.0f); });

    // GameOver — a descending tone.
    add(sfx[ix(EventType::GameOver)], 0.75f,
        [](float t) { return 0.28f * sine(660.0f - (500.0f * (t / 0.75f)), t) * decay(t, 2.5f); });

    return sfx;
}

} // namespace

struct AudioEngine::Impl {
    ma_device device{};
    bool running = false;
    std::array<std::vector<float>, kEventCount> sfx = build_sfx();

    struct Voice {
        const std::vector<float>* buffer = nullptr;
        std::size_t pos = 0;
    };

    std::array<Voice, kVoiceCount> voices{};
    std::mutex mutex;

    // Runs on the audio thread: sum the active voices into the output buffer.
    void mix(float* out, ma_uint32 frames) {
        for (ma_uint32 f = 0; f < frames; ++f) {
            out[f] = 0.0f;
        }
        const std::lock_guard<std::mutex> lock(mutex);
        for (auto& voice : voices) {
            if (voice.buffer == nullptr) {
                continue;
            }
            for (ma_uint32 f = 0; f < frames; ++f) {
                if (voice.pos >= voice.buffer->size()) {
                    voice.buffer = nullptr;
                    break;
                }
                out[f] += (*voice.buffer)[voice.pos];
                ++voice.pos;
            }
        }
        for (ma_uint32 f = 0; f < frames; ++f) {
            out[f] = std::clamp(out[f], -1.0f, 1.0f);
        }
    }

    void play(EventType type) {
        const std::size_t idx = ix(type);
        if (idx >= sfx.size() || sfx[idx].empty()) {
            return;
        }
        const std::lock_guard<std::mutex> lock(mutex);
        for (auto& voice : voices) {
            if (voice.buffer == nullptr) {
                voice.buffer = &sfx[idx];
                voice.pos = 0;
                return;
            }
        }
        voices[0].buffer = &sfx[idx]; // all busy: steal the oldest slot
        voices[0].pos = 0;
    }

    static void data_callback(ma_device* device, void* output, [[maybe_unused]] const void* input,
                              ma_uint32 frames) {
        static_cast<Impl*>(device->pUserData)->mix(static_cast<float*>(output), frames);
    }
};

AudioEngine::AudioEngine() : impl_{std::make_unique<Impl>()} {
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = 48000;
    config.dataCallback = &Impl::data_callback;
    config.pUserData = impl_.get();
    if (ma_device_init(nullptr, &config, &impl_->device) == MA_SUCCESS) {
        impl_->running = (ma_device_start(&impl_->device) == MA_SUCCESS);
    }
}

AudioEngine::~AudioEngine() {
    if (impl_->running) {
        ma_device_uninit(&impl_->device);
    }
}

void AudioEngine::handle_events(std::span<const Event> events) noexcept {
    if (!impl_->running) {
        return;
    }
    for (const auto& event : events) {
        impl_->play(event.type);
    }
}

} // namespace md
