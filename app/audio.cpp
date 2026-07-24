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
constexpr std::size_t kEventCount = 10; // EventType values, Fire..WaveStarted
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

// A thunder-like crash: a sharp initial crack, then a long low-passed noise
// rumble that rolls (slow amplitude swells) and fades. Two one-pole low-passes
// turn white noise into a deep brown-ish rumble; the whole thing is normalised
// to a stable peak so the mix level is predictable.
void add_thunder(std::vector<float>& v, Pcg32& rng) {
    constexpr float dur = 1.9f;
    const std::size_t n = static_cast<std::size_t>(dur * kSampleRate);
    std::vector<float> buf(n);
    float lp = 0.0f;
    float lp2 = 0.0f;
    float peak = 1e-6f;
    for (std::size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / kSampleRate;
        const float white = (rng.next_float() * 2.0f) - 1.0f;
        lp += 0.035f * (white - lp); // two-pole low-pass -> deep rumble
        lp2 += 0.035f * (lp - lp2);
        const float roll = 0.5f + (0.5f * std::sin((2.0f * kPi * 1.3f * t) + 0.7f) *
                                   std::sin(2.0f * kPi * 0.5f * t)); // rolling swells
        const float crack = white * std::exp(-32.0f * t);            // sharp strike
        const float sub = std::sin(2.0f * kPi * 42.0f * t) * std::exp(-2.2f * t);
        const float s = (7.0f * lp2 * roll * decay(t, 1.3f)) + (0.5f * crack) + (0.3f * sub);
        buf[i] = s;
        peak = std::max(peak, std::fabs(s));
    }
    const float gain = 0.9f / peak;
    for (const float s : buf) {
        v.push_back(s * gain);
    }
}

// A German WWII "Fliegeralarm" air-raid siren: a low tone whose pitch howls up
// and down, built from a rich harmonic stack (the mechanical rotor is nearly a
// pulse train) plus a slightly detuned second rotor for a rough, beating wail.
// It winds up from a low, ominous pitch. Peak-normalised. Deliberately menacing.
void add_siren(std::vector<float>& v) {
    constexpr float dur = 3.4f;
    constexpr float fc = 380.0f; // centre pitch (low)
    constexpr float fd = 150.0f; // wail depth -> sweeps ~230..530 Hz
    constexpr float fm = 0.42f;  // wail rate (~2.4 s per up-and-down)
    const auto rotor = [](float ph) {
        return std::sin(ph) + (0.6f * std::sin(2.0f * ph)) + (0.4f * std::sin(3.0f * ph)) +
               (0.24f * std::sin(4.0f * ph)) + (0.15f * std::sin(5.0f * ph));
    };
    const std::size_t n = static_cast<std::size_t>(dur * kSampleRate);
    std::vector<float> buf(n);
    float peak = 1e-6f;
    for (std::size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / kSampleRate;
        // Phase = exact integral of fc - fd*cos(2*pi*fm*t): starts low, winds up.
        const float phase = (2.0f * kPi * fc * t) - ((fd / fm) * std::sin(2.0f * kPi * fm * t));
        const float env = std::min(1.0f, t / 0.5f) * std::min(1.0f, (dur - t) / 0.6f);
        const float s = env * (rotor(phase) + (0.7f * rotor(phase * 1.008f)));
        buf[i] = s;
        peak = std::max(peak, std::fabs(s));
    }
    const float gain = 0.85f / peak;
    for (const float s : buf) {
        v.push_back(s * gain);
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

    // ThreatKilled — a punchy crackle with some low-end.
    add(sfx[ix(EventType::ThreatKilled)], 0.2f, [&noise](float t) {
        return ((0.3f * noise()) + (0.18f * sine(110.0f, t)) + (0.12f * sine(220.0f, t))) *
               decay(t, 11.0f);
    });

    // CityLost — thunder: a crack followed by a long rolling rumble.
    add_thunder(sfx[ix(EventType::CityLost)], rng);

    // BaseLost — an even deeper, longer boom.
    add(sfx[ix(EventType::BaseLost)], 0.9f, [&noise](float t) {
        const float f = 50.0f - (24.0f * (t / 0.9f)); // descending 50 -> 26 Hz
        const float sub = 0.6f * sine(f, t);
        const float body = 0.2f * sine(f * 1.5f, t);
        const float rumble = 0.3f * noise() * decay(t, 3.0f);
        return (sub + body + rumble) * decay(t, 3.3f);
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

    // MirvSplit — a warbling "shhk" as it fragments.
    add(sfx[ix(EventType::MirvSplit)], 0.22f, [&noise](float t) {
        const float warble = sine(620.0f + (140.0f * sine(30.0f, t)), t);
        return ((0.2f * warble) + (0.12f * noise())) * decay(t, 10.0f);
    });

    // WaveStarted — a menacing WWII air-raid siren.
    add_siren(sfx[ix(EventType::WaveStarted)]);

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
