// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "audio.hpp"

#include "md/rng.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <miniaudio.h>
#include <mutex>
#include <numbers>
#include <utility>
#include <vector>

#ifdef _WIN32
// Audio device init must run in a COM multi-threaded apartment — see AudioEngine().
#define NOMINMAX
#include <objbase.h>
#include <thread>
#endif

namespace md {

namespace {

constexpr float kPi = std::numbers::pi_v<float>;
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
    const auto n = static_cast<std::size_t>(dur * kSampleRate);
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
    const auto n = static_cast<std::size_t>(dur * kSampleRate);
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

// A German E57 motor air-raid siren. The rotor is driven by a motor that winds
// up quickly, holds near full speed, then coasts down slowly — an asymmetric
// wail (not a symmetric sine). The chopped airflow is nearly a pulse train, so
// the tone is rich in harmonics ("reedy"); a slightly detuned second rotor adds
// a rough, beating chorus. Phase is accumulated per sample so any pitch curve
// stays click-free. Amplitude swells with speed. Peak-normalised. Menacing.
void add_siren(std::vector<float>& v) {
    constexpr float dur = 4.2f;
    constexpr float cycle = 2.8f;    // one wind-up + wind-down
    constexpr float spin_up = 0.26f; // fraction of the cycle spent accelerating
    constexpr float hold = 0.22f;    // fraction held near full speed
    constexpr float f_lo = 190.0f;   // idle pitch
    constexpr float f_hi = 450.0f;   // full-speed pitch (E57 territory)
    const auto rotor = [](float ph) {
        return std::sin(ph) + (0.7f * std::sin(2.0f * ph)) + (0.5f * std::sin(3.0f * ph)) +
               (0.34f * std::sin(4.0f * ph)) + (0.22f * std::sin(5.0f * ph)) +
               (0.12f * std::sin(6.0f * ph));
    };
    const auto n = static_cast<std::size_t>(dur * kSampleRate);
    const float dt = 1.0f / kSampleRate;
    std::vector<float> buf(n);
    float peak = 1e-6f;
    float phase = 0.0f;
    float phase2 = 0.0f; // detuned rotor
    for (std::size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) * dt;
        const float u = std::fmod(t, cycle) / cycle; // position within the wail cycle
        float s = 0.0f;
        if (u < spin_up) {
            s = u / spin_up; // fast wind-up
        } else if (u < spin_up + hold) {
            s = 1.0f; // hold at full speed
        } else {
            s = 1.0f - ((u - spin_up - hold) / (1.0f - spin_up - hold)); // slow coast-down
        }
        s = s * s * (3.0f - (2.0f * s)); // smoothstep -> rounded wail
        const float freq = f_lo + ((f_hi - f_lo) * s);
        phase += 2.0f * kPi * freq * dt;
        phase2 += 2.0f * kPi * freq * 1.007f * dt;
        const float edges = std::min(1.0f, t / 0.6f) * std::min(1.0f, (dur - t) / 0.7f);
        const float amp = 0.55f + (0.45f * s); // swells as it winds up
        const float sample = edges * amp * (rotor(phase) + (0.7f * rotor(phase2)));
        buf[i] = sample;
        peak = std::max(peak, std::fabs(sample));
    }
    const float gain = 0.82f / peak;
    for (const float sample : buf) {
        v.push_back(sample * gain);
    }
}

// A short looping chiptune track: tense A-minor FM synthesis — a driving bass, a
// bright sixteenth-note arpeggio, a sparse bell lead, and noise drums. Rendered
// once; the mixer loops it seamlessly (note tails wrap around the loop end).
std::vector<float> build_music() {
    constexpr float bpm = 142.0f;
    constexpr int steps_per_bar = 16; // sixteenth-note grid
    constexpr int bars = 8;
    constexpr int total_steps = steps_per_bar * bars;
    const auto step_n = static_cast<std::size_t>((60.0f / bpm / 4.0f) * kSampleRate);
    const std::size_t total_n = step_n * static_cast<std::size_t>(total_steps);
    std::vector<float> buf(total_n, 0.0f);

    struct Chord {
        int bass;
        std::array<int, 4> arp;
    };

    // i - VI - VII - V in A minor (the major V = E adds the tension), 2 bars each.
    const std::array<Chord, 4> prog = {{
        {33, {{57, 60, 64, 69}}}, // Am
        {29, {{53, 57, 60, 65}}}, // F
        {31, {{55, 59, 62, 67}}}, // G
        {28, {{52, 56, 59, 64}}}, // E
    }};

    const auto midi = [](int n) {
        return 440.0f * std::pow(2.0f, static_cast<float>(n - 69) / 12.0f);
    };

    // A 2-operator FM voice with an exponential decay, added into the looping buffer.
    const auto add_note = [&](std::size_t start, std::size_t dur, float freq, float ratio,
                              float index, float decay_rate, float amp) {
        for (std::size_t i = 0; i < dur; ++i) {
            const float t = static_cast<float>(i) / kSampleRate;
            const float env = std::exp(-decay_rate * t);
            const float mod = index * std::sin(2.0f * kPi * freq * ratio * t);
            buf[(start + i) % total_n] += amp * env * std::sin((2.0f * kPi * freq * t) + mod);
        }
    };

    Pcg32 drng{99};
    const auto add_kick = [&](std::size_t start) {
        for (std::size_t i = 0; i < step_n * 2; ++i) {
            const float t = static_cast<float>(i) / kSampleRate;
            const float pitch = (140.0f * std::exp(-32.0f * t)) + 46.0f; // pitch drop
            buf[(start + i) % total_n] +=
                0.5f * std::exp(-8.0f * t) * std::sin(2.0f * kPi * pitch * t);
        }
    };
    const auto add_noise_hit = [&](std::size_t start, float amp, float decay_rate) {
        for (std::size_t i = 0; i < step_n; ++i) {
            const float t = static_cast<float>(i) / kSampleRate;
            buf[(start + i) % total_n] +=
                amp * std::exp(-decay_rate * t) * ((drng.next_float() * 2.0f) - 1.0f);
        }
    };

    for (int s = 0; s < total_steps; ++s) {
        const std::size_t at = step_n * static_cast<std::size_t>(s);
        const int bar = s / steps_per_bar;
        const Chord& ch = prog[static_cast<std::size_t>((bar / 2) % 4)];
        const int in_bar = s % steps_per_bar;
        const auto arp_note = ch.arp[static_cast<std::size_t>(s % 4)];

        if (in_bar % 2 == 0) { // driving eighth-note FM bass on the chord root
            add_note(at, step_n * 2, midi(ch.bass), 1.0f, 3.0f, 4.5f, 0.30f);
        }
        // Sixteenth-note bright arpeggio through the chord tones.
        add_note(at, step_n, midi(arp_note), 2.0f, 2.2f, 8.0f, 0.13f);
        if (in_bar == 6 || in_bar == 14) { // sparse bell lead, an octave up
            add_note(at, step_n * 3, midi(arp_note + 12), 3.5f, 3.5f, 5.0f, 0.10f);
        }
        // Drums: kick on beats 1 & 3, snare on 2 & 4, hats on the offbeats.
        if (in_bar == 0 || in_bar == 8) {
            add_kick(at);
        }
        if (in_bar == 4 || in_bar == 12) {
            add_noise_hit(at, 0.22f, 22.0f); // snare
        }
        if (in_bar % 2 == 1) {
            add_noise_hit(at, 0.07f, 55.0f); // hi-hat
        }
    }

    float peak = 1.0e-6f;
    for (const float v : buf) {
        peak = std::max(peak, std::fabs(v));
    }
    const float gain = 0.9f / peak;
    for (float& v : buf) {
        v *= gain;
    }
    return buf;
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
    std::atomic<bool> enabled{true};  // SFX
    std::atomic<bool> music_on{true}; // background music
    std::vector<float> music = build_music();
    std::size_t music_pos = 0; // audio thread only

    // Runs on the audio thread: sum the active SFX voices + looping music.
    void mix(float* out, ma_uint32 frames) {
        for (ma_uint32 f = 0; f < frames; ++f) {
            out[f] = 0.0f;
        }
        if (enabled.load(std::memory_order_relaxed)) {
            const std::scoped_lock lock(mutex);
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
        }
        if (music_on.load(std::memory_order_relaxed) && !music.empty()) {
            constexpr float music_gain = 0.38f; // sits under the SFX
            for (ma_uint32 f = 0; f < frames; ++f) {
                out[f] += music[music_pos] * music_gain;
                if (++music_pos >= music.size()) {
                    music_pos = 0; // seamless loop
                }
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
        const std::scoped_lock lock(mutex);
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

namespace {

// Initialise the miniaudio device, running `fn` in a context where COM is usable
// for WASAPI. Qt puts the GUI thread in a single-threaded apartment (STA), where
// miniaudio's WASAPI device enumeration faults; run it on a throwaway thread that
// joins a multi-threaded apartment (MTA) instead. On other platforms, run inline.
template <class Fn> void init_audio_device(Fn&& fn) {
#ifdef _WIN32
    std::thread worker([&] {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        std::forward<Fn>(fn)(); // invoked exactly once, on this thread
        if (SUCCEEDED(hr)) {
            CoUninitialize();
        }
    });
    worker.join(); // join before returning: synchronises fn's writes to impl_
#else
    std::forward<Fn>(fn)();
#endif
}

} // namespace

AudioEngine::AudioEngine() : impl_{std::make_unique<Impl>()} {
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = 48000;
    config.dataCallback = &Impl::data_callback;
    config.pUserData = impl_.get();
    init_audio_device([&] {
        // Report failures. A game that is silently mute is indistinguishable from
        // one the player muted, so a failed device has to say so — otherwise the
        // only symptom is "no audio" with nothing to go on.
        const ma_result opened = ma_device_init(nullptr, &config, &impl_->device);
        if (opened != MA_SUCCESS) {
            std::fputs("audio: no playback device: ", stderr);
            std::fputs(ma_result_description(opened), stderr);
            std::fputs(" (is a default output device set and enabled?)\n", stderr);
            return;
        }
        const ma_result started = ma_device_start(&impl_->device);
        if (started != MA_SUCCESS) {
            std::fputs("audio: device opened but would not start: ", stderr);
            std::fputs(ma_result_description(started), stderr);
            std::fputs("\n", stderr);
            ma_device_uninit(&impl_->device);
            return;
        }
        impl_->running = true;
    });
}

AudioEngine::~AudioEngine() {
    if (impl_->running) {
        ma_device_uninit(&impl_->device);
    }
}

void AudioEngine::handle_events(std::span<const Event> events) noexcept {
    if (!impl_->running || !impl_->enabled.load(std::memory_order_relaxed)) {
        return;
    }
    for (const auto& event : events) {
        impl_->play(event.type);
    }
}

void AudioEngine::set_enabled(bool on) noexcept {
    impl_->enabled.store(on, std::memory_order_relaxed);
    if (!on) { // silence anything already playing
        const std::scoped_lock lock(impl_->mutex);
        for (auto& voice : impl_->voices) {
            voice.buffer = nullptr;
        }
    }
}

bool AudioEngine::enabled() const noexcept {
    return impl_->enabled.load(std::memory_order_relaxed);
}

void AudioEngine::set_music_enabled(bool on) noexcept {
    impl_->music_on.store(on, std::memory_order_relaxed);
}

bool AudioEngine::music_enabled() const noexcept {
    return impl_->music_on.load(std::memory_order_relaxed);
}

} // namespace md
