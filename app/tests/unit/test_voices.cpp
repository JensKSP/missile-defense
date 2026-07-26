// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// The voice bank, which is the half of the audio engine that can be reasoned
// about without a sound card. What it has to get right is what happens when
// more sounds arrive than there are slots to play them in — which is not a
// hypothetical: the game-over cascade emits six CityLost, three BaseLost, a
// detonation per interceptor in flight and a GameOver, all in one tick, against
// sixteen slots.
#include "voices.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <span>
#include <vector>

namespace {

std::vector<float> ramp(std::size_t n, float value = 1.0f) {
    return std::vector<float>(n, value);
}

} // namespace

TEST_CASE("A voice plays its samples and then frees its slot", "[unit][app][audio]") {
    md::VoiceBank bank;
    const std::vector<float> sound = ramp(3, 0.5f);
    bank.start(sound);
    REQUIRE(bank.active() == 1);

    std::array<float, 3> out{};
    bank.mix(out);
    CHECK(out[0] == 0.5f);
    CHECK(out[2] == 0.5f);
    // Played to its end, so the slot must be free again — otherwise a game
    // leaks voices until every sound is competing for a stolen one.
    CHECK(bank.active() == 0);
}

TEST_CASE("Mixing adds to the buffer rather than replacing it", "[unit][app][audio]") {
    md::VoiceBank bank;
    const std::vector<float> a = ramp(2, 0.25f);
    const std::vector<float> b = ramp(2, 0.5f);
    bank.start(a);
    bank.start(b);

    std::array<float, 2> out{1.0f, 1.0f};
    bank.mix(out);
    CHECK(out[0] == 1.75f);
    CHECK(out[1] == 1.75f);
}

TEST_CASE("A voice continues where the previous buffer left off", "[unit][app][audio]") {
    md::VoiceBank bank;
    const std::vector<float> sound{1.0f, 2.0f, 3.0f, 4.0f};
    bank.start(sound);

    std::array<float, 2> first{};
    bank.mix(first);
    CHECK(first[0] == 1.0f);
    CHECK(first[1] == 2.0f);

    std::array<float, 2> second{};
    bank.mix(second);
    CHECK(second[0] == 3.0f);
    CHECK(second[1] == 4.0f);
    CHECK(bank.active() == 0);
}

TEST_CASE("A full bank steals the voice nearest to finishing", "[unit][app][audio]") {
    md::VoiceBank bank;
    // Fill every slot with a long sound, then advance exactly one of them most
    // of the way through, so there is an unambiguous "nearest done".
    const std::vector<float> just_started = ramp(1000, 1.0f);
    const std::vector<float> nearly_done = ramp(4, 1.0f);
    for (std::size_t i = 0; i < md::VoiceBank::capacity - 1; ++i) {
        bank.start(just_started);
    }
    bank.start(nearly_done);
    REQUIRE(bank.active() == md::VoiceBank::capacity);

    std::array<float, 3> drain{};
    bank.mix(drain); // nearly_done now has one sample left; the rest have 997
    REQUIRE(bank.active() == md::VoiceBank::capacity);

    // The seventeenth sound has to displace something. Taking slot 0 — which is
    // what the code did before this test existed — restarts a sound that had
    // barely begun, and doing that once per event through a 128-event cascade
    // is a slot being re-triggered over and over rather than a bank that is
    // simply full. The one with least left to play is the one to lose.
    const std::vector<float> arriving = ramp(500, 1.0f);
    bank.start(arriving);
    CHECK(bank.active() == md::VoiceBank::capacity);
    CHECK(bank.played_of(0) > 0); // slot 0 kept its progress
}

TEST_CASE("A cascade larger than the bank never exceeds its capacity", "[unit][app][audio]") {
    md::VoiceBank bank;
    const std::vector<float> sound = ramp(500, 0.1f);
    // `Sim` hands over at most max_events per step, and a losing final wave
    // gets close to it. However many arrive, the bank is fixed.
    for (int i = 0; i < 128; ++i) {
        bank.start(sound);
    }
    CHECK(bank.active() == md::VoiceBank::capacity);

    std::array<float, 8> out{};
    bank.mix(out);
    // Sixteen voices at 0.1 is 1.6; a hundred and twenty-eight would be 12.8.
    // The bound is the seventeenth voice rather than 1.6 exactly, because
    // summing sixteen 0.1f lands a rounding step above it. The clamp itself
    // lives in the engine, which is the half that knows what else went in.
    CHECK(out[0] < 1.7f);
}

TEST_CASE("Silencing frees every slot at once", "[unit][app][audio]") {
    md::VoiceBank bank;
    const std::vector<float> sound = ramp(500, 1.0f);
    bank.start(sound);
    bank.start(sound);
    REQUIRE(bank.active() == 2);

    bank.silence();
    CHECK(bank.active() == 0);

    std::array<float, 4> out{};
    bank.mix(out);
    CHECK(out[0] == 0.0f);
}

TEST_CASE("An empty sound is ignored rather than occupying a slot", "[unit][app][audio]") {
    md::VoiceBank bank;
    bank.start(std::span<const float>{});
    CHECK(bank.active() == 0);
}
