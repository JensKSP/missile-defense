#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

using md::Action;
using md::BaseId;
using md::Sim;
using md::Vec2;

namespace {

// A deterministic, state-independent scripted policy: every 12 ticks, fire from a
// cycling base at a cycling aim point. Exercises firing, blasts, and scoring.
Action scripted_action(std::uint64_t t) {
    if (t % 12 != 0) {
        return Action::noop();
    }
    const std::uint64_t phase = t / 12;
    const auto base = static_cast<BaseId>(phase % 3);
    const float x = 40.0f + (static_cast<float>(phase % 6) * 40.0f);
    return Action::fire(base, Vec2{x, 120.0f});
}

// FNV-1a fold over the observable state. Sensitive to float positions, so the
// running hash diverges immediately if the trajectory is not bit-identical.
std::uint64_t fold(std::uint64_t h, const Sim& sim) {
    const auto mix = [&h](std::uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    const auto mix_f = [&mix](float f) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        mix(bits);
    };
    mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(sim.score())));
    mix(sim.tick());
    mix(sim.wave());
    for (const auto& threat : sim.threats()) {
        mix_f(threat.pos.x);
        mix_f(threat.pos.y);
    }
    for (const auto& interceptor : sim.interceptors()) {
        mix_f(interceptor.pos.x);
        mix_f(interceptor.pos.y);
    }
    for (const auto& blast : sim.blasts()) {
        mix_f(blast.center.x);
        mix_f(blast.radius);
    }
    for (const auto& city : sim.cities()) {
        mix(city.alive ? 1ULL : 0ULL);
    }
    return h;
}

std::uint64_t run_checksum(std::uint64_t seed, std::uint64_t ticks) {
    Sim sim;
    sim.reset(seed);
    std::uint64_t h = 1469598103934665603ULL; // FNV-1a offset basis
    for (std::uint64_t t = 0; t < ticks; ++t) {
        sim.step(scripted_action(t));
        h = fold(h, sim);
    }
    return h;
}

} // namespace

TEST_CASE("A no-input episode runs to termination", "[e2e]") {
    Sim sim;
    sim.reset(2024);

    bool terminated = false;
    for (int i = 0; i < 100000 && !terminated; ++i) {
        terminated = sim.step(Action::noop()).terminated;
    }

    REQUIRE(terminated);
    for (const auto& city : sim.cities()) {
        REQUIRE_FALSE(city.alive);
    }
}

TEST_CASE("Same seed + actions produce an identical trajectory", "[e2e]") {
    REQUIRE(run_checksum(777, 1500) == run_checksum(777, 1500));
}

TEST_CASE("Trajectory checksum is stable across builds (Debug == Release)", "[e2e]") {
    // Golden value pins the whole trajectory: if Debug and Release ever diverge
    // (e.g. FP contraction), one of them fails this exact-match check.
    constexpr std::uint64_t golden = 0x8f60794f15979469ULL;
    REQUIRE(run_checksum(777, 1500) == golden);
}
