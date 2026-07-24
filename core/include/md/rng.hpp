#pragma once

#include <cstdint>

namespace md {

/// Deterministic PCG32 generator (PCG-XSH-RR, 64-bit state / 32-bit output).
///
/// Per-instance state, seedable in the constructor — the *sole* source of
/// randomness in the simulation. Same seed ⇒ same stream, which underpins the
/// determinism contract (identical trajectories from identical seeds).
class Pcg32 {
  public:
    constexpr explicit Pcg32(std::uint64_t seed = default_seed,
                             std::uint64_t stream = default_stream) noexcept {
        inc_ = (stream << 1u) | 1u; // inc must be odd
        advance();
        state_ += seed;
        advance();
    }

    /// Next 32-bit output.
    constexpr std::uint32_t next_u32() noexcept {
        const std::uint64_t old = state_;
        state_ = old * multiplier + inc_;
        const auto xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const auto rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    /// Uniform float in [0, 1). Uses 24 random bits (exact in float32).
    float next_float() noexcept {
        return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
    }

    /// Uniform float in [lo, hi).
    float uniform(float lo, float hi) noexcept { return lo + (hi - lo) * next_float(); }

    /// Unbiased integer in [0, bound). Rejection-samples to remove modulo bias.
    std::uint32_t below(std::uint32_t bound) noexcept {
        const std::uint32_t threshold = (~bound + 1u) % bound; // == (2^32 - bound) % bound
        for (;;) {
            const std::uint32_t r = next_u32();
            if (r >= threshold) {
                return r % bound;
            }
        }
    }

  private:
    static constexpr std::uint64_t multiplier = 6364136223846793005ULL;
    static constexpr std::uint64_t default_seed = 0x853c49e6748fea9bULL;
    static constexpr std::uint64_t default_stream = 0xda3e39cb94b95bdbULL;

    constexpr void advance() noexcept { state_ = state_ * multiplier + inc_; }

    std::uint64_t state_ = 0;
    std::uint64_t inc_ = 0;
};

} // namespace md
