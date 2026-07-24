#pragma once

#include <cmath>

namespace md {

/// A 2D vector of floats. The core geometric type for positions, velocities,
/// and directions in the simulation. Arithmetic is constexpr; only the
/// sqrt-based helpers (length, normalize) are runtime.
struct Vec2 {
    float x{};
    float y{};

    constexpr Vec2() = default;

    constexpr Vec2(float x_, float y_) noexcept : x{x_}, y{y_} {}

    constexpr Vec2& operator+=(Vec2 o) noexcept {
        x += o.x;
        y += o.y;
        return *this;
    }

    constexpr Vec2& operator-=(Vec2 o) noexcept {
        x -= o.x;
        y -= o.y;
        return *this;
    }

    constexpr Vec2& operator*=(float s) noexcept {
        x *= s;
        y *= s;
        return *this;
    }

    /// Squared magnitude — constexpr, avoids the sqrt when only comparing.
    [[nodiscard]] constexpr float length_sq() const noexcept { return x * x + y * y; }

    /// Euclidean magnitude.
    [[nodiscard]] float length() const noexcept { return std::sqrt(length_sq()); }

    /// Unit vector in the same direction; the zero vector maps to zero.
    [[nodiscard]] Vec2 normalized() const noexcept {
        const float len = length();
        return len > 0.0f ? Vec2{x / len, y / len} : Vec2{};
    }

    friend constexpr bool operator==(Vec2, Vec2) = default;
};

constexpr Vec2 operator+(Vec2 a, Vec2 b) noexcept {
    return {a.x + b.x, a.y + b.y};
}

constexpr Vec2 operator-(Vec2 a, Vec2 b) noexcept {
    return {a.x - b.x, a.y - b.y};
}

constexpr Vec2 operator-(Vec2 a) noexcept {
    return {-a.x, -a.y};
}

constexpr Vec2 operator*(Vec2 a, float s) noexcept {
    return {a.x * s, a.y * s};
}

constexpr Vec2 operator*(float s, Vec2 a) noexcept {
    return {a.x * s, a.y * s};
}

constexpr float dot(Vec2 a, Vec2 b) noexcept {
    return a.x * b.x + a.y * b.y;
}

constexpr float distance_sq(Vec2 a, Vec2 b) noexcept {
    return (a - b).length_sq();
}

inline float distance(Vec2 a, Vec2 b) noexcept {
    return (a - b).length();
}

} // namespace md
