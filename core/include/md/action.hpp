#pragma once

#include "md/config.hpp"
#include "md/vec2.hpp"

#include <cstdint>
#include <type_traits>

namespace md {

/// The single control primitive, shared by human and AI. `Fire` launches an
/// interceptor from `base` toward `target`; `NoOp` does nothing this tick.
struct Action {
    enum class Kind : std::uint8_t { NoOp = 0, Fire = 1 };

    Kind kind = Kind::NoOp;
    BaseId base = BaseId::Alpha;
    Vec2 target{};

    [[nodiscard]] static constexpr Action noop() noexcept { return Action{}; }

    [[nodiscard]] static constexpr Action fire(BaseId b, Vec2 t) noexcept {
        return Action{.kind = Kind::Fire, .base = b, .target = t};
    }
};

static_assert(std::is_trivially_copyable_v<Action>);

} // namespace md
