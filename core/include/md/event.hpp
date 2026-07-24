#pragma once

#include "md/vec2.hpp"

#include <cstdint>
#include <type_traits>

namespace md {

/// Discrete things that happened during a `step()`. One event stream feeds the
/// app's sound effects (human), the AI's observation (parity — the model "hears"
/// what the human hears), and replays (deterministic from seed + actions).
enum class EventType : std::uint8_t {
    Fire,         // an interceptor was launched
    Detonate,     // an interceptor detonated into a blast
    ThreatKilled, // a threat was destroyed by a blast
    CityLost,     // a city was destroyed
    BaseLost,     // a battery was destroyed
    WaveCleared,  // a wave was completed
    BonusCity,    // a destroyed city was rebuilt
    GameOver,     // the last city fell
};

struct Event {
    EventType type = EventType::Fire;
    Vec2 pos{}; // where it happened (world units) — for spatial audio / observation
};

static_assert(std::is_trivially_copyable_v<Event>);

} // namespace md
