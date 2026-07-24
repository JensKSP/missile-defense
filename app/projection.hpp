// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/vec2.hpp"

namespace md {

/// Orthographic world<->screen mapping: y-flip plus an aspect-preserving
/// letterbox. `clip.xy = worldPos * (ax,ay) + (bx,by)`. Shared by the renderer
/// (push constants) and the window (mouse -> world), so both agree exactly.
struct Projection {
    float ax = 1.0f;
    float ay = 1.0f;
    float bx = 0.0f;
    float by = 0.0f;

    [[nodiscard]] static Projection make(float world_w, float world_h, float view_w,
                                         float view_h) noexcept {
        const float world_aspect = world_w / world_h;
        const float view_aspect = view_w / view_h;
        float fx = 1.0f;
        float fy = 1.0f;
        if (view_aspect >= world_aspect) {
            fx = world_aspect / view_aspect;
        } else {
            fy = view_aspect / world_aspect;
        }
        return Projection{(2.0f * fx) / world_w, (-2.0f * fy) / world_h, -fx, fy};
    }

    /// Screen pixel (top-left origin) -> world coordinates.
    [[nodiscard]] Vec2 screen_to_world(float px, float py, float view_w,
                                       float view_h) const noexcept {
        const float ndc_x = ((px / view_w) * 2.0f) - 1.0f;
        const float ndc_y = ((py / view_h) * 2.0f) - 1.0f;
        return Vec2{(ndc_x - bx) / ax, (ndc_y - by) / ay};
    }
};

} // namespace md
