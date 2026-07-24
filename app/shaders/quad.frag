// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#version 450

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vLocal;
layout(location = 2) in float vShape;

layout(location = 0) out vec4 outColor;

void main() {
    float d = length(vLocal) * 2.0; // 0 at centre, 1 at the inscribed radius
    if (vShape > 1.5) {
        // Radial glow: opaque core fading to transparent at the edge.
        float a = clamp(1.0 - d, 0.0, 1.0);
        outColor = vec4(vColor.rgb, vColor.a * a * a);
    } else if (vShape > 0.5) {
        // Solid circle.
        if (d > 1.0) {
            discard;
        }
        outColor = vColor;
    } else {
        outColor = vColor;
    }
}
