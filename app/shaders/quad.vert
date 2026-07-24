#version 450

// Per-vertex: a unit quad corner in [-0.5, 0.5].
layout(location = 0) in vec2 inCorner;
// Per-instance: an axis-aligned box in world units, a colour, and a shape flag
// (0 = rectangle, 1 = circle inscribed in the box).
layout(location = 1) in vec2 inCenter;
layout(location = 2) in vec2 inHalfSize;
layout(location = 3) in vec3 inColor;
layout(location = 4) in float inShape;

layout(push_constant) uniform Push {
    vec2 a;
    vec2 b;
} pc;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vLocal;
layout(location = 2) out float vShape;

void main() {
    vec2 worldPos = inCenter + (inCorner * inHalfSize * 2.0);
    gl_Position = vec4((worldPos * pc.a) + pc.b, 0.0, 1.0);
    vColor = inColor;
    vLocal = inCorner; // [-0.5, 0.5]
    vShape = inShape;
}
