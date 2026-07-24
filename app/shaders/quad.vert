#version 450

// Per-vertex: a unit quad corner in [-0.5, 0.5].
layout(location = 0) in vec2 inCorner;
// Per-instance: an oriented box in world units, an RGBA colour, and a shape flag
// (0 = rectangle, 1 = solid circle, 2 = radial-glow circle).
layout(location = 1) in vec2 inCenter;
layout(location = 2) in vec2 inHalfSize;
layout(location = 3) in float inAngle;
layout(location = 4) in vec4 inColor;
layout(location = 5) in float inShape;

layout(push_constant) uniform Push {
    vec2 a;
    vec2 b;
} pc;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vLocal;
layout(location = 2) out float vShape;

void main() {
    vec2 local = inCorner * inHalfSize * 2.0;
    float c = cos(inAngle);
    float s = sin(inAngle);
    vec2 rotated = vec2((c * local.x) - (s * local.y), (s * local.x) + (c * local.y));
    vec2 worldPos = inCenter + rotated;
    gl_Position = vec4((worldPos * pc.a) + pc.b, 0.0, 1.0);
    vColor = inColor;
    vLocal = inCorner; // [-0.5, 0.5]
    vShape = inShape;
}
