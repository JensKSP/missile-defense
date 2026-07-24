#version 450

// Per-vertex: a unit quad corner in [-0.5, 0.5].
layout(location = 0) in vec2 inCorner;
// Per-instance: an axis-aligned box in world units, plus a colour.
layout(location = 1) in vec2 inCenter;
layout(location = 2) in vec2 inHalfSize;
layout(location = 3) in vec3 inColor;

// World -> Vulkan clip space: clip.xy = worldPos * a + b (includes the y-flip
// and aspect-preserving letterbox; computed on the CPU each frame).
layout(push_constant) uniform Push {
    vec2 a;
    vec2 b;
} pc;

layout(location = 0) out vec3 vColor;

void main() {
    vec2 worldPos = inCenter + (inCorner * inHalfSize * 2.0);
    gl_Position = vec4((worldPos * pc.a) + pc.b, 0.0, 1.0);
    vColor = inColor;
}
