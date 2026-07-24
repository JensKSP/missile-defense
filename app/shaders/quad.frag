#version 450

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vLocal;
layout(location = 2) in float vShape;

layout(location = 0) out vec4 outColor;

void main() {
    if (vShape > 0.5) {
        // Circle inscribed in the quad: vLocal is in [-0.5, 0.5], so |vLocal|*2
        // is 0 at the centre and 1 at the inscribed radius.
        if (length(vLocal) * 2.0 > 1.0) {
            discard;
        }
    }
    outColor = vec4(vColor, 1.0);
}
