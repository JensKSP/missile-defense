# app/ — Human client + replay viewer (Qt 6 + Vulkan)

The graphical front-end. Reads `md::core` state and renders it; feeds human
input back into the same `Action` primitive the AI uses.

- Windowing/UI/input via **Qt 6** (`QVulkanInstance` / `QVulkanWindow`).
- Rendering via **Vulkan** (instanced quads for entities, a line pipeline for
  trajectories/blast radii).
- Shaders compiled GLSL → SPIR-V with `glslangValidator` as a CMake build step.

Deferred until the core sim is frozen. Reused later for the Step-4 replay viewer
and Step-5 live AI spectator mode (no throwaway renderer).
