// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "projection.hpp"
#include "terrain.hpp"

#include <QVulkanWindow>
#include <chrono>
#include <cstdint>
#include <vector>

namespace md {

class GameWindow;

namespace replay {
class MatchPlayer;
} // namespace replay

/// Per-instance vertex data: an oriented box (world units), an RGBA colour, and
/// a shape flag (0 = rectangle, 1 = solid circle, 2 = radial-glow circle).
///
/// In the header only because the split screen needs it: two viewports are two
/// draws over slices of one buffer, so building and submitting can no longer be
/// the same function.
struct InstanceData {
    float cx, cy;
    float hx, hy;
    float angle;
    float r, g, b, a;
    float shape;
};

/// One background star: a fixed sky position with an independent twinkle.
struct Star {
    float x, y;  // world position
    float base;  // base brightness / alpha
    float phase; // twinkle phase offset
    float speed; // twinkle speed (rad/s)
    float size;  // world-unit radius
};

/// Draws the game with Vulkan: the field plus the live entities (threats,
/// interceptors, blasts) and the aim crosshair, as instanced quads/circles under
/// an orthographic world->screen projection. Instance data is rebuilt each frame
/// from the Sim into per-frame-in-flight buffers.
class Renderer : public QVulkanWindowRenderer {
  public:
    explicit Renderer(GameWindow* window);

    void initResources() override;
    void releaseResources() override;
    void startNextFrame() override;

  private:
    /// One draw: a slice of the frame's instances, under its own projection,
    /// into its own rectangle of the swapchain image. A normal frame has one; a
    /// match has three (left world, right world, and the overlay that spans
    /// both and owns the divider and the shared transport).
    struct Pass {
        Projection proj;
        VkViewport viewport{};
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    void createPipeline();
    void createBuffers();
    void build_stars();
    void build_landscape();

    /// Upload the frame's instances once and record one draw per pass.
    void submit(std::vector<InstanceData>& inst, const std::vector<Pass>& passes);

    /// The split-screen match view: two synchronized recordings, side by side.
    void draw_match(const replay::MatchPlayer& match, QSize size);

    GameWindow* window_;
    QVulkanDeviceFunctions* dev_ = nullptr;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkBuffer quad_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory quad_mem_ = VK_NULL_HANDLE;

    // One instance buffer per frame in flight (persistently mapped).
    std::vector<VkBuffer> instance_bufs_;
    std::vector<VkDeviceMemory> instance_mems_;
    std::vector<void*> instance_mapped_;

    // Animated background starfield.
    std::vector<Star> stars_;

    // The landscape: the heightfield the cities stand on, and the instances that
    // draw it. Both are fixed for the life of the window — the ground does not
    // move, so it is built once and copied into each frame rather than rebuilt.
    Terrain terrain_;
    std::vector<InstanceData> ground_;
    std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
};

} // namespace md
