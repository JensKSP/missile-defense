// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <QVulkanWindow>
#include <chrono>
#include <cstdint>
#include <vector>

namespace md {

class GameWindow;

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
    void createPipeline();
    void createBuffers();
    void build_stars();

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
    std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
};

} // namespace md
