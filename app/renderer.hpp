#pragma once

#include <QVulkanWindow>
#include <cstdint>
#include <vector>

namespace md {

class GameWindow;

/// Draws the game with Vulkan: the field plus the live entities (threats,
/// interceptors, blasts) and the aim crosshair, as instanced quads/circles under
/// an orthographic world->screen projection. Instance data is rebuilt each frame
/// from the Sim into per-frame-in-flight buffers.
class Renderer : public QVulkanWindowRenderer {
  public:
    explicit Renderer(GameWindow* window) noexcept;

    void initResources() override;
    void releaseResources() override;
    void startNextFrame() override;

  private:
    void createPipeline();
    void createBuffers();

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
};

} // namespace md
