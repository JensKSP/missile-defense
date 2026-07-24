#pragma once

#include <QVulkanWindow>
#include <cstdint>

namespace md {

class GameWindow;

/// Draws the game with Vulkan. This sub-increment renders the static field
/// (ground, cities, bases) as instanced coloured quads under an orthographic
/// world->screen projection. Moving entities arrive next.
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
    VkBuffer instance_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory instance_mem_ = VK_NULL_HANDLE;
    std::uint32_t instance_count_ = 0;
};

} // namespace md
