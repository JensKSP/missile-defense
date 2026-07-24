#include "renderer.hpp"

#include <QVulkanDeviceFunctions>
#include <QVulkanInstance>
#include <array>
#include <cstdint>

namespace md {

Renderer::Renderer(QVulkanWindow* window) noexcept : window_{window} {}

void Renderer::startNextFrame() {
    const QSize size = window_->swapChainImageSize();

    // Night-sky clear. The default render pass has a colour + depth/stencil
    // attachment, plus a colour resolve attachment when MSAA is enabled.
    std::array<VkClearValue, 3> clear{};
    clear[0].color = {{0.02f, 0.02f, 0.06f, 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    clear[2].color = clear[0].color;

    const bool msaa = window_->sampleCountFlagBits() > VK_SAMPLE_COUNT_1_BIT;

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = window_->defaultRenderPass();
    begin.framebuffer = window_->currentFramebuffer();
    begin.renderArea.extent.width = static_cast<std::uint32_t>(size.width());
    begin.renderArea.extent.height = static_cast<std::uint32_t>(size.height());
    begin.clearValueCount = msaa ? 3u : 2u;
    begin.pClearValues = clear.data();

    const VkCommandBuffer cb = window_->currentCommandBuffer();
    QVulkanDeviceFunctions* dev = window_->vulkanInstance()->deviceFunctions(window_->device());
    dev->vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    // Nothing to draw yet — this sub-increment only proves the clear.
    dev->vkCmdEndRenderPass(cb);

    window_->frameReady();
    window_->requestUpdate(); // keep animating
}

} // namespace md
