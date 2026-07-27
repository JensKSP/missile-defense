// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// A bare `QVulkanWindow` with no project code in it, used to answer one
// question: when the validation layer complains, is it complaining about us?
//
// `Renderer::submit` carries a per-frame `vkQueueWaitIdle` that exists only to
// stop Qt reusing a swapchain acquire semaphore that still has a wait pending
// (`VUID-vkAcquireNextImageKHR-semaphore-01779`). That workaround costs a frame
// of CPU/GPU overlap, so it should be deleted the day Qt no longer needs it —
// and kept, with evidence, for as long as it does.
//
// This program is that evidence. It creates a `QVulkanWindow`, renders nothing
// at all, and reports how many times the layer raised 01779. Two facts follow
// from a run of it, and `python/tests/e2e/test_vulkan_validation.py` asserts
// both:
//
//   * it reproduces 01779 with zero lines of this project in the process, so
//     the defect is upstream and not in `app/renderer.cpp`;
//   * if it ever stops reproducing it, Qt has been fixed and the workaround in
//     `Renderer::submit` should go.
//
// Built only when MD_VULKAN_VALIDATION is on, because without the layer there is
// nothing to observe. Never installed.

#include <QGuiApplication>
#include <QVersionNumber>
#include <QVulkanInstance>
#include <QVulkanWindow>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string_view>

namespace {

//: Long enough for Qt to cycle its two frame-resource sets many times over;
//: short enough that the whole check is a couple of seconds.
constexpr int kFrames = 200;

//: Handles named by a 01779 message. A set, because the interesting number is
//: how many *distinct* semaphores Qt reuses unsafely (two, matching its
//: hardcoded frameLag) rather than how many messages the layer chose to print —
//: the layer deduplicates after ten by default.
std::set<std::uint64_t> g_semaphores;
int g_reports = 0;

//: Whether a debug messenger was actually created. Without one there is nothing
//: listening, and "no violations" would be indistinguishable from "not asked".
bool g_messenger_installed = false;

VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                          VkDebugUtilsMessageTypeFlagsEXT,
                                          const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (data->pMessageIdName == nullptr ||
        std::string_view(data->pMessageIdName).find("01779") == std::string_view::npos) {
        return VK_FALSE;
    }
    ++g_reports;
    for (std::uint32_t i = 0; i < data->objectCount; ++i) {
        if (data->pObjects[i].objectType == VK_OBJECT_TYPE_SEMAPHORE) {
            g_semaphores.insert(data->pObjects[i].objectHandle);
        }
    }
    return VK_FALSE;
}

class Renderer : public QVulkanWindowRenderer {
  public:
    explicit Renderer(QVulkanWindow* window) : window_(window) {}

    void startNextFrame() override {
        if (++frames_ >= kFrames) {
            // One machine-readable line, mirroring the game's own `--report`.
            //
            // `messenger_installed` is not decoration. A run reporting zero
            // violations because it never had a messenger to hear them looks
            // exactly like a run reporting zero because there were none, and
            // only one of those means Qt is fixed.
            std::printf(R"({"vuid_01779_reports": %d, "distinct_semaphores": %zu, )"
                        R"("concurrent_frames": %d, "swapchain_images": %d, )"
                        R"("messenger_installed": %s})"
                        "\n",
                        g_reports, g_semaphores.size(), window_->concurrentFrameCount(),
                        window_->swapChainImageCount(), g_messenger_installed ? "true" : "false");
            std::fflush(stdout);
            QGuiApplication::quit();
            return;
        }
        window_->frameReady();
        window_->requestUpdate();
    }

  private:
    QVulkanWindow* window_;
    int frames_ = 0;
};

class Window : public QVulkanWindow {
  public:
    QVulkanWindowRenderer* createRenderer() override { return new Renderer(this); }
};

} // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);

    QVulkanInstance instance;
    instance.setApiVersion(QVersionNumber(1, 0)); // as app/main.cpp does; see the note there
    instance.setLayers({"VK_LAYER_KHRONOS_validation"});
    instance.setExtensions({"VK_EXT_debug_utils"});
    if (!instance.create()) {
        std::fprintf(stderr, "no Vulkan instance with the validation layer: %d\n",
                     instance.errorCode());
        return 2;
    }

    // Our own messenger rather than Qt's logging: the object handles are the
    // whole point, and Qt's default output prints only the message text.
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    auto* create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        instance.getInstanceProcAddr("vkCreateDebugUtilsMessengerEXT"));
    if (create != nullptr) {
        VkDebugUtilsMessengerCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        info.pfnUserCallback = on_message;
        g_messenger_installed =
            create(instance.vkInstance(), &info, nullptr, &messenger) == VK_SUCCESS;
    }

    Window window;
    window.setVulkanInstance(&instance);
    window.resize(1280, 720);
    window.show();
    return QGuiApplication::exec();
}
