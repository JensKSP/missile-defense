// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// A bare `QVulkanWindow` that renders nothing and then closes, used to answer
// one question: when this game segfaults on exit under Wayland, is it us?
//
// `app/main.cpp` asks for the xcb platform on Wayland sessions. That is a
// workaround, and workarounds outlive their cause unless something fails the
// day the cause goes away. This program is that something.
//
// What it demonstrates, run under `QT_QPA_PLATFORM=wayland`: the crash needs no
// swapchain of ours, no renderer of ours, no line of this project at all. It
// happens inside `QWindowPrivate::destroy()`, which frees the `wl_surface` in
// `setVisible(false)` and only afterwards sends the event on which
// `QVulkanWindow` releases the swapchain built on it. The application never
// gets between the two; see the long note in `app/main.cpp`.
//
// Two things must be said about how it is built, because both can turn a
// crashing program into a passing one and neither is visible from the output:
//
//   * ASan's quarantine keeps freed blocks mapped, so an instrumented build
//     reads the stale memory without faulting. It reports its own instrumentation
//     so the test can decline to conclude anything from a sanitised run rather
//     than reading silence as a fix.
//   * The validation layer is not requested. The layer sees this ordering as
//     `VUID-vkDestroySurfaceKHR-surface-01266`, which is worth knowing but is
//     not what is being measured here — the exit status is.
//
// Never installed. `python/tests/e2e/test_wayland_teardown.py` runs it.

#include <QGuiApplication>
#include <QVersionNumber>
#include <QVulkanInstance>
#include <QVulkanWindow>
#include <cstdio>

namespace {

//: Enough for Qt to have a swapchain in steady state; the interesting part is
//: teardown, not throughput.
constexpr int kFrames = 60;

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif
#elif defined(__SANITIZE_ADDRESS__)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif

class Renderer : public QVulkanWindowRenderer {
  public:
    explicit Renderer(QVulkanWindow* window) : window_(window) {}

    void startNextFrame() override {
        if (++frames_ >= kFrames) {
            // Printed *before* teardown, because after it there may be no
            // process left to print anything — which is the result.
            std::printf(R"({"rendered": true, "sanitized": %s})"
                        "\n",
                        kSanitized ? "true" : "false");
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
    if (!instance.create()) {
        std::fprintf(stderr, "no Vulkan instance: %d\n", instance.errorCode());
        return 2;
    }

    Window window;
    window.setVulkanInstance(&instance);
    window.resize(640, 480);
    window.show();
    const int code = QGuiApplication::exec();

    // Reached only if the window came apart without faulting. On Wayland it is
    // not reached; on xcb it is. That difference is the whole test.
    std::puts(R"({"survived_teardown": true})");
    std::fflush(stdout);
    return code;
}
