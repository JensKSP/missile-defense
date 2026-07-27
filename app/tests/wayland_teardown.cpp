// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// A bare `QVulkanWindow` that renders nothing and then closes, run twice: once
// as Qt would have it, and once the way `GameWindow::event` does it. Between
// them they hold the Wayland workaround to two claims that can each be wrong.
//
//   `--plain`  Qt's own teardown. Expected to die. This is the evidence that
//              the workaround has a cause: the crash needs no swapchain of ours,
//              no renderer of ours, no line of this project at all. When it
//              stops dying, Qt has been fixed (QTBUG-123214) and
//              `GameWindow::event` can go back to doing nothing.
//
//   `--detach` the same program, releasing the Vulkan instance on `Close`.
//              Expected to survive. This is the evidence that the workaround
//              works for a reason rather than by luck — and it is the smallest
//              possible statement of it, with the game held out of the picture.
//
// Measured across 24 runs per mode, on the NVIDIA driver and on lavapipe: plain
// died 24/24 on both, detach survived 24/24 on both. Two implementations sharing
// no code do not agree by coincidence.
//
// Two things must be said about how this is built, because both can turn a
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

#include <QEvent>
#include <QGuiApplication>
#include <QVersionNumber>
#include <QVulkanInstance>
#include <QVulkanWindow>
#include <cstdio>
#include <cstring>

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
            // `close()` rather than `quit()`, because `Close` is the event the
            // workaround hangs off and the game's own EXIT goes through it too.
            window_->close();
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
    explicit Window(bool detach) : detach_(detach) {}

    QVulkanWindowRenderer* createRenderer() override { return new Renderer(this); }

    bool event(QEvent* event) override {
        // The one line under test. `GameWindow::event` does exactly this and
        // nothing else; keeping the witness's copy this small is what lets a
        // result here be read as a statement about that line.
        if (detach_ && event->type() == QEvent::Close) {
            setVulkanInstance(nullptr);
        }
        return QVulkanWindow::event(event);
    }

  private:
    bool detach_;
};

} // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);

    bool detach = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--detach") == 0) {
            detach = true;
        }
    }

    QVulkanInstance instance;
    instance.setApiVersion(QVersionNumber(1, 0)); // as app/main.cpp does; see the note there
    if (!instance.create()) {
        std::fprintf(stderr, "no Vulkan instance: %d\n", instance.errorCode());
        return 2;
    }

    Window window(detach);
    window.setVulkanInstance(&instance);
    window.resize(640, 480);
    window.show();
    const int code = QGuiApplication::exec();

    // Reached only if the window came apart without faulting. Without `--detach`
    // on Wayland it is not reached; with it, and on xcb either way, it is. That
    // difference is the whole test.
    std::puts(R"({"survived_teardown": true})");
    std::fflush(stdout);
    return code;
}
