#include "game_window.hpp"

#include "renderer.hpp"

namespace md {

GameWindow::GameWindow() {
    sim_.reset(0);
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

} // namespace md
