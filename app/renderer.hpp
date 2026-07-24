#pragma once

#include <QVulkanWindow>

namespace md {

/// Draws the game each frame. For now it just clears the screen; entity drawing
/// (instanced quads + a line/circle pipeline) arrives in later sub-increments.
class Renderer : public QVulkanWindowRenderer {
  public:
    explicit Renderer(QVulkanWindow* window) noexcept;

    void startNextFrame() override;

  private:
    QVulkanWindow* window_;
};

} // namespace md
