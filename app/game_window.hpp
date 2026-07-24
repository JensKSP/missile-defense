#pragma once

#include "md/sim.hpp"

#include <QVulkanWindow>

namespace md {

/// The top-level game window. Owns the simulation and creates the Vulkan
/// renderer that draws it. (Input + the timed game loop arrive in a later step.)
class GameWindow : public QVulkanWindow {
  public:
    GameWindow();

    QVulkanWindowRenderer* createRenderer() override;

    [[nodiscard]] const Sim& sim() const noexcept { return sim_; }

  private:
    Sim sim_;
};

} // namespace md
