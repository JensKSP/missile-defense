#pragma once

#include "md/sim.hpp"

#include <QElapsedTimer>
#include <QVulkanWindow>
#include <cstdint>

class QKeyEvent;
class QMouseEvent;

namespace md {

/// The top-level game window. Owns the simulation, drives it on a fixed
/// timestep from real elapsed time, and turns Qt mouse/keyboard input into the
/// shared `Action` primitive (the same one tests and the AI feed to the sim).
class GameWindow : public QVulkanWindow {
  public:
    GameWindow();

    QVulkanWindowRenderer* createRenderer() override;

    /// Advance the sim by the real time elapsed since the last call, in fixed
    /// `dt` steps. Called once per rendered frame.
    void advance();

    [[nodiscard]] const Sim& sim() const noexcept { return sim_; }

    [[nodiscard]] Vec2 aim() const noexcept { return aim_; }

  protected:
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

  private:
    void update_aim(float px, float py);
    [[nodiscard]] BaseId nearest_base_with_ammo(Vec2 target) const;

    Sim sim_;
    QElapsedTimer clock_;
    double accumulator_ = 0.0;
    bool started_ = false;
    std::uint64_t seed_ = 1;
    Action pending_ = Action::noop();
    Vec2 aim_{};
};

} // namespace md
