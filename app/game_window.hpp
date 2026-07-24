#pragma once

#include "md/sim.hpp"

#include <QElapsedTimer>
#include <QVulkanWindow>
#include <cstdint>

class QKeyEvent;
class QMouseEvent;

namespace md {

/// The top-level game window. Owns the simulation and a simple game-state
/// machine (menu / playing / paused / game-over / highscores), drives the sim on
/// a fixed timestep while playing, and turns Qt input into the shared `Action`.
class GameWindow : public QVulkanWindow {
  public:
    enum class State { Menu, Playing, Paused, GameOver, Highscores };

    GameWindow();

    QVulkanWindowRenderer* createRenderer() override;

    /// Advance the sim by real elapsed time (fixed `dt`); only while Playing.
    void advance();

    [[nodiscard]] const Sim& sim() const noexcept { return sim_; }

    [[nodiscard]] Vec2 aim() const noexcept { return aim_; }

    [[nodiscard]] State state() const noexcept { return state_; }

    [[nodiscard]] int menu_index() const noexcept { return menu_index_; }

  protected:
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

  private:
    void update_aim(float px, float py);
    void start_game();
    [[nodiscard]] BaseId nearest_base_with_ammo(Vec2 target) const;

    Sim sim_;
    QElapsedTimer clock_;
    double accumulator_ = 0.0;
    bool started_ = false;
    std::uint64_t seed_ = 1;
    Action pending_ = Action::noop();
    Vec2 aim_{};
    State state_ = State::Menu;
    int menu_index_ = 0;
};

} // namespace md
