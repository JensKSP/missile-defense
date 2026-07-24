#pragma once

#include "md/sim.hpp"

#include <QElapsedTimer>
#include <QVulkanWindow>
#include <cstdint>
#include <string_view>

class QKeyEvent;
class QMouseEvent;

namespace md {

/// The top-level game window. Owns the simulation and a game-state machine
/// (menu / playing / game-over / highscores / help), drives the sim on a fixed
/// timestep while playing, and turns Qt input into the shared `Action`.
///
/// Escape while playing pauses the game and shows the menu (the game is frozen
/// and preserved); from there RESUME continues it, NEW GAME abandons it.
class GameWindow : public QVulkanWindow {
  public:
    enum class State { Menu, Playing, GameOver, Highscores, Help };

    GameWindow();

    QVulkanWindowRenderer* createRenderer() override;

    /// Advance the sim by real elapsed time (fixed `dt`); only while Playing.
    void advance();

    /// Start a game immediately, skipping the menu (used by the `--play` flag).
    void play_now() { start_game(); }

    [[nodiscard]] const Sim& sim() const noexcept { return sim_; }

    [[nodiscard]] Vec2 aim() const noexcept { return aim_; }

    [[nodiscard]] State state() const noexcept { return state_; }

    [[nodiscard]] bool in_progress() const noexcept { return in_progress_; }

    [[nodiscard]] int menu_index() const noexcept { return menu_index_; }

    [[nodiscard]] int menu_count() const noexcept;
    [[nodiscard]] std::string_view menu_label(int index) const;

  protected:
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

  private:
    enum class MenuAction { Resume, NewGame, Help, Highscores, Exit };

    void update_aim(float px, float py);
    void start_game();
    void open_menu();
    void select_menu();
    [[nodiscard]] MenuAction action_at(int index) const;
    [[nodiscard]] BaseId nearest_base_with_ammo(Vec2 target) const;

    Sim sim_;
    QElapsedTimer clock_;
    double accumulator_ = 0.0;
    bool started_ = false;
    bool in_progress_ = false; // a game is running or paused-in-menu
    std::uint64_t seed_ = 1;
    Action pending_ = Action::noop();
    Vec2 aim_{};
    State state_ = State::Menu;
    int menu_index_ = 0;
};

} // namespace md
