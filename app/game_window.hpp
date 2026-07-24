// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "audio.hpp"
#include "highscores.hpp"
#include "md/sim.hpp"

#include <QElapsedTimer>
#include <QVulkanWindow>
#include <array>
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
    enum class State : std::uint8_t {
        Menu,
        Playing,
        GameOver,
        Highscores,
        Help,
        Options,
        EnterScore
    };

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

    // Options screen (a second centered list): AUDIO / MUSIC / FULLSCREEN + BACK.
    [[nodiscard]] static int options_count() noexcept;
    [[nodiscard]] std::string_view options_label(int index) const;

    [[nodiscard]] bool audio_on() const noexcept { return audio_on_; }

    [[nodiscard]] bool music_on() const noexcept { return music_on_; }

    [[nodiscard]] bool fullscreen() const noexcept { return fullscreen_; }

    // Menu/Options share a centered vertical-list layout (world units) — the
    // single source of truth for the renderer (drawing) and mouse hit-testing.
    [[nodiscard]] float menu_text_px() const noexcept;
    [[nodiscard]] float menu_item_top_y(int index) const noexcept;
    [[nodiscard]] int menu_hit(Vec2 world) const noexcept; // item under a point in the active list

    // Highscores + arcade initials entry (for the Highscores / EnterScore screens).
    [[nodiscard]] const HighscoreTable& highscores() const noexcept { return highscores_; }

    [[nodiscard]] std::array<char, 3> entry_initials() const noexcept { return entry_initials_; }

    [[nodiscard]] int entry_slot() const noexcept { return entry_slot_; }

    [[nodiscard]] int final_score() const noexcept { return final_score_; }

  protected:
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

  private:
    enum class MenuAction : std::uint8_t { Resume, NewGame, Help, Options, Highscores, Exit };

    void update_aim(float px, float py);
    void start_game();
    void end_game(); // termination -> initials entry (if a high score) or game over
    void handle_score_entry(int key); // arcade initials input
    void toggle_fullscreen();
    void toggle_audio();
    void toggle_music();
    void open_menu();
    void open_options();
    void activate(int index); // activate the item at index in the active list
    [[nodiscard]] MenuAction action_at(int index) const;
    [[nodiscard]] int active_count() const noexcept; // active list length (menu/options)
    [[nodiscard]] std::string_view active_label(int index) const; // active list label
    [[nodiscard]] BaseId nearest_base_with_ammo(Vec2 target) const;

    Sim sim_;
    AudioEngine audio_;
    HighscoreTable highscores_;
    QElapsedTimer clock_;
    double accumulator_ = 0.0;
    bool started_ = false;
    bool in_progress_ = false; // a game is running or paused-in-menu
    std::uint64_t seed_ = 1;
    Action pending_ = Action::noop();
    Vec2 aim_{};
    State state_ = State::Menu;
    int menu_index_ = 0;
    bool cursor_hidden_ = false;
    bool audio_on_ = true;
    bool music_on_ = true;    // looping FM-synth background music
    bool fullscreen_ = false; // windowed by default
    int final_score_ = 0;     // score captured at game over (for the entry screen)
    std::array<char, 3> entry_initials_{{'A', 'A', 'A'}};
    int entry_slot_ = 0;
};

} // namespace md
