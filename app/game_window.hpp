// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "audio.hpp"
#include "highscores.hpp"
#include "md/agent/heuristic.hpp"
#include "md/replay/recording.hpp"
#include "md/sim.hpp"

#include <QElapsedTimer>
#include <QVulkanWindow>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
        About,
        Options,
        EnterScore,
        Replays
    };

    GameWindow();

    QVulkanWindowRenderer* createRenderer() override;

    /// Advance the sim by real elapsed time (fixed `dt`); only while Playing.
    void advance();

    /// Start a game immediately, skipping the menu (used by the `--play` flag).
    void play_now() { start_game(); }

    /// Start a game with the scripted agent at the controls (the `--watch` flag).
    void watch_now() { start_ai_game(); }

    /// Play a recorded run (the `--replay` flag). False if it could not be read.
    bool watch_replay(const std::string& path);

    /// Is the scripted agent driving rather than the mouse?
    [[nodiscard]] bool ai_driving() const noexcept { return ai_driving_; }

    /// Is a recorded run being played back?
    [[nodiscard]] bool replaying() const noexcept { return replay_.has_value(); }

    /// The recording's label and how far through it we are — for the HUD.
    [[nodiscard]] std::string_view replay_label() const noexcept;
    [[nodiscard]] float replay_progress() const noexcept;

    // The REPLAYS browser: recordings found in the runs directory.
    [[nodiscard]] int replay_count() const noexcept {
        return static_cast<int>(replay_files_.size());
    }

    [[nodiscard]] std::string_view replay_name(int index) const;

    /// Did the agent drive any part of this game? Such a run is never eligible for
    /// the highscore table — those are the human's.
    [[nodiscard]] bool ai_assisted() const noexcept { return ai_assisted_; }

    /// Simulation ticks run per frame — 1 is real time, higher fast-forwards.
    [[nodiscard]] int speed() const noexcept { return speed_; }

    /// The simulation on screen. While replaying that is the player's own sim, so
    /// the renderer, audio and HUD all read one source whichever driver is active.
    [[nodiscard]] const Sim& sim() const noexcept {
        return replay_.has_value() ? replay_->sim() : sim_;
    }

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

    // The recording list is a chooser, not a notice board, so it gets the same
    // treatment: one layout both the renderer and the hit test read. `scroll` is
    // the first visible row and moves only under the keyboard — if hovering
    // changed it, the rows would slide out from under the pointer that selected
    // them, and the highlight would chase itself down the list.
    static constexpr int replay_rows_visible = 8;

    [[nodiscard]] int replay_scroll() const noexcept { return replay_scroll_; }

    [[nodiscard]] float replay_row_px() const noexcept;
    [[nodiscard]] float replay_row_top_y(int index) const noexcept;
    [[nodiscard]] int replay_hit(Vec2 world) const noexcept;
    void scroll_replays_into_view() noexcept;
    void play_selected_replay();

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
    enum class MenuAction : std::uint8_t {
        Resume,
        NewGame,
        WatchAi,
        Replays,
        Help,
        Options,
        Highscores,
        About,
        Exit
    };

    void update_aim(float px, float py);
    void start_game();
    void start_ai_game(); // same, but the scripted agent supplies the actions
    void end_game();      // termination -> initials entry (if a high score) or game over
    void handle_score_entry(int key); // arcade initials input
    void toggle_fullscreen();
    void toggle_audio();
    void toggle_music();
    void load_settings();       // restore persisted audio/music/fullscreen on startup
    void save_settings() const; // persist audio/music/fullscreen after a toggle
    void open_menu();
    void open_options();
    void open_replays();      // scan the runs directory and show what is there
    void scrub(int seconds);  // seek the active replay, relative
    void activate(int index); // activate the item at index in the active list
    [[nodiscard]] MenuAction action_at(int index) const;
    [[nodiscard]] int active_count() const noexcept; // active list length (menu/options)
    [[nodiscard]] std::string_view active_label(int index) const; // active list label
    [[nodiscard]] BaseId nearest_base_with_ammo(Vec2 target) const;

    Sim sim_;
    agent::Heuristic agent_{};              // the M4 baseline, used in watch mode
    std::optional<replay::Player> replay_;  // a recorded run being played back
    std::vector<std::string> replay_files_; // paths offered by the REPLAYS screen
    std::vector<std::string> replay_names_; // ...their display names, uppercased
    AudioEngine audio_;
    HighscoreTable highscores_;
    QElapsedTimer clock_;
    double accumulator_ = 0.0;
    bool started_ = false;
    bool in_progress_ = false; // a game is running or paused-in-menu
    std::uint64_t seed_ = 1;
    bool fire_pending_ = false; // a click arrived; fire on the next sim tick
    bool ai_driving_ = false;   // the scripted agent is at the controls
    bool ai_assisted_ = false;  // ... at any point this game (sticky; blocks highscores)
    int speed_ = 1;             // sim ticks per frame: 1 = real time, up to 8x
    Vec2 aim_{};
    State state_ = State::Menu;
    int menu_index_ = 0;
    int replay_scroll_ = 0; // first visible row of the recording list
    bool cursor_hidden_ = false;
    bool audio_on_ = true;
    bool music_on_ = true;    // looping FM-synth background music
    bool fullscreen_ = false; // windowed by default
    int final_score_ = 0;     // score captured at game over (for the entry screen)
    std::array<char, 3> entry_initials_{{'A', 'A', 'A'}};
    int entry_slot_ = 0;
};

} // namespace md
