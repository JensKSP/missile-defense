// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "audio.hpp"
#include "highscores.hpp"
#include "human_input.hpp"
#include "install.hpp"
#include "md/agent/eval.hpp"
#include "md/agent/handicap.hpp"
#include "md/agent/heuristic.hpp"
#include "md/agent/policy.hpp"
#include "md/replay/match.hpp"
#include "md/replay/recording.hpp"
#include "md/sim.hpp"
#include "trainer.hpp"

#include <QElapsedTimer>
#include <QVulkanWindow>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
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
        Replays,
        /// The WATCH AI submenu: SCRIPTED / PRETRAINED / BACK. A screen rather
        /// than a cycling toggle, because which agent you are about to watch is
        /// the whole question and a toggle answers it only after the fact.
        Watch,
        /// Why TRAIN AI could not just start something, and what would fix it.
        /// A screen rather than a silence: the entry is now always in the menu,
        /// because on Windows and macOS *not installed* is the ordinary state
        /// and hiding it meant most people never learned the trainer exists.
        TrainNotice,
        /// Two recordings of the same seed, side by side on one clock. Its own
        /// state and not a flavour of Replays: there is no single `sim()` to
        /// hand the renderer, and every transport key moves both sides at once.
        Match
    };

    GameWindow();

    QVulkanWindowRenderer* createRenderer() override;

    /// Advance the sim by real elapsed time (fixed `dt`); only while Playing.
    void advance();

    /// Start a game immediately, skipping the menu (used by the `--play` flag).
    void play_now() { start_game(); }

    /// Start a game with the scripted agent at the controls (the `--watch` flag).
    void watch_now() { start_ai_game(); }

    /// ... at a chosen skill (`--watch-scripted low|medium|high`).
    void watch_now(agent::Skill skill) { start_ai_game(skill); }

    /// Parse a skill name, or nothing when it is not one of the three.
    [[nodiscard]] static std::optional<agent::Skill> skill_named(std::string_view name);

    /// Start a game a learned policy plays (the `--watch-model <path>` flag).
    ///
    /// False when the file is not a policy this build can run, with the reason
    /// on stderr. A refusal rather than a silent fall back to the scripted
    /// agent: watching the wrong agent and not being told is worse than not
    /// watching at all, and it is exactly the confusion Step 4b exists to end.
    bool watch_model(const std::string& path);

    /// Play this exact seed instead of a fresh one — `--seed`.
    ///
    /// What makes *peeking* at a running contest possible: the evaluator and
    /// the game are both deterministic, so the same policy on the same seed is
    /// the same episode, tick for tick. Without it a peek would show a
    /// different game from the one being scored, which is worse than no peek.
    ///
    /// Pinned for the whole session rather than for one game, so `R` and a
    /// second START replay the episode being watched rather than wandering off
    /// the seed somebody asked for.
    void set_seed(std::uint64_t seed) noexcept {
        seed_ = seed;
        pinned_seed_ = true;
    }

    /// Play a recorded run (the `--replay` flag). False if it could not be read.
    bool watch_replay(const std::string& path);

    /// Quit after this many rendered frames; 0 (the default) means never.
    ///
    /// Without an upper bound the game can only be *driven* by a human closing
    /// the window, which makes every automated check of it a job that hangs
    /// rather than a test that fails (docs/TESTING.md). It is not only a test
    /// affordance: a fixed frame count is also how a renderer change is timed.
    void set_frame_budget(std::uint64_t frames) noexcept { frame_budget_ = frames; }

    /// Quit as soon as a game or a recording ends, instead of showing game over.
    ///
    /// This is what makes a replay assertion deterministic: a recording has a
    /// fixed length, so "play it to the end and say what happened" has exactly
    /// one answer, where a frame budget would stop somewhere arbitrary in it.
    void set_exit_when_done(bool on) noexcept { exit_when_done_ = on; }

    /// Frames rendered since start — what `--frames` counts and `--report` says.
    [[nodiscard]] std::uint64_t frames() const noexcept { return frames_; }

    /// Run with no sound at all, and without remembering that.
    ///
    /// An automated run must not come out of the speakers of whoever is at the
    /// machine — and must not leave their sound switched off afterwards either,
    /// which is why this suppresses persistence rather than just toggling the
    /// two flags the Options screen writes.
    void set_silent() noexcept;

    /// Is an agent driving rather than the mouse?
    [[nodiscard]] bool ai_driving() const noexcept { return ai_driving_; }

    /// Who is at the controls, for the HUD and for `--report`.
    ///
    /// **Asked for directly** (docs/ROADMAP.md, M8): watching two agents and
    /// being unable to tell which one is on screen makes the whole feature
    /// nearly useless. `SCRIPTED`, or the model's display name out of its
    /// `.mdp` — never a path, because `policy-best.pt` says nothing about which
    /// run produced it. Empty while a human is playing.
    [[nodiscard]] std::string_view driver_name() const noexcept { return driver_name_; }

    /// Does this build have a bundled learned policy to offer in the menu?
    ///
    /// False in a source checkout with no `models/pretrained.mdp` and in every
    /// package until one ships, which is the honest state today: WATCH AI then
    /// starts the scripted agent directly rather than offering a choice of one.
    [[nodiscard]] bool has_pretrained() const noexcept { return pretrained_.has_value(); }

    /// How many models this install can actually run, bundled and promoted.
    ///
    /// Public and static so `--report` can state it without a window: a model
    /// the trainer promoted and the game silently will not offer is exactly the
    /// failure a packaging test has to be able to see.
    [[nodiscard]] static int installed_model_count();

    /// How many recordings the browser would find, for the same reason.
    [[nodiscard]] static int discovered_recording_count();

    /// Is a recorded run being played back?
    [[nodiscard]] bool replaying() const noexcept { return replay_.has_value(); }

    /// Play two recordings of the same seed side by side (`--match`).
    ///
    /// False, with the reason on stderr, if the manifest or either recording
    /// could not be used — a match that silently falls back to one side would
    /// be worse than no match at all.
    bool watch_match(const std::string& manifest);

    /// Pair two recordings directly, with no manifest (`--match-left/-right`).
    bool watch_match(const std::string& left, const std::string& right);

    /// The match on screen, or nullptr. The renderer asks; nothing else should.
    [[nodiscard]] const replay::MatchPlayer* match() const noexcept {
        return match_.has_value() ? &*match_ : nullptr;
    }

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

    /// Where a bundled learned policy would be found, or empty if none is.
    /// Static so the packaging tests can ask without building a window.
    [[nodiscard]] static std::filesystem::path pretrained_path();

    /// Every `.mdp` the package ships, name-sorted — the WATCH AI ladder. Three
    /// of them: the same network at `learned-low`, `learned-medium` and
    /// `learned-high`, so a player can watch one policy learn rather than take
    /// the claim on trust.
    [[nodiscard]] static std::vector<std::filesystem::path> bundled_models();

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

    /// Does this install have a trainer to offer? The game-only
    /// package is the promise (docs/PACKAGING.md), so on one of those this is
    /// false and TRAIN AI is simply not in the menu — the entry is never shown
    /// disabled, because "installed but greyed out" is a different product
    /// claim than "not part of this product".
    [[nodiscard]] bool can_train() const noexcept { return trainer_.has_value(); }

    /// What TRAIN AI will do, decided once at startup like the lookup itself.
    ///
    /// Read by the renderer to pick the notice's wording, which is why it is
    /// public: the strings live with the rest of the UI text, and this is the
    /// one fact they vary on.
    [[nodiscard]] install::Offer train_offer() const noexcept { return offer_; }

    // Options screen (a second centered list): AUDIO / MUSIC / FULLSCREEN + BACK.
    [[nodiscard]] static int options_count() noexcept;
    [[nodiscard]] std::string_view options_label(int index) const;

    // The WATCH AI submenu (a third): SCRIPTED / the model's name / BACK.
    [[nodiscard]] int watch_count() const noexcept;
    [[nodiscard]] std::string_view watch_label(int index) const;

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

    /// Which list the browser screen is showing.
    ///
    /// One screen, two contents: recordings and installed models scroll, hover,
    /// select and page identically, and a second copy of that layout would
    /// drift from this one the moment either moved.
    enum class Browse : std::uint8_t { Replays, Models };

    [[nodiscard]] Browse browsing() const noexcept { return browse_; }

    [[nodiscard]] int replay_scroll() const noexcept { return replay_scroll_; }

    [[nodiscard]] float replay_row_px() const noexcept;
    [[nodiscard]] float replay_row_top_y(int index) const noexcept;
    [[nodiscard]] int replay_hit(Vec2 world) const noexcept;
    void scroll_replays_into_view() noexcept;
    void open_selected();

    // Highscores + arcade initials entry (for the Highscores / EnterScore screens).
    [[nodiscard]] const HighscoreTable& highscores() const noexcept { return highscores_; }

    [[nodiscard]] std::array<char, 3> entry_initials() const noexcept { return entry_initials_; }

    [[nodiscard]] int entry_slot() const noexcept { return entry_slot_; }

    [[nodiscard]] int final_score() const noexcept { return final_score_; }

  protected:
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

    /// Detaches from the Vulkan instance on `Close`, which is what lets this
    /// window come apart on Wayland instead of segfaulting, and destroys the
    /// surface that detaching orphans once the window is past `destroy()`. See
    /// the long note on the definition; the first half is the whole reason the
    /// game can run natively there, and the second is why it costs nothing.
    bool event(QEvent* event) override;

  private:
    enum class MenuAction : std::uint8_t {
        Resume,
        NewGame,
        WatchAi,
        WatchScripted,
        WatchPretrained,
        TrainAi,
        Replays,
        Help,
        Options,
        Highscores,
        About,
        Exit
    };

    /// Whether this run has reached whichever end it was given.
    [[nodiscard]] bool finished() const noexcept;

    void update_aim(float px, float py);
    void start_game();
    /// Same, but the scripted agent supplies the actions — at `skill`, which
    /// is the published baseline unless the menu or a flag says otherwise.
    void start_ai_game(agent::Skill skill = agent::Skill::high);
    /// ... and this one, but a learned policy does. `policy` empty means the
    /// bundled one; `--watch-model` passes a file it has just loaded.
    void start_model_game(std::optional<agent::Policy> policy = std::nullopt);
    void end_game(); // termination -> initials entry (if a high score) or game over
    void advance_death(float seconds); // age the final explosions, then end_game()
    void handle_score_entry(int key);  // arcade initials input
    void toggle_fullscreen();
    void toggle_audio();
    void toggle_music();
    /// LOW -> MEDIUM -> HIGH -> LOW, rebuilding the scripted agent as it goes.
    void cycle_ai_skill();
    void load_settings();       // restore persisted audio/music/fullscreen on startup
    void save_settings() const; // persist audio/music/fullscreen after a toggle
    void open_menu();
    void open_options();
    void open_replays();             // scan the runs directory and show what is there
    void open_models();              // scan for installed models and show what is there
    void open_watch();               // the WATCH AI submenu: which agent is playing
    void open_trainer();             // start the trainer, or say why it cannot
    static void open_instructions(); // hand HOW-TO-TRAIN.html to the desktop
    void start_trainer_install();    // spawn the terminal that pip installs it
    void scrub(int seconds);         // seek the active replay, relative
    void advance_match();            // drive both sides of a match on one clock
    void scrub_match(int seconds);   // seek the match's shared clock, relative
    void activate(int index);        // activate the item at index in the active list
    [[nodiscard]] MenuAction action_at(int index) const;
    [[nodiscard]] int active_count() const noexcept; // active list length (menu/options)
    [[nodiscard]] std::string_view active_label(int index) const; // active list label
    [[nodiscard]] BaseId nearest_base_with_ammo(Vec2 target) const;

    Sim sim_;
    /// How to start the trainer, resolved once at startup because it
    /// is a filesystem search and the menu asks on every frame. Empty on a
    /// game-only install, which is what removes the TRAIN AI entry.
    std::optional<trainer::Command> trainer_;
    /// The machine as `install::decide` saw it, and its answer. Both resolved
    /// once at startup for the same reason the lookup is: neither can change
    /// while the game is running, and probing interpreters costs process
    /// launches that must not happen while a menu is being drawn.
    install::Machine machine_;
    install::Offer offer_ = install::Offer::NeedsPackage;
    /// The bundled learned policy, loaded once at startup. Empty until one is
    /// shipped — see `pretrained_path`.
    std::optional<agent::Policy> pretrained_;
    /// Its display name, upper-cased once: the pixel font has no lower case,
    /// and `watch_label` hands back a view rather than a string.
    std::string pretrained_label_;
    /// The policy currently at the controls, and its driver. Held separately
    /// from `pretrained_` because `--watch-model` can name any file.
    std::optional<agent::Policy> watched_;
    std::optional<agent::PolicyDriver> watch_driver_;
    std::string driver_name_;
    agent::Heuristic agent_{}; // the scripted agent, at whatever skill was chosen
    //: The scripted agent as a `Driver`, so it can wear the same handicap a
    //: learned policy does. `agent_` stays for the takeover path, which hands a
    //: half-played game back to a human rather than to a contestant.
    std::optional<agent::ScriptedDriver> scripted_driver_;
    //: What is actually driving a watched game: whichever contestant, wearing
    //: `md::agent::canonical_handicap`. Without it the game would show HIGH
    //: scoring what an unhandicapped agent scores while the trainer reports the
    //: handicapped number for the same name — two ladders, one label.
    std::optional<agent::HandicappedDriver> handicapped_;
    std::optional<replay::Player> replay_; // a recorded run being played back
    // Two recordings on one clock. Separate from `replay_` and mutually
    // exclusive with it: a match has no single sim, so nothing that reads
    // `sim()` can be allowed to think it is watching one recording.
    std::optional<replay::MatchPlayer> match_;
    bool match_paused_ = false; // SPACE, and where a finished match parks
    /// W, and remembered between sessions. On by default: two agents on
    /// different waves are not answering the same problem, which is the one
    /// thing a split screen is for. See `replay::MatchPlayer::set_wave_sync`.
    bool match_wave_sync_ = true;
    std::vector<std::string> replay_files_; // paths offered by the browser screen
    std::vector<std::string> replay_names_; // ...their display names, uppercased
    Browse browse_ = Browse::Replays;       // which of the two that screen is showing
    int installed_models_ = 0;              // refreshed when the WATCH AI submenu opens
    AudioEngine audio_;
    HighscoreTable highscores_;
    QElapsedTimer clock_;
    double accumulator_ = 0.0;

    /// How long the last explosions keep burning after the game is lost, before
    /// the GAME OVER screen replaces them.
    ///
    /// The final warhead lands, the city goes, and `Sim` terminates in that same
    /// tick — so switching screens on termination cut the explosion off at
    /// radius zero and swallowed its sound. `Sim::step` is a no-op once
    /// terminated (deliberately: a finished episode must not be advanced by
    /// accident), so the blasts cannot simply be left to the simulation. They
    /// are copied out and aged here instead, which is presentation and belongs
    /// in the window rather than in `md::core`.
    static constexpr float death_seconds = 2.0F;

    //: Counts up during the death throes; negative means not dying.
    float dying_for_ = -1.0F;
    //: The interceptor blasts as they stood when the game was lost.
    std::vector<Blast> dying_blasts_;
    //: And the ground impacts — including the one that took the last city, which
    //: is the whole point: that is the explosion the player never got to see.
    std::vector<Explosion> dying_explosions_;

  public:
    /// The explosions to draw: the simulation's, or the dying copy once it has
    /// stopped advancing them.
    [[nodiscard]] std::span<const Blast> visible_blasts() const noexcept {
        return dying_for_ >= 0.0F ? std::span<const Blast>{dying_blasts_} : sim().blasts();
    }

    /// The ground impacts to draw, on the same terms.
    [[nodiscard]] std::span<const Explosion> visible_explosions() const noexcept {
        return dying_for_ >= 0.0F ? std::span<const Explosion>{dying_explosions_}
                                  : sim().explosions();
    }

  private:
    bool started_ = false;
    bool in_progress_ = false; // a game is running or paused-in-menu
    std::uint64_t frames_ = 0;
    std::uint64_t frame_budget_ = 0; // 0 = run until the window is closed
    bool exit_when_done_ = false;
    bool closing_ = false; // a close is already queued; do not queue a second
    bool silent_ = false;  // --silent: no sound, and no writing that preference
    std::uint64_t seed_ = 1;
    bool pinned_seed_ = false;  // --seed: every game this process starts is that one
    HumanFireLatch fire_latch_; // a click waits until Sim samples its next action
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
    /// Which scripted agent WATCH AI runs. Defaults to the published baseline,
    /// so what a player watches out of the box is the agent the README quotes.
    agent::Skill ai_skill_ = agent::Skill::high;
    int final_score_ = 0; // score captured at game over (for the entry screen)
    std::array<char, 3> entry_initials_{{'A', 'A', 'A'}};
    int entry_slot_ = 0;
};

} // namespace md
