// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "game_window.hpp"

#include "projection.hpp"
#include "renderer.hpp"

#include <QCursor>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QSettings>
#include <QStandardPaths>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace md {

GameWindow::GameWindow() {
    sim_.reset(seed_);
    aim_ = Vec2{sim_.config().world_width * 0.5f, sim_.config().world_height * 0.5f};
    highscores_.load();
    load_settings(); // restore audio/music/fullscreen from the previous session
}

// Persisted via QSettings — QGuiApplication's organization/application name (set
// in main) route this to the platform store (registry on Windows, INI elsewhere).
void GameWindow::load_settings() {
    const QSettings settings;
    audio_on_ = settings.value("audio/sfx", audio_on_).toBool();
    music_on_ = settings.value("audio/music", music_on_).toBool();
    fullscreen_ = settings.value("video/fullscreen", fullscreen_).toBool();
    audio_.set_enabled(audio_on_);
    audio_.set_music_enabled(music_on_);
    // Fullscreen is applied by main() at startup (before the window is shown).
}

void GameWindow::save_settings() const {
    QSettings settings;
    settings.setValue("audio/sfx", audio_on_);
    settings.setValue("audio/music", music_on_);
    settings.setValue("video/fullscreen", fullscreen_);
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

int GameWindow::menu_count() const noexcept {
    return in_progress_ ? 6 : 8; // WATCH AI, REPLAYS and ABOUT are main-menu only
}

GameWindow::MenuAction GameWindow::action_at(int index) const {
    if (in_progress_) {
        const std::array<MenuAction, 6> acts{MenuAction::Resume,     MenuAction::NewGame,
                                             MenuAction::Help,       MenuAction::Options,
                                             MenuAction::Highscores, MenuAction::Exit};
        return acts[static_cast<std::size_t>(index)];
    }
    const std::array<MenuAction, 8> acts{
        MenuAction::NewGame, MenuAction::WatchAi,    MenuAction::Replays, MenuAction::Help,
        MenuAction::Options, MenuAction::Highscores, MenuAction::About,   MenuAction::Exit};
    return acts[static_cast<std::size_t>(index)];
}

std::string_view GameWindow::menu_label(int index) const {
    switch (action_at(index)) {
    case MenuAction::Resume:
        return "RESUME";
    case MenuAction::NewGame:
        return in_progress_ ? "NEW GAME" : "START";
    case MenuAction::WatchAi:
        return "WATCH AI";
    case MenuAction::Replays:
        return "REPLAYS";
    case MenuAction::Help:
        return "HELP";
    case MenuAction::Options:
        return "OPTIONS";
    case MenuAction::Highscores:
        return "HIGHSCORES";
    case MenuAction::About:
        return "ABOUT";
    case MenuAction::Exit:
        return "EXIT";
    }
    return "";
}

int GameWindow::options_count() noexcept {
    return 4; // AUDIO, MUSIC, FULLSCREEN, BACK
}

std::string_view GameWindow::options_label(int index) const {
    switch (index) {
    case 0:
        return audio_on_ ? "AUDIO ON" : "AUDIO OFF";
    case 1:
        return music_on_ ? "MUSIC ON" : "MUSIC OFF";
    case 2:
        return fullscreen_ ? "FULLSCREEN ON" : "FULLSCREEN OFF";
    default:
        return "BACK";
    }
}

int GameWindow::active_count() const noexcept {
    return state_ == State::Options ? options_count() : menu_count();
}

std::string_view GameWindow::active_label(int index) const {
    return state_ == State::Options ? options_label(index) : menu_label(index);
}

// The menu list is laid out inside this band (world-height fractions): clear of
// the byline above and the hint lines below. Both the type size and the row step
// are derived from it, so a longer menu shrinks to fit instead of colliding.
namespace {
constexpr float menu_band_top = 0.70f;
constexpr float menu_band_bottom = 0.16f;
constexpr float menu_leading = 1.35f; // row step, in glyph heights
constexpr float menu_glyph_rows = 5.0f;
} // namespace

float GameWindow::menu_text_px() const noexcept {
    // Shrink the type once the list is long enough to need it: seven items at a
    // flat 0.015h overlapped each other. Capped, so short lists look unchanged.
    const float h = sim_.config().world_height;
    const float band = h * (menu_band_top - menu_band_bottom);
    const auto rows = static_cast<float>(std::max(active_count() - 1, 0));
    const float glyph = band / ((rows * menu_leading) + 1.0f);
    return std::min(h * 0.015f, glyph / menu_glyph_rows);
}

float GameWindow::menu_item_top_y(int index) const noexcept {
    // Center the list block in the band between the byline and the bottom hint.
    // The step *shrinks* when the list is long enough to need it: with a fixed
    // 0.09h step, growing the menu to seven items (WATCH AI) pushed START up into
    // "BY JENS KOEHLER" and EXIT down into "ARROWS ENTER OR MOUSE".
    // `first_top` is the top item's top edge; glyphs hang `glyph` below their top.
    const float h = sim_.config().world_height;
    const float glyph = menu_glyph_rows * menu_text_px();
    const float band_top = h * menu_band_top;
    const float band_bottom = h * menu_band_bottom;
    const int count = active_count();
    const float spacing = std::min(h * 0.09f, menu_leading * glyph);
    const float block = static_cast<float>(count - 1) * spacing;
    const float first_top = ((band_top + band_bottom + glyph) * 0.5f) + (block * 0.5f);
    return first_top - (static_cast<float>(index) * spacing);
}

int GameWindow::menu_hit(Vec2 world) const noexcept {
    const float px = menu_text_px();
    const float advance = px * 4.0f; // per-glyph horizontal step (matches draw_text)
    const float center_x = sim_.config().world_width * 0.5f;
    const float pad_x = advance * 0.5f;
    const float pad_y = px * 0.5f;
    for (int i = 0; i < active_count(); ++i) {
        const float top_y = menu_item_top_y(i);
        const float bottom_y = top_y - (5.0f * px); // glyphs span ~5 rows below top_y
        const auto chars = static_cast<float>(active_label(i).size());
        const float half_w = (chars * advance * 0.5f) + pad_x;
        if (std::abs(world.x - center_x) <= half_w && world.y <= (top_y + pad_y) &&
            world.y >= (bottom_y - pad_y)) {
            return i;
        }
    }
    return -1;
}

void GameWindow::end_game() {
    in_progress_ = false;
    accumulator_ = 0.0;
    // sim() and not sim_: a replay's score lives in the player's own simulation.
    final_score_ = sim().score() < 0 ? 0 : sim().score();
    // The highscore table is the human's. A run the agent drove — even partly,
    // before a takeover — never enters it, however well it scored.
    if (!ai_assisted_ && highscores_.qualifies(final_score_)) {
        entry_initials_ = {'A', 'A', 'A'};
        entry_slot_ = 0;
        state_ = State::EnterScore; // arcade initials entry
    } else {
        state_ = State::GameOver;
    }
}

void GameWindow::handle_score_entry(int key) {
    char& slot = entry_initials_[static_cast<std::size_t>(entry_slot_)];
    if (key >= Qt::Key_A && key <= Qt::Key_Z) {
        slot = static_cast<char>('A' + (key - Qt::Key_A));
        entry_slot_ = std::min(entry_slot_ + 1, 2); // typing advances to the next slot
    } else if (key == Qt::Key_Up) {
        slot = (slot >= 'Z') ? 'A' : static_cast<char>(slot + 1);
    } else if (key == Qt::Key_Down) {
        slot = (slot <= 'A') ? 'Z' : static_cast<char>(slot - 1);
    } else if (key == Qt::Key_Left || key == Qt::Key_Backspace) {
        entry_slot_ = std::max(entry_slot_ - 1, 0);
    } else if (key == Qt::Key_Right) {
        entry_slot_ = std::min(entry_slot_ + 1, 2);
    } else if (key == Qt::Key_Return || key == Qt::Key_Enter) {
        highscores_.insert(entry_initials_, final_score_);
        state_ = State::Highscores;
        menu_index_ = 0;
    }
}

void GameWindow::toggle_fullscreen() {
    fullscreen_ = !fullscreen_;
    if (fullscreen_) {
        showFullScreen();
    } else {
        showNormal();
    }
    save_settings();
}

void GameWindow::toggle_audio() {
    audio_on_ = !audio_on_;
    audio_.set_enabled(audio_on_);
    save_settings();
}

void GameWindow::toggle_music() {
    music_on_ = !music_on_;
    audio_.set_music_enabled(music_on_);
    save_settings();
}

void GameWindow::open_menu() {
    state_ = State::Menu;
    menu_index_ = 0;
}

void GameWindow::open_options() {
    state_ = State::Options;
    menu_index_ = 0;
}

void GameWindow::start_game() {
    sim_.reset(++seed_);
    state_ = State::Playing;
    in_progress_ = true;
    started_ = false;
    accumulator_ = 0.0;
    fire_pending_ = false;
    ai_driving_ = false;
    ai_assisted_ = false;
    replay_.reset(); // a new game is never still playing back a recording
    speed_ = 1;
}

/// Hand the controls to the scripted agent. Nothing else changes: it drives the
/// same `Action` primitive through the same `Sim::step`, under the same crosshair
/// and trigger limits, so what you watch is exactly the run `poe eval` measured
/// for this seed — the simulation and the agent are both deterministic, so the
/// seed alone reproduces it.
void GameWindow::start_ai_game() {
    start_game();
    ai_driving_ = true;
    ai_assisted_ = true; // sticky: taking over later does not make it your score
}

/// Play back a recorded run — typically an episode a training run dropped on disk.
/// The recording carries its own Config, so what is shown is the simulation as it
/// was when recorded, not as the app is configured now.
bool GameWindow::watch_replay(const std::string& path) {
    std::optional<replay::Recording> recording = replay::load(path);
    if (!recording.has_value()) {
        return false;
    }
    start_game();
    ai_assisted_ = true; // a watched run is never the human's score
    replay_.emplace(std::move(*recording));
    return true;
}

std::string_view GameWindow::replay_label() const noexcept {
    return replay_.has_value() ? replay_->recording().label_text() : std::string_view{};
}

float GameWindow::replay_progress() const noexcept {
    return replay_.has_value() ? replay_->progress() : 0.0f;
}

/// Where the training loop writes its episodes, by the rule in `md/paths.py`:
/// `$MD_RUNS_DIR`, else `./runs` when it exists, else the per-user data directory
/// this app already keeps its high scores in.
///
/// The working directory alone is not enough. Started from a desktop entry it is
/// `$HOME` or `/`, so an installed game looking "beside the shell" for `runs/`
/// finds nothing — while the trainer, installed the same way, is writing happily
/// into `~/.local/share/MissileDefense/runs`.
static std::filesystem::path runs_directory() {
    if (const char* override_dir = std::getenv("MD_RUNS_DIR"); override_dir != nullptr) {
        return std::filesystem::path{override_dir};
    }
    std::error_code ec;
    if (const std::filesystem::path local{"runs"}; std::filesystem::is_directory(local, ec)) {
        return local; // a checkout keeps behaving exactly as it did
    }
    const QString data = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    return std::filesystem::path{data.toStdString()} / "runs";
}

/// Rescanned on every visit, so a run that is still training shows its newest
/// episodes without restarting the app.
void GameWindow::open_replays() {
    replay_files_.clear();
    replay_names_.clear();
    std::error_code ec; // the directory simply may not exist yet; that is not an error
    for (const auto& entry : std::filesystem::directory_iterator{runs_directory(), ec}) {
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".mdr") {
            continue;
        }
        replay_files_.push_back(entry.path().string());
        std::string name = entry.path().stem().string();
        // The pixel font has no lower case, and no punctuation beyond the period.
        std::ranges::transform(name, name.begin(), [](unsigned char c) {
            return static_cast<char>(c == '_' || c == '-' ? ' ' : std::toupper(c));
        });
        replay_names_.push_back(std::move(name));
    }
    // Newest first: while training, the interesting episode is the latest one.
    std::ranges::sort(replay_files_, std::greater{});
    std::ranges::sort(replay_names_, std::greater{});
    menu_index_ = 0;
    state_ = State::Replays;
}

std::string_view GameWindow::replay_name(int index) const {
    if (index < 0 || index >= replay_count()) {
        return {};
    }
    return replay_names_[static_cast<std::size_t>(index)];
}

/// Seek the active replay by `seconds` of play, forward or back.
void GameWindow::scrub(int seconds) {
    if (!replay_.has_value()) {
        return;
    }
    const auto per_second = static_cast<double>(1.0F / sim_.config().dt);
    const auto delta = static_cast<std::int64_t>(static_cast<double>(seconds) * per_second);
    const auto now = static_cast<std::int64_t>(replay_->ticks_played());
    replay_->seek(static_cast<std::uint64_t>(std::max<std::int64_t>(0, now + delta)));
    accumulator_ = 0.0;
    started_ = false; // restart the wall clock so the seek does not fast-forward
}

void GameWindow::activate(int index) {
    if (state_ == State::Options) {
        switch (index) {
        case 0:
            toggle_audio();
            break;
        case 1:
            toggle_music();
            break;
        case 2:
            toggle_fullscreen();
            break;
        default:
            open_menu(); // BACK
            break;
        }
        return;
    }
    switch (action_at(index)) {
    case MenuAction::Resume:
        state_ = State::Playing;
        started_ = false;
        break;
    case MenuAction::NewGame:
        start_game();
        break;
    case MenuAction::WatchAi:
        start_ai_game();
        break;
    case MenuAction::Replays:
        open_replays();
        break;
    case MenuAction::Help:
        state_ = State::Help;
        break;
    case MenuAction::Options:
        open_options();
        break;
    case MenuAction::Highscores:
        state_ = State::Highscores;
        break;
    case MenuAction::About:
        state_ = State::About;
        break;
    case MenuAction::Exit:
        close();
        break;
    }
}

void GameWindow::advance() {
    // Hide the OS cursor while playing so the on-screen crosshair is the pointer.
    const bool hide = (state_ == State::Playing);
    if (hide != cursor_hidden_) {
        setCursor(hide ? Qt::BlankCursor : Qt::ArrowCursor);
        cursor_hidden_ = hide;
    }
    if (state_ != State::Playing) {
        started_ = false; // restart the wall clock when (re)entering play
        return;
    }
    if (!started_) {
        clock_.start();
        started_ = true;
        return;
    }
    accumulator_ += static_cast<double>(clock_.restart()) / 1000.0;
    accumulator_ = std::min(accumulator_, 0.25); // clamp to avoid a spiral of death

    const auto dt = static_cast<double>(sim_.config().dt);
    while (accumulator_ >= dt) {
        // `speed_` ticks per frame fast-forwards a watched game; it is always 1
        // while a human plays, since their input arrives per frame.
        for (int repeat = 0; repeat < speed_; ++repeat) {
            if (replay_.has_value()) {
                // A recording drives itself: it re-decodes its own action indices
                // against its own sim, exactly as the trainer did. Nothing here
                // may touch sim_, or the replay would stop matching the run.
                const bool played = replay_->tick();
                if (speed_ == 1) {
                    audio_.handle_events(replay_->sim().events());
                }
                if (!played) {
                    end_game();
                    return;
                }
                continue;
            }
            Action action;
            if (ai_driving_) {
                // The agent is just another driver: same Action, same Sim::step,
                // same crosshair and trigger limits a hand is held to.
                action = agent_.act(sim_);
            } else {
                // Steer the crosshair toward the mouse every tick (the sim caps how
                // far it travels); a click fires exactly once, from the battery
                // nearest the crosshair — which is where the shot will detonate.
                action = Action::aim_at(aim_);
                if (fire_pending_) {
                    action.fire = true;
                    action.base = nearest_base_with_ammo(sim_.crosshair());
                    fire_pending_ = false;
                }
            }
            sim_.step(action);
            if (speed_ == 1) {
                audio_.handle_events(sim_.events()); // fast-forward would be a din
            }
            if (sim_.terminated()) {
                break;
            }
        }
        accumulator_ -= dt;
        if (sim_.terminated()) {
            end_game();
            break;
        }
    }
}

void GameWindow::update_aim(float px, float py) {
    const Projection proj =
        Projection::make(sim_.config().world_width, sim_.config().world_height,
                         static_cast<float>(width()), static_cast<float>(height()));
    aim_ = proj.screen_to_world(px, py, static_cast<float>(width()), static_cast<float>(height()));
}

void GameWindow::mouseMoveEvent(QMouseEvent* event) {
    update_aim(static_cast<float>(event->position().x()),
               static_cast<float>(event->position().y()));
    if (state_ == State::Menu || state_ == State::Options) {
        const int hit = menu_hit(aim_); // aim_ is the world point under the cursor
        if (hit >= 0) {
            menu_index_ = hit; // hover highlights the item under the pointer
        }
    }
}

void GameWindow::mousePressEvent(QMouseEvent* event) {
    update_aim(static_cast<float>(event->position().x()),
               static_cast<float>(event->position().y()));
    switch (state_) {
    case State::Menu:
    case State::Options: {
        const int hit = menu_hit(aim_);
        if (hit >= 0) {
            menu_index_ = hit;
            activate(hit); // click an item to activate it
        }
        break;
    }
    case State::Playing:
        fire_pending_ = true; // consumed by the next sim tick (advance)
        break;
    case State::GameOver:
    case State::Highscores:
    case State::Help:
    case State::About:
    case State::Replays:
        open_menu(); // a click dismisses these screens back to the menu
        break;
    case State::EnterScore:
        break; // initials entry is keyboard-only
    }
}

void GameWindow::keyPressEvent(QKeyEvent* event) {
    const int key = event->key();

    // Global toggles, available in every state except while typing initials
    // (where the letters are the input): F = fullscreen, M = music, A = audio.
    if (state_ != State::EnterScore) {
        if (key == Qt::Key_F) {
            toggle_fullscreen();
            return;
        }
        if (key == Qt::Key_M) {
            toggle_music();
            return;
        }
        if (key == Qt::Key_A) {
            toggle_audio();
            return;
        }
    }

    switch (state_) {
    case State::Menu:
    case State::Options:
        if (key == Qt::Key_Up || key == Qt::Key_W) {
            menu_index_ = (menu_index_ + active_count() - 1) % active_count();
        } else if (key == Qt::Key_Down || key == Qt::Key_S) {
            menu_index_ = (menu_index_ + 1) % active_count();
        } else if (key == Qt::Key_Return || key == Qt::Key_Enter) {
            activate(menu_index_);
        } else if (key == Qt::Key_Escape) {
            if (state_ == State::Options) {
                open_menu(); // Escape leaves options
            } else if (in_progress_) {
                state_ = State::Playing; // Escape resumes the paused game
                started_ = false;
            }
        }
        break;
    case State::Playing:
        if (key == Qt::Key_Escape || key == Qt::Key_P) {
            open_menu(); // pause -> menu (game frozen and preserved)
        } else if (replay_.has_value() && key == Qt::Key_T) {
            // Take over from a recording: adopt the state it has reached and drop
            // the player. The sim is a value, so this is a copy — from here on the
            // run diverges from what was recorded, which is the point.
            sim_ = replay_->sim();
            aim_ = sim_.crosshair();
            replay_.reset();
            speed_ = 1;
        } else if (ai_driving_ && key == Qt::Key_T) {
            // Take over mid-game. The sim is a value and the agent holds no state,
            // so switching the action source is all it takes — the run simply
            // continues from here under new management.
            ai_driving_ = false;
            speed_ = 1;
        } else if ((ai_driving_ || replay_.has_value()) &&
                   (key == Qt::Key_BracketRight || key == Qt::Key_Plus || key == Qt::Key_Equal)) {
            speed_ = std::min(speed_ * 2, 8);
        } else if ((ai_driving_ || replay_.has_value()) &&
                   (key == Qt::Key_BracketLeft || key == Qt::Key_Minus)) {
            speed_ = std::max(speed_ / 2, 1);
        } else if (replay_.has_value() && key == Qt::Key_Left) {
            scrub(-5); // a recording can be rewound; a live game cannot
        } else if (replay_.has_value() && key == Qt::Key_Right) {
            scrub(5);
        } else if (replay_.has_value() && key == Qt::Key_R) {
            replay_->restart();
            accumulator_ = 0.0;
            started_ = false;
        }
        break;
    case State::Replays: {
        const bool confirm = key == Qt::Key_Return || key == Qt::Key_Enter;
        // Escape always backs out; so does Enter when there is nothing to choose.
        if (key == Qt::Key_Escape || (confirm && replay_count() == 0)) {
            open_menu();
        } else if (replay_count() > 0) {
            if (key == Qt::Key_Up || key == Qt::Key_W) {
                menu_index_ = (menu_index_ + replay_count() - 1) % replay_count();
            } else if (key == Qt::Key_Down || key == Qt::Key_S) {
                menu_index_ = (menu_index_ + 1) % replay_count();
            } else if (confirm) {
                const std::string path = replay_files_[static_cast<std::size_t>(menu_index_)];
                if (!watch_replay(path)) {
                    open_menu(); // unreadable (wrong build, truncated): do not pretend
                }
            }
        }
        break;
    }
    case State::GameOver:
    case State::Highscores:
    case State::Help:
    case State::About:
        if (key == Qt::Key_Return || key == Qt::Key_Enter || key == Qt::Key_Escape) {
            open_menu();
        }
        break;
    case State::EnterScore:
        handle_score_entry(key);
        break;
    }
}

BaseId GameWindow::nearest_base_with_ammo(Vec2 target) const {
    const auto bases = sim_.bases();
    std::uint32_t best = 0;
    float best_dist = 1.0e30f;
    bool found = false;
    for (std::uint32_t i = 0; i < bases.size(); ++i) {
        if (!bases[i].alive || bases[i].ammo == 0) {
            continue;
        }
        const float d = std::abs(bases[i].pos.x - target.x);
        if (d < best_dist) {
            best_dist = d;
            best = i;
            found = true;
        }
    }
    if (!found) { // no ammo anywhere; pick the nearest so the click is a no-op
        for (std::uint32_t i = 0; i < bases.size(); ++i) {
            const float d = std::abs(bases[i].pos.x - target.x);
            if (d < best_dist) {
                best_dist = d;
                best = i;
            }
        }
    }
    return static_cast<BaseId>(best);
}

} // namespace md
