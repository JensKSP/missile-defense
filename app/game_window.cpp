// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "game_window.hpp"

#include "md/protocol.hpp"
#include "projection.hpp"
#include "renderer.hpp"

#include <QCoreApplication>
#include <QCursor>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QProcess>
#include <QSettings>
#include <QStandardPaths>
#include <QStringList>
#include <QTimer>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <print>
#include <system_error>

namespace md {

GameWindow::GameWindow() {
    sim_.reset(seed_);
    aim_ = Vec2{sim_.config().world_width * 0.5f, sim_.config().world_height * 0.5f};
    highscores_.load();
    load_settings(); // restore audio/music/fullscreen from the previous session
    // Once, here, rather than whenever the menu is drawn: the answer cannot
    // change while the game is running, and it costs a handful of stat() calls.
    trainer_ = trainer::command(
        trainer::machine_lookup(QCoreApplication::applicationFilePath().toStdString()));
    // The bundled agent, if this build ships one. A missing file is the normal
    // state today and not an error; a *present but unreadable* one is worth
    // saying out loud, because it means the package is broken rather than lean.
    if (const std::filesystem::path bundled = pretrained_path(); !bundled.empty()) {
        try {
            pretrained_ = agent::Policy::load(bundled);
            pretrained_label_ = pretrained_->display_name();
            std::ranges::transform(
                pretrained_label_, pretrained_label_.begin(),
                [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            if (pretrained_label_.empty()) {
                pretrained_label_ = "PRETRAINED";
            }
        } catch (const agent::Policy::Error& error) {
            std::println(stderr, "bundled model could not be loaded: {}", error.what());
        }
    }
}

/// Where a bundled learned policy lives, or empty when this build ships none.
///
/// Beside the executable on every platform, which is the one layout all three
/// installers already produce: `md_app.exe` and `models/` in the same directory
/// on Windows, `Contents/Resources/models` inside the macOS bundle, and
/// `/usr/share/missile-defense/models` from the Debian package — reached from
/// `/usr/games/missile-defense` by the relative hop below. A checkout finds the
/// source tree's own `models/`, so `--watch-model` is not the only way to try
/// one during development.
/// Every `.mdp` the package ships, sorted by name.
///
/// A list rather than one named file: the game bundles the same network at three
/// points in its training — `learned-low`, `learned-medium`, `learned-high` —
/// so a player can watch one policy acquire ammunition discipline instead of
/// taking the claim on trust. Looking for a single `pretrained.mdp` would have
/// shown one of them and silently dropped the other two.
///
/// Sorted, because the directory order is the filesystem's and the menu's is a
/// ladder: low, medium, high is the order the names already spell.
std::vector<std::filesystem::path> GameWindow::bundled_models() {
    const QString here = QCoreApplication::applicationDirPath();
    const std::array<QString, 4> candidates{
        here + "/models",
        here + "/../Resources/models",             // macOS bundle
        here + "/../share/missile-defense/models", // Debian
        here + "/../../../models",                 // build/<preset>/app -> the checkout
    };
    for (const QString& candidate : candidates) {
        const QString resolved = QDir::cleanPath(candidate);
        if (!QFileInfo::exists(resolved)) {
            continue;
        }
        std::vector<std::filesystem::path> found;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator{
                 std::filesystem::path{resolved.toStdString()}, ec}) {
            if (entry.is_regular_file(ec) && entry.path().extension() == ".mdp") {
                found.push_back(entry.path());
            }
        }
        if (!found.empty()) {
            // Ladder order, not alphabetical — which would read high, low,
            // medium and make a progression look like a shuffle. The suffix is
            // the rung; anything without one sorts after, by name.
            static constexpr std::array<std::string_view, 3> rungs{"-low", "-medium", "-high"};
            const auto rank = [](const std::filesystem::path& path) {
                const std::string stem = path.stem().string();
                for (std::size_t i = 0; i < rungs.size(); ++i) {
                    if (stem.ends_with(rungs[i])) {
                        return i;
                    }
                }
                return rungs.size();
            };
            std::ranges::sort(found, [&rank](const auto& a, const auto& b) {
                const std::size_t ra = rank(a);
                const std::size_t rb = rank(b);
                return ra != rb ? ra < rb : a < b;
            });
            return found;
        }
    }
    return {};
}

std::filesystem::path GameWindow::pretrained_path() {
    // The strongest bundled model, which is the one the HUD names and `--report`
    // means by `pretrained`. Last by name is `learned-high` by construction.
    const std::vector<std::filesystem::path> models = bundled_models();
    return models.empty() ? std::filesystem::path{} : models.back();
}

// Persisted via QSettings — QGuiApplication's organization/application name (set
// in main) route this to the platform store (registry on Windows, INI elsewhere).
void GameWindow::load_settings() {
    const QSettings settings;
    audio_on_ = settings.value("audio/sfx", audio_on_).toBool();
    music_on_ = settings.value("audio/music", music_on_).toBool();
    fullscreen_ = settings.value("video/fullscreen", fullscreen_).toBool();
    match_wave_sync_ = settings.value("match/wave_sync", match_wave_sync_).toBool();
    // Clamped rather than trusted: this is a number in a file a person can edit,
    // and an out-of-range one would index a switch that has three arms.
    const int skill = settings.value("ai/skill", static_cast<int>(ai_skill_)).toInt();
    ai_skill_ = agent::Skill::high;
    if (skill == static_cast<int>(agent::Skill::low)) {
        ai_skill_ = agent::Skill::low;
    } else if (skill == static_cast<int>(agent::Skill::medium)) {
        ai_skill_ = agent::Skill::medium;
    }
    agent_ = agent::Heuristic{agent::params_for(ai_skill_)};
    audio_.set_enabled(audio_on_);
    audio_.set_music_enabled(music_on_);
    // Fullscreen is applied by main() at startup (before the window is shown).
}

void GameWindow::set_silent() noexcept {
    silent_ = true;
    audio_on_ = false;
    music_on_ = false;
    audio_.set_enabled(false);
    audio_.set_music_enabled(false);
}

void GameWindow::save_settings() const {
    if (silent_) {
        // Borrowed, not changed: a run told to be quiet must not be the reason
        // the next human to start the game finds the sound switched off.
        return;
    }
    QSettings settings;
    settings.setValue("audio/sfx", audio_on_);
    settings.setValue("audio/music", music_on_);
    settings.setValue("video/fullscreen", fullscreen_);
    settings.setValue("match/wave_sync", match_wave_sync_);
    settings.setValue("ai/skill", static_cast<int>(ai_skill_));
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

int GameWindow::menu_count() const noexcept {
    // Eight on a full install, seven on a game-only one. TRAIN AI is *absent*
    // rather than disabled: on the game-only package there is no trainer to
    // enable, so an item explaining that would be advertising a product the
    // person did not install. The layout follows — menu_item_top_y() derives
    // its row step from active_count(), so one more item shrinks the list
    // instead of pushing START into the byline.
    //
    // Starting a game *adds* RESUME and removes nothing. WATCH AI, TRAIN AI
    // and ABOUT used to disappear here, which made the pause menu a different
    // menu wearing the same frame: every item below the first moved, so the
    // muscle memory built in the main menu was wrong exactly when a player was
    // mid-game and least willing to re-read it.
    //
    // REPLAYS is withdrawn for now — see `action_at`. Everything behind it is
    // still here and still reachable, so this is one number and one line in the
    // table below rather than a feature to put back.
    return (in_progress_ ? 1 : 0) + (can_train() ? 8 : 7);
}

GameWindow::MenuAction GameWindow::action_at(int index) const {
    auto slot = static_cast<std::size_t>(index);
    if (in_progress_) {
        if (slot == 0) {
            return MenuAction::Resume;
        }
        --slot; // the rest of the list is the main menu's, unchanged
    }
    // TRAIN AI sits next to WATCH AI, which is the other thing in this menu
    // about the agent rather than about playing.
    //
    // REPLAYS is commented out rather than deleted: the browser, the discovery
    // and the playback all still work and are still tested, and the MODELS
    // browser reaches the same screen through WATCH AI. What it has no answer
    // for yet is the person who opens it on a machine that has never trained
    // anything — an empty list is the *only* thing it can say, and it is the
    // common case. Put the line back when there is something to show there.
    const std::array<MenuAction, 8> acts{MenuAction::NewGame,
                                         MenuAction::WatchAi,
                                         MenuAction::TrainAi,
                                         /* MenuAction::Replays, */ MenuAction::Help,
                                         MenuAction::Options,
                                         MenuAction::Highscores,
                                         MenuAction::About,
                                         MenuAction::Exit};
    return acts[can_train() || slot < 2 ? slot : slot + 1];
}

std::string_view GameWindow::menu_label(int index) const {
    switch (action_at(index)) {
    case MenuAction::Resume:
        return "RESUME";
    case MenuAction::NewGame:
        return in_progress_ ? "NEW GAME" : "START";
    case MenuAction::WatchAi:
        return "WATCH AI";
    case MenuAction::WatchScripted:
        return "SCRIPTED";
    case MenuAction::WatchPretrained:
        return "PRETRAINED";
    case MenuAction::TrainAi:
        return "TRAIN AI";
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

namespace {
/// The scripted ladder, in the order the menu offers it — worst first, so the
/// list reads as a difficulty ramp rather than as three unrelated agents.
/// Indexed by `Skill`, so `watch_skills[skill].label` is the HUD's name for it.
struct Rung {
    agent::Skill skill;
    std::string_view label; // upper case: this font has no lower
    std::string_view name;  // what `--watch-scripted` accepts
};

constexpr std::array<Rung, 3> watch_skills{{
    {agent::Skill::low, "SCRIPTED LOW", "low"},
    {agent::Skill::medium, "SCRIPTED MEDIUM", "medium"},
    {agent::Skill::high, "SCRIPTED HIGH", "high"},
}};
} // namespace

int GameWindow::options_count() noexcept {
    return 5; // AUDIO, MUSIC, FULLSCREEN, AI SKILL, BACK
}

/// The WATCH AI submenu. Two agents and a way back — or, until a model ships,
/// only the scripted one, in which case WATCH AI never opens this screen at all
/// (see `activate`). A chooser with one choice on it is a screen that wastes a
/// keypress and implies a second option exists.
int GameWindow::watch_count() const noexcept {
    // Three scripted rungs, MODELS when this install has any, and BACK.
    return static_cast<int>(watch_skills.size()) + (installed_models_ > 0 ? 2 : 1);
}

std::string_view GameWindow::watch_label(int index) const {
    const auto rungs = static_cast<int>(watch_skills.size());
    if (index >= 0 && index < rungs) {
        return watch_skills[static_cast<std::size_t>(index)].label;
    }
    if (installed_models_ > 0 && index == rungs) {
        // A list and not the one model's name: an install can now have several,
        // and a row that names one of them hides the rest.
        return "MODELS";
    }
    return "BACK";
}

std::string_view GameWindow::options_label(int index) const {
    switch (index) {
    case 0:
        return audio_on_ ? "AUDIO ON" : "AUDIO OFF";
    case 1:
        return music_on_ ? "MUSIC ON" : "MUSIC OFF";
    case 2:
        return fullscreen_ ? "FULLSCREEN ON" : "FULLSCREEN OFF";
    case 3:
        // Which scripted agent WATCH AI runs. Named rather than numbered
        // because each step is a behaviour switched off, not a difficulty
        // multiplier — see `md::agent::Skill`.
        if (ai_skill_ == md::agent::Skill::low) {
            return "AI SKILL LOW";
        }
        return ai_skill_ == md::agent::Skill::medium ? "AI SKILL MEDIUM" : "AI SKILL HIGH";
    default:
        return "BACK";
    }
}

int GameWindow::active_count() const noexcept {
    switch (state_) {
    case State::Options:
        return options_count();
    case State::Watch:
        return watch_count();
    default:
        return menu_count();
    }
}

std::string_view GameWindow::active_label(int index) const {
    switch (state_) {
    case State::Options:
        return options_label(index);
    case State::Watch:
        return watch_label(index);
    default:
        return menu_label(index);
    }
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

float GameWindow::replay_row_px() const noexcept {
    return sim_.config().world_height * 0.011f;
}

float GameWindow::replay_row_top_y(int index) const noexcept {
    const float h = sim_.config().world_height;
    return h * (0.76f - (static_cast<float>(index - replay_scroll_) * 0.075f));
}

int GameWindow::replay_hit(Vec2 world) const noexcept {
    const float px = replay_row_px();
    const float advance = px * 4.0f; // per-glyph horizontal step (matches draw_text)
    const float center_x = sim_.config().world_width * 0.5f;
    const float pad_x = advance * 0.5f;
    const float pad_y = px * 0.5f;
    const int last = std::min(replay_count(), replay_scroll_ + replay_rows_visible);
    for (int i = replay_scroll_; i < last; ++i) {
        const float top_y = replay_row_top_y(i);
        const float bottom_y = top_y - (5.0f * px); // glyphs span ~5 rows below top_y
        const auto chars = static_cast<float>(replay_name(i).size());
        const float half_w = (chars * advance * 0.5f) + pad_x;
        if (std::abs(world.x - center_x) <= half_w && world.y <= (top_y + pad_y) &&
            world.y >= (bottom_y - pad_y)) {
            return i;
        }
    }
    return -1;
}

void GameWindow::scroll_replays_into_view() noexcept {
    const int count = replay_count();
    const int most = std::max(0, count - replay_rows_visible);
    if (menu_index_ < replay_scroll_) {
        replay_scroll_ = menu_index_;
    } else if (menu_index_ >= replay_scroll_ + replay_rows_visible) {
        replay_scroll_ = menu_index_ - replay_rows_visible + 1;
    }
    replay_scroll_ = std::clamp(replay_scroll_, 0, most);
}

void GameWindow::open_selected() {
    if (menu_index_ < 0 || menu_index_ >= replay_count()) {
        return;
    }
    const std::string path = replay_files_[static_cast<std::size_t>(menu_index_)];
    const bool started = browse_ == Browse::Models ? watch_model(path) : watch_replay(path);
    if (!started) {
        open_menu(); // unreadable (wrong build, truncated): do not pretend
    }
}

bool GameWindow::finished() const noexcept {
    if (frame_budget_ != 0 && frames_ > frame_budget_) {
        return true;
    }
    // GameOver *and* EnterScore: a qualifying score diverts to initials entry,
    // and a run driven to its end unattended has nobody to type them.
    // A match never reaches GameOver — it has no single game to end — so its own
    // completion is what an unattended run waits for.
    if (exit_when_done_ && state_ == State::Match) {
        return match_.has_value() && match_->finished();
    }
    return exit_when_done_ && (state_ == State::GameOver || state_ == State::EnterScore);
}

void GameWindow::advance_death(float seconds) {
    dying_for_ += seconds;
    // Age the copies exactly as `Sim` would have, using the same curve, so the
    // explosion the player watches is the one the simulation resolved.
    for (Blast& blast : dying_blasts_) {
        blast.age += seconds;
        blast.radius = blast_radius(blast.age, sim().config());
    }
    std::erase_if(dying_blasts_, [this](const Blast& blast) {
        return blast.age >= sim().config().blast_lifetime;
    });
    for (Explosion& explosion : dying_explosions_) {
        explosion.age += seconds;
        explosion.radius = explosion_radius(explosion, sim().config());
    }
    std::erase_if(dying_explosions_, [this](const Explosion& explosion) {
        return explosion.age >= sim().config().explosion_lifetime;
    });
    if (dying_for_ >= death_seconds) {
        dying_for_ = -1.0F;
        dying_blasts_.clear();
        dying_explosions_.clear();
        end_game();
    }
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

void GameWindow::cycle_ai_skill() {
    // LOW -> MEDIUM -> HIGH -> LOW. A three-way setting on a menu with no
    // left/right affordance has to cycle; wrapping at the top is what makes one
    // key enough to reach all three.
    if (ai_skill_ == agent::Skill::low) {
        ai_skill_ = agent::Skill::medium;
    } else if (ai_skill_ == agent::Skill::medium) {
        ai_skill_ = agent::Skill::high;
    } else {
        ai_skill_ = agent::Skill::low;
    }
    // Rebuilt rather than mutated: `Heuristic` holds its parameters by value and
    // a game already under way should pick the new skill up on its next
    // decision, not at the next round.
    agent_ = agent::Heuristic{agent::params_for(ai_skill_)};
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
    // `++seed_` and not `seed_`: each new game is a new problem. Unless one was
    // pinned with `--seed`, in which case it is the *same* problem on purpose —
    // that is what a peek at a running evaluation is.
    sim_.reset(pinned_seed_ ? seed_ : ++seed_);
    state_ = State::Playing;
    in_progress_ = true;
    started_ = false;
    accumulator_ = 0.0;
    fire_latch_.clear();
    ai_driving_ = false;
    ai_assisted_ = false;
    // A new game has no contestant until one is chosen, and a stale decorator
    // would point at a driver that is about to be replaced.
    handicapped_.reset();
    scripted_driver_.reset();
    dying_for_ = -1.0F; // and it is certainly not still dying
    dying_blasts_.clear();
    dying_explosions_.clear();
    replay_.reset();       // a new game is never still playing back a recording
    watch_driver_.reset(); // ... nor still being driven by the last policy
    driver_name_.clear();
    speed_ = 1;
}

/// Hand the controls to the scripted agent. Nothing else changes: it drives the
/// same `Action` primitive through the same `Sim::step`, under the same crosshair
/// and trigger limits, so what you watch is exactly the run `poe eval` measured
/// for this seed — the simulation and the agent are both deterministic, so the
/// seed alone reproduces it.
void GameWindow::start_ai_game(agent::Skill skill) {
    start_game();
    agent_ = agent::Heuristic{agent::params_for(skill)};
    ai_driving_ = true;
    ai_assisted_ = true; // sticky: taking over later does not make it your score
    watch_driver_.reset();
    // Through the published handicap, so the rung on screen plays the game the
    // trainer's ladder was measured on.
    scripted_driver_.emplace(skill);
    handicapped_.emplace(*scripted_driver_, agent::canonical_handicap);
    // The rung, on screen. "SCRIPTED" alone was fine when there was one; with
    // three that differ by ~78,000 points it would be the most misleading label
    // in the game — you would have no way to tell which one you were watching.
    driver_name_ = std::string{watch_skills[static_cast<std::size_t>(skill)].label};
}

std::optional<agent::Skill> GameWindow::skill_named(std::string_view name) {
    for (const auto& rung : watch_skills) {
        if (rung.name == name) {
            return rung.skill;
        }
    }
    return std::nullopt;
}

/// The same, with a learned policy at the controls instead.
///
/// `watched_` holds the policy and `watch_driver_` the thing that turns an
/// observation into an `Action` — the *same* `PolicyDriver` the evaluator uses,
/// so what is on screen is the agent `md_agent_eval` scores and not a second
/// implementation of it that might steer differently.
void GameWindow::start_model_game(std::optional<agent::Policy> policy) {
    if (policy.has_value()) {
        watched_ = std::move(policy);
    } else if (pretrained_.has_value()) {
        watched_ = pretrained_;
    } else {
        return;
    }
    // Constructed before `start_game`, because a policy trained against a
    // different observation encoding throws here — and a refusal must not
    // leave a game running with nobody driving it.
    // Constructed before `start_game`, because the constructor throws when the
    // policy was trained against a different observation encoding — and a
    // refusal must not leave a game running with nobody driving it.
    agent::PolicyDriver driver{*watched_, ObsSpec{}};
    start_game(); // clears watch_driver_, so the move below has to follow it
    driver_name_ = driver.name();
    watch_driver_ = std::move(driver);
    // A learned policy wears the same handicap the scripted rungs do. Anything
    // else would make "beat the baseline" a race between a contestant carrying
    // weights and one that is not.
    scripted_driver_.reset();
    handicapped_.emplace(*watch_driver_, agent::canonical_handicap);
    ai_driving_ = true;
    ai_assisted_ = true;
}

/// `--watch-model <path>`: play a policy from anywhere on disk.
///
/// The development and packaging-test path, and the one that makes a promoted
/// model watchable before it is bundled. A refusal is loud rather than a silent
/// fall back to the scripted agent: watching the wrong agent and not being told
/// is worse than not watching at all.
bool GameWindow::watch_model(const std::string& path) {
    try {
        start_model_game(agent::Policy::load(path));
    } catch (const agent::Policy::Error& error) {
        std::println(stderr, "could not load the model: {}", error.what());
        return false;
    }
    return true;
}

/// The WATCH AI submenu, when there is more than one agent to choose between.
void GameWindow::open_watch() {
    // Counted on every visit, not at startup: the trainer can promote a model
    // while the game sits on its menu, and a MODELS row that only appears after
    // a restart is a feature the person has no reason to look for again.
    installed_models_ = installed_model_count();
    state_ = State::Watch;
    menu_index_ = 0;
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

namespace {

/// Report why a match could not be shown, once, on stderr.
///
/// A refusal and not a fallback: a "match" that quietly showed one side, or two
/// unrelated episodes, would be a comparison the viewer had no way to distrust.
bool refuse_match(const std::exception& why) {
    std::cerr << "md_app: " << why.what() << '\n';
    return false;
}

} // namespace

bool GameWindow::watch_match(const std::string& manifest) {
    try {
        match_.emplace(replay::MatchPlayer::load(manifest));
    } catch (const replay::MatchPlayer::Error& error) {
        return refuse_match(error);
    }
    match_->set_wave_sync(match_wave_sync_);
    state_ = State::Match;
    match_paused_ = false;
    ai_assisted_ = true; // nothing here is the human's score
    started_ = false;
    return true;
}

bool GameWindow::watch_match(const std::string& left, const std::string& right) {
    try {
        match_.emplace(replay::MatchPlayer::pair(left, right));
    } catch (const replay::MatchPlayer::Error& error) {
        return refuse_match(error);
    }
    match_->set_wave_sync(match_wave_sync_);
    state_ = State::Match;
    match_paused_ = false;
    ai_assisted_ = true;
    started_ = false;
    return true;
}

std::string_view GameWindow::replay_label() const noexcept {
    return replay_.has_value() ? replay_->recording().label_text() : std::string_view{};
}

float GameWindow::replay_progress() const noexcept {
    return replay_.has_value() ? replay_->progress() : 0.0f;
}

/// Where the training loop writes its episodes, by the rule in `missile_defense/paths.py`:
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
    if (const std::filesystem::path local{protocol::runs_dir};
        std::filesystem::is_directory(local, ec)) {
        return local; // a checkout keeps behaving exactly as it did
    }
    const QString data = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    return std::filesystem::path{data.toStdString()} / protocol::runs_dir;
}

/// Where the trainer installs models, by the rule in `missile_defense/paths.py`:
/// `$MD_MODELS_DIR`, else a `models/` sibling of the runs directory.
///
/// Sibling and not a subdirectory of a run: a model is promoted precisely so it
/// survives the run being cleaned up, archived or deleted, and storing it
/// inside the thing it must outlive would defeat that on the first tidy-up.
static std::filesystem::path models_directory() {
    if (const char* override_dir = std::getenv("MD_MODELS_DIR"); override_dir != nullptr) {
        return std::filesystem::path{override_dir};
    }
    return runs_directory().parent_path() / protocol::models_dir;
}

/// A name as this menu can draw it: the pixel font has no lower case, and no
/// punctuation but the period.
static std::string menu_label(std::string name) {
    std::ranges::transform(name, name.begin(), [](unsigned char c) {
        return static_cast<char>(c == '_' || c == '-' ? ' ' : std::toupper(c));
    });
    return name;
}

/// Every model this install can play: the bundled one, then whatever the
/// trainer has promoted into the league.
///
/// `Policy::describe` and not `Policy::load`: naming eight models on a screen
/// should not read eight multi-megabyte tensor blocks and verify eight
/// checksums. A file that cannot be described is skipped rather than listed —
/// offering something that will fail on Enter is worse than not offering it.
///
/// Names on this screen are made unique, because two rows reading `DEADLINE
/// 1330` are two rows nobody can choose between. `missile_defense.league` refuses a
/// duplicate display name at the moment a model is promoted, which handles
/// every model this program put there — but not one copied in by hand, not one
/// promoted before that rule existed, and not the case this screen creates for
/// itself: uppercasing folds `Amber Anvil` and `amber-anvil` onto the same
/// label. So a repeat falls back to the directory, which is unique by
/// construction.
static std::vector<std::pair<std::string, std::string>> installed_models() {
    std::vector<std::pair<std::string, std::string>> found;

    // What this build's observation looks like. A model trained against an older
    // one parses fine and then cannot be run — listing it would offer something
    // that fails on Enter, which is worse than not offering it.
    const std::size_t expected = ObsSpec{}.size();

    const auto add = [&found, expected](const std::filesystem::path& path) {
        try {
            const auto described = agent::Policy::describe(path);
            if (described.observation_size != expected) {
                std::println(stderr,
                             "skipping {}: it expects a {}-feature observation, this "
                             "build has {} — retrain or re-export it",
                             path.string(), described.observation_size, expected);
                return;
            }
            // Two layouts, and the fallback has to tell them apart. A promoted
            // model is `models/<name>/policy.mdp`, where the *directory* is the
            // name; a bundled one is `models/<name>.mdp`, where the file is. Using
            // the parent for both listed every bundled model as `MODELS`, because
            // that is what their shared directory is called.
            std::string name = described.display_name;
            if (name.empty()) {
                name = path.filename() == "policy.mdp" ? path.parent_path().filename().string()
                                                       : path.stem().string();
            }
            found.emplace_back(path.string(), menu_label(std::move(name)));
        } catch (const agent::Policy::Error& error) {
            // Not this build's, or not a policy. Absent from the list rather
            // than offered and then failing on Enter — but said out loud, since
            // a model the trainer installed and the game will not show is
            // exactly the situation a person needs a reason for.
            std::println(stderr, "skipping {}: {}", path.string(), error.what());
        }
    };

    for (const std::filesystem::path& bundled : GameWindow::bundled_models()) {
        add(bundled);
    }
    std::error_code ec; // the directory simply may not exist yet; that is not an error
    for (const auto& entry : std::filesystem::directory_iterator{models_directory(), ec}) {
        if (!entry.is_directory(ec)) {
            continue;
        }
        const std::filesystem::path policy = entry.path() / "policy.mdp";
        if (std::filesystem::is_regular_file(policy, ec)) {
            add(policy);
        }
    }

    // Repeats fall back to the directory the model lives in — `AMBER ANVIL 2`
    // rather than a second `AMBER ANVIL`. The directory and not a counter,
    // because `directory_iterator` promises no order: a counter would move
    // between launches and label a different model each time, which is worse
    // than the duplicate it fixes.
    std::map<std::string, int> times_seen;
    for (const auto& [path, name] : found) {
        ++times_seen[name];
    }
    for (auto& [path, name] : found) {
        if (times_seen[name] > 1) {
            name = menu_label(std::filesystem::path{path}.parent_path().filename().string());
        }
    }
    return found;
}

int GameWindow::installed_model_count() {
    return static_cast<int>(installed_models().size());
}

/// Rescanned on every visit, so a model promoted while the game was open shows
/// up without restarting it.
void GameWindow::open_models() {
    replay_files_.clear();
    replay_names_.clear();
    for (auto& [path, name] : installed_models()) {
        replay_files_.push_back(std::move(path));
        replay_names_.push_back(std::move(name));
    }
    browse_ = Browse::Models;
    menu_index_ = 0;
    replay_scroll_ = 0;
    state_ = State::Replays;
}

namespace {

/// One discovered recording: where it is, what to call it, and when it landed.
///
/// Path, label and sort key travel together deliberately. They used to be two
/// vectors sorted independently — and since a label is the path uppercased with
/// separators turned into spaces, the two orders are *not* the same one, so a
/// row could show one episode's name above another episode's file.
struct Found {
    std::filesystem::path path;
    std::string label;
    std::filesystem::file_time_type modified{};
};

/// Fold a filename into something the pixel font can draw.
std::string shout_stem(std::string name) {
    std::ranges::transform(name, name.begin(), [](unsigned char c) {
        return static_cast<char>(c == '_' || c == '-' ? ' ' : std::toupper(c));
    });
    return name;
}

/// Add every `.mdr` directly inside `directory` to `found`.
void collect_recordings(const std::filesystem::path& directory, std::string_view run,
                        std::vector<Found>& found) {
    std::error_code ec; // the directory simply may not exist yet; that is not an error
    for (const auto& entry : std::filesystem::directory_iterator{directory, ec}) {
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".mdr") {
            continue;
        }
        std::string label = shout_stem(entry.path().stem().string());
        if (!run.empty()) {
            // Which *run* produced it, first. Every run names its episodes
            // `update-00025`, so a flat list of those is a list of duplicates.
            std::string named = shout_stem(std::string{run});
            named += "  ";
            named += label;
            label = std::move(named);
        }
        if (label.size() > 40) {
            label = label.substr(0, 37) + "...";
        }
        found.push_back(Found{entry.path(), std::move(label), entry.last_write_time(ec)});
    }
}

/// Rescanned on every visit, so a run that is still training shows its newest
/// episodes without restarting the app.
///
/// Two levels, because there are two layouts and both are real: a runs
/// directory whose episodes sit directly in it (what `--out runs/` produces),
/// and a *library* of managed runs, each its own directory (what the trainer
/// creates). Scanning only the first is why the browser was empty for anyone
/// who had used the trainer — every recording they could see there was one
/// level down from where the game looked.
///
/// Not recursive beyond that, and no symlinks followed: a runs directory is a
/// place a person points at, and walking an arbitrary tree from it is how a
/// browser ends up listing somebody's home directory.
std::vector<Found> discovered_recordings(const std::filesystem::path& root) {
    std::vector<Found> found;
    collect_recordings(root, {}, found);

    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator{root, ec}) {
        // `is_directory` and not `status().type()`: a symlink to a directory
        // answers true to the former, so it is checked separately.
        if (!entry.is_directory(ec) || entry.is_symlink(ec)) {
            continue;
        }
        collect_recordings(entry.path(), entry.path().filename().string(), found);
    }

    // Newest first, once, over whole records. While training, the interesting
    // episode is the one just written — whatever it happens to be called.
    std::ranges::sort(found, [](const Found& a, const Found& b) {
        return a.modified != b.modified ? a.modified > b.modified : a.path > b.path;
    });
    return found;
}

} // namespace

int GameWindow::discovered_recording_count() {
    return static_cast<int>(discovered_recordings(runs_directory()).size());
}

void GameWindow::open_replays() {
    browse_ = Browse::Replays;
    replay_files_.clear();
    replay_names_.clear();

    for (Found& record : discovered_recordings(runs_directory())) {
        replay_files_.push_back(record.path.string());
        replay_names_.push_back(std::move(record.label));
    }

    menu_index_ = 0;
    replay_scroll_ = 0; // a fresh visit starts at the top, however it was left
    state_ = State::Replays;
}

/// Start the trainer and stay where we are.
///
/// Detached, and that is the architecture rather than a shortcut: the trainer
/// outlives the game, a run outlives the trainer, and neither should be able to
/// take the other down (docs/ROADMAP.md, M8). So the game does not keep the
/// handle, does not wait, and does not report an exit code — from here it is a
/// separate application that happens to have been started from this menu.
///
/// Nothing visible happens in the game itself, which is deliberate: the trainer
/// is a window of its own and will raise itself. Returning to the menu rather
/// than to a "launching..." screen means a second press simply opens a second
/// trainer, the same as double-clicking its desktop entry twice.
void GameWindow::open_trainer() {
    if (!trainer_.has_value()) {
        return; // no entry is drawn in this case, so this is belt and braces
    }
    QStringList arguments;
    for (std::size_t i = 1; i < trainer_->argv.size(); ++i) {
        arguments << QString::fromStdString(trainer_->argv[i]);
    }
    QProcess process;
    process.setProgram(QString::fromStdString(trainer_->argv.front()));
    process.setArguments(arguments);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    if (!trainer_->python_path.empty()) {
        // The checkout case: `missile_defense` is not installed anywhere this interpreter
        // would find on its own, so the import path has to be handed over.
        environment.insert("PYTHONPATH", QString::fromStdString(trainer_->python_path.string()));
    }
    // Do *not* hand the trainer this game's platform plugin. The game is forced
    // onto `xcb` because it needs an X11 surface for Vulkan and for being
    // screenshot-able; the trainer is Qt Widgets and needs neither. Inherited,
    // that choice put it on XWayland, where NVIDIA's lack of implicit sync
    // produces the tearing and artefacts a Wayland-native window does not have.
    // Removing the variable lets Qt pick for itself, which is Wayland on a
    // Wayland session and xcb on an X11 one.
    environment.remove("QT_QPA_PLATFORM");
    process.setProcessEnvironment(environment);
    process.startDetached();
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

void GameWindow::scrub_match(int seconds) {
    if (!match_.has_value()) {
        return;
    }
    // Seeks the *shared* clock, never one side: `MatchPlayer::seek` moves both,
    // and a side shorter than the target lands on its own final state.
    const auto per_second = static_cast<double>(1.0F / sim_.config().dt);
    const auto delta = static_cast<std::int64_t>(static_cast<double>(seconds) * per_second);
    const auto now = static_cast<std::int64_t>(match_->tick_count());
    match_->seek(static_cast<std::uint64_t>(std::max<std::int64_t>(0, now + delta)));
    // A seek backwards out of a finished match makes it playable again.
    match_paused_ = match_paused_ && match_->finished();
    accumulator_ = 0.0;
    started_ = false;
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
        case 3:
            cycle_ai_skill();
            break;
        default:
            open_menu(); // BACK
            break;
        }
        return;
    }
    if (state_ == State::Watch) {
        const auto rungs = static_cast<int>(watch_skills.size());
        if (index >= 0 && index < rungs) {
            start_ai_game(watch_skills[static_cast<std::size_t>(index)].skill);
        } else if (installed_models_ > 0 && index == rungs) {
            open_models();
        } else {
            open_menu(); // BACK
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
        // Always a choice now: there are three scripted rungs whether or not a
        // model ships, and which one you are watching is the whole question.
        open_watch();
        break;
    case MenuAction::WatchScripted:
        start_ai_game();
        break;
    case MenuAction::WatchPretrained:
        start_model_game();
        break;
    case MenuAction::TrainAi:
        open_trainer();
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

/// Drive both sides of a match on the one clock they share.
///
/// Deliberately not `advance()`'s loop with a branch in it: a match has no
/// `sim_`, no audio (two episodes playing at once is noise, not sound), no
/// human input to sample, and no game to end — almost nothing that loop does
/// applies, and threading four conditions through it would make the live path
/// harder to read for the sake of sharing eight lines of accumulator.
void GameWindow::advance_match() {
    if (!match_.has_value() || match_paused_) {
        started_ = false; // restart the wall clock when play resumes
        return;
    }
    if (!started_) {
        clock_.start();
        started_ = true;
        return;
    }
    accumulator_ += static_cast<double>(clock_.restart()) / 1000.0;
    accumulator_ = std::min(accumulator_, 0.25); // clamp to avoid a spiral of death

    if (dying_for_ >= 0.0F) {
        // The game is lost and the simulation has stopped; the explosions have
        // not. Nothing is stepped here — only the copies age — so a death throe
        // cannot change a score, a recording or an episode's length.
        advance_death(static_cast<float>(accumulator_));
        accumulator_ = 0.0;
        return;
    }

    const auto dt = static_cast<double>(sim_.config().dt);
    while (accumulator_ >= dt) {
        for (int repeat = 0; repeat < speed_; ++repeat) {
            if (!match_->tick()) {
                // Both sides are done. Park on the final frame rather than
                // closing or looping: the result is the thing the viewer came
                // for, and it should still be there when they look up.
                match_paused_ = true;
                break;
            }
        }
        accumulator_ -= dt;
    }
}

void GameWindow::advance() {
    // Counted before anything can return early, because what `--frames` bounds is
    // the *run*, not the part of it spent playing: a window stuck on the menu has
    // to hit its budget too, or the bound it exists to provide is not one.
    ++frames_;
    if (finished() && !closing_) {
        // Deferred out of the render callback, not quit() on the spot. advance()
        // runs from inside startNextFrame(), so tearing the application down here
        // destroys the swapchain and the device while this frame is still being
        // built — which is a segfault at exit rather than a clean run. A queued
        // close lands between frames, where QVulkanWindow releases its resources
        // the way it expects to, and the last window closing ends exec().
        closing_ = true;
        QTimer::singleShot(0, this, &GameWindow::close);
        return;
    }
    // Hide the OS cursor while playing so the on-screen crosshair is the pointer.
    // Not during the death throes: the game is lost, nothing responds to aiming
    // any more, and a hidden pointer over a screen that ignores it reads as a
    // frozen application rather than as an ending.
    const bool hide = (state_ == State::Playing && dying_for_ < 0.0F);
    if (hide != cursor_hidden_) {
        setCursor(hide ? Qt::BlankCursor : Qt::ArrowCursor);
        cursor_hidden_ = hide;
    }
    if (state_ == State::Match) {
        advance_match();
        return;
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

    if (dying_for_ >= 0.0F) {
        // The game is lost and the simulation has stopped; the explosions have
        // not. Nothing is stepped here — only the copies age — so a death throe
        // cannot change a score, a recording or an episode's length.
        advance_death(static_cast<float>(accumulator_));
        accumulator_ = 0.0;
        return;
    }

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
            if (handicapped_.has_value()) {
                // Whichever contestant, wearing the published handicap — the
                // same decorator `md_agent_eval` puts on, so the agent on screen
                // is the one the ladder was measured on rather than a second
                // implementation that might steer differently.
                action = handicapped_->act(sim_);
            } else if (ai_driving_) {
                // Taking over left the agent driving without a contestant: the
                // bare heuristic, as before.
                action = agent_.act(sim_);
            } else {
                // Keep aiming at the mouse; the sim samples that intent at its
                // 15 Hz decision cadence and keeps steering between samples. A
                // click stays pending until the next sample, then fires exactly
                // once from the battery nearest the crosshair.
                action = Action::aim_at(aim_);
                if (fire_latch_.pending()) {
                    fire_latch_.apply(sim_, action, nearest_base_with_ammo(sim_.crosshair()));
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
            // Not `end_game()` yet: the blast that just took the last city is
            // at radius zero, and switching screens here is what cut the
            // explosion off mid-detonation. Take a copy and let it burn.
            dying_for_ = 0.0F;
            const auto live = sim_.blasts();
            dying_blasts_.assign(live.begin(), live.end());
            const auto impacts = sim_.explosions();
            dying_explosions_.assign(impacts.begin(), impacts.end());
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
    // Every centred list, not a list of them: WATCH AI was added to the click
    // and key handlers and missed here, so it was the one screen in the game
    // that did not light up under the pointer. Asking "is this a menu?" once
    // means the next screen cannot be forgotten in the same way.
    if (state_ == State::Menu || state_ == State::Options || state_ == State::Watch) {
        const int hit = menu_hit(aim_); // aim_ is the world point under the cursor
        if (hit >= 0) {
            menu_index_ = hit; // hover highlights the item under the pointer
        }
    } else if (state_ == State::Replays) {
        const int hit = replay_hit(aim_);
        if (hit >= 0) {
            menu_index_ = hit; // the recording list highlights on hover too
        }
    }
}

void GameWindow::mousePressEvent(QMouseEvent* event) {
    update_aim(static_cast<float>(event->position().x()),
               static_cast<float>(event->position().y()));
    switch (state_) {
    case State::Match:
        // Nothing to click: a match is watched, and the mouse would have to
        // mean something different on each half of the screen to mean anything.
        break;
    case State::Menu:
    case State::Options:
    case State::Watch: {
        const int hit = menu_hit(aim_);
        if (hit >= 0) {
            menu_index_ = hit;
            activate(hit); // click an item to activate it
        }
        break;
    }
    case State::Playing:
        // WATCH AI and replay playback have their own action sources. Do not
        // queue a stray mouse edge that would fire immediately if the human
        // later presses T to take over.
        if (!ai_driving_ && !replay_.has_value()) {
            fire_latch_.request(); // consumed at the next action-sampling tick
        }
        break;
    case State::Replays: {
        // A chooser, not a notice board: click a recording to watch it. Clicking
        // off the list still backs out, so the mouse alone can leave the screen.
        const int hit = replay_hit(aim_);
        if (hit >= 0) {
            menu_index_ = hit;
            open_selected();
        } else {
            open_menu();
        }
        break;
    }
    case State::GameOver:
    case State::Highscores:
    case State::Help:
    case State::About:
        open_menu(); // a click dismisses these screens back to the menu
        break;
    case State::EnterScore:
        break; // initials entry is keyboard-only
    }
}

/// Lets go of the Vulkan instance before Qt takes the window apart.
///
/// One line, and without it this game cannot run on Wayland at all — which is
/// the default session on current KDE and GNOME, so it is most of the desktop.
///
/// The defect is Qt's, upstream as QTBUG-123214, and it is a destruction order.
/// `QWindowPrivate::destroy()` runs
///
///     q->setVisible(false);                              // 1
///     QPlatformSurfaceEvent e(SurfaceAboutToBeDestroyed);
///     QGuiApplication::sendEvent(q, &e);                 // 2
///     delete std::exchange(platformWindow, nullptr);     // 3
///
/// Step 1 reaches `QWaylandWindow::reset()`, which tears down the `wl_surface`.
/// Step 2 is where `QVulkanWindow` destroys the swapchain built on that surface,
/// and the driver reads memory step 1 already freed. Qt's own source comments on
/// exactly this hazard — "the surface is managed by the QPlatformWindow which may
/// be gone already when the unexpose comes" — and believes listening for the
/// surface event fixes it. On Wayland it does not, because the surface dies one
/// step earlier than Qt assumes.
///
/// Detaching here, while the window is still whole, is what makes the teardown
/// finish: Qt calls `releaseSwapChainResources()` **and** `releaseResources()`,
/// the process exits 0, and Valgrind reports no invalid read, write or free.
///
/// **What it costs, stated plainly: the `VkSurfaceKHR` is never destroyed.** The
/// platform window destroys that surface through the instance it can reach from
/// the window, and a detached window offers it none. That leak is the mechanism
/// rather than a side effect — the surface destruction Qt would have performed is
/// exactly the invalid one, since the `wl_surface` beneath it is already gone.
/// The validation layer sees the leak as `VUID-vkDestroyInstance-instance-00629`
/// at shutdown, which is the honest trade: one leaked handle in a process that is
/// exiting, instead of a segfault in a process that is exiting.
///
/// **Hence the platform test.** On xcb the surface is destroyed correctly and
/// there is nothing to work around, so detaching there would trade a clean
/// teardown for a leaked handle and buy nothing. This is not defensive coding:
/// applying it everywhere is what made CI's Vulkan gate report 00629 on X11.
///
/// Honesty about how well the Wayland half is understood: the effect is measured,
/// not derived. A bare `QVulkanWindow` crashes 24 of 24 runs without this and
/// survives 24 of 24 with it, on the NVIDIA driver and on lavapipe — two
/// implementations sharing no code, so not a timing coincidence.
/// `app/tests/wayland_teardown.cpp` measures both halves so a future Qt cannot
/// quietly invalidate either.
///
/// `Close` is the right moment because every way out of this game reaches it —
/// the menu's EXIT, the compositor's close button, a window manager asking — and
/// it is delivered immediately before `destroy()`, so nothing renders after it.
bool GameWindow::event(QEvent* event) {
    if (event->type() == QEvent::Close && QGuiApplication::platformName() == "wayland") {
        setVulkanInstance(nullptr);
    }
    return QVulkanWindow::event(event);
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
    case State::Watch:
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
    case State::Match:
        // One transport for both sides — the same keys a single replay answers
        // to, so watching two is not a second thing to learn.
        if (!match_.has_value()) {
            break; // the state and the match are set together; belt and braces
        }
        if (key == Qt::Key_Escape) {
            match_.reset();
            speed_ = 1;
            open_menu();
        } else if (key == Qt::Key_Space || key == Qt::Key_P) {
            match_paused_ = !match_paused_;
            started_ = false;
        } else if (key == Qt::Key_R) {
            match_->restart();
            match_paused_ = false;
            accumulator_ = 0.0;
            started_ = false;
        } else if (key == Qt::Key_Left) {
            scrub_match(-5);
        } else if (key == Qt::Key_Right) {
            scrub_match(5);
        } else if (key == Qt::Key_W) {
            // Live, and from wherever the match has reached: the leading side
            // is already ahead, so switching sync on simply makes it wait at
            // the next threshold rather than needing a restart to take effect.
            match_wave_sync_ = !match_wave_sync_;
            match_->set_wave_sync(match_wave_sync_);
            save_settings();
        } else if (key == Qt::Key_BracketRight || key == Qt::Key_Plus || key == Qt::Key_Equal) {
            speed_ = std::min(speed_ * 2, 8);
        } else if (key == Qt::Key_BracketLeft || key == Qt::Key_Minus) {
            speed_ = std::max(speed_ / 2, 1);
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
            // Take over mid-game. The sim is a value and neither driver holds
            // state that outlives a decision, so switching the action source is
            // all it takes — the run continues from here under new management.
            ai_driving_ = false;
            watch_driver_.reset();
            driver_name_.clear();
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
                scroll_replays_into_view();
            } else if (key == Qt::Key_Down || key == Qt::Key_S) {
                menu_index_ = (menu_index_ + 1) % replay_count();
                scroll_replays_into_view();
            } else if (confirm) {
                open_selected();
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
