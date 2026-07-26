// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "game_window.hpp"

#include "projection.hpp"
#include "renderer.hpp"

#include <QCoreApplication>
#include <QCursor>
#include <QDir>
#include <QFileInfo>
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
    console_ = console::command(
        console::machine_lookup(QCoreApplication::applicationFilePath().toStdString()));
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
std::filesystem::path GameWindow::pretrained_path() {
    const QString here = QCoreApplication::applicationDirPath();
    const std::array<QString, 4> candidates{
        here + "/models/pretrained.mdp",
        here + "/../Resources/models/pretrained.mdp",             // macOS bundle
        here + "/../share/missile-defense/models/pretrained.mdp", // Debian
        here + "/../../../models/pretrained.mdp", // build/<preset>/app -> the checkout
    };
    for (const QString& candidate : candidates) {
        const QString resolved = QDir::cleanPath(candidate);
        if (QFileInfo::exists(resolved)) {
            return std::filesystem::path{resolved.toStdString()};
        }
    }
    return {};
}

// Persisted via QSettings — QGuiApplication's organization/application name (set
// in main) route this to the platform store (registry on Windows, INI elsewhere).
void GameWindow::load_settings() {
    const QSettings settings;
    audio_on_ = settings.value("audio/sfx", audio_on_).toBool();
    music_on_ = settings.value("audio/music", music_on_).toBool();
    fullscreen_ = settings.value("video/fullscreen", fullscreen_).toBool();
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
    settings.setValue("ai/skill", static_cast<int>(ai_skill_));
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

int GameWindow::menu_count() const noexcept {
    // Nine on a full install, eight on a game-only one. TRAIN AI is *absent*
    // rather than disabled: on the game-only package there is no console to
    // enable, so an item explaining that would be advertising a product the
    // person did not install. The layout follows — menu_item_top_y() derives
    // its row step from active_count(), so a ninth item shrinks the list
    // instead of pushing START into the byline.
    //
    // Starting a game *adds* RESUME and removes nothing. WATCH AI, TRAIN AI,
    // REPLAYS and ABOUT used to disappear here, which made the pause menu a
    // different menu wearing the same frame: every item below the first moved,
    // so the muscle memory built in the main menu was wrong exactly when a
    // player was mid-game and least willing to re-read it.
    return (in_progress_ ? 1 : 0) + (can_train() ? 9 : 8);
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
    const std::array<MenuAction, 9> acts{
        MenuAction::NewGame,    MenuAction::WatchAi, MenuAction::TrainAi,
        MenuAction::Replays,    MenuAction::Help,    MenuAction::Options,
        MenuAction::Highscores, MenuAction::About,   MenuAction::Exit};
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
    // Three scripted rungs, the bundled model when there is one, and BACK.
    return static_cast<int>(watch_skills.size()) + (pretrained_.has_value() ? 2 : 1);
}

std::string_view GameWindow::watch_label(int index) const {
    const auto rungs = static_cast<int>(watch_skills.size());
    if (index >= 0 && index < rungs) {
        return watch_skills[static_cast<std::size_t>(index)].label;
    }
    if (pretrained_.has_value() && index == rungs) {
        // The model's own display name, never the filename: a path is not a
        // name. Upper-cased at load, because this font has no lower case.
        return pretrained_label_;
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

void GameWindow::play_selected_replay() {
    if (menu_index_ < 0 || menu_index_ >= replay_count()) {
        return;
    }
    const std::string path = replay_files_[static_cast<std::size_t>(menu_index_)];
    if (!watch_replay(path)) {
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
    sim_.reset(++seed_);
    state_ = State::Playing;
    in_progress_ = true;
    started_ = false;
    accumulator_ = 0.0;
    fire_latch_.clear();
    ai_driving_ = false;
    ai_assisted_ = false;
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
    replay_scroll_ = 0; // a fresh visit starts at the top, however it was left
    state_ = State::Replays;
}

/// Start the training console and stay where we are.
///
/// Detached, and that is the architecture rather than a shortcut: the console
/// outlives the game, a run outlives the console, and neither should be able to
/// take the other down (docs/ROADMAP.md, M8). So the game does not keep the
/// handle, does not wait, and does not report an exit code — from here it is a
/// separate application that happens to have been started from this menu.
///
/// Nothing visible happens in the game itself, which is deliberate: the console
/// is a window of its own and will raise itself. Returning to the menu rather
/// than to a "launching..." screen means a second press simply opens a second
/// console, the same as double-clicking its desktop entry twice.
void GameWindow::open_console() {
    if (!console_.has_value()) {
        return; // no entry is drawn in this case, so this is belt and braces
    }
    QStringList arguments;
    for (std::size_t i = 1; i < console_->argv.size(); ++i) {
        arguments << QString::fromStdString(console_->argv[i]);
    }
    QProcess process;
    process.setProgram(QString::fromStdString(console_->argv.front()));
    process.setArguments(arguments);
    if (!console_->python_path.empty()) {
        // The checkout case: `md` is not installed anywhere this interpreter
        // would find on its own, so the import path has to be handed over.
        QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
        environment.insert("PYTHONPATH", QString::fromStdString(console_->python_path.string()));
        process.setProcessEnvironment(environment);
    }
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
        } else if (pretrained_.has_value() && index == rungs) {
            start_model_game();
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
        open_console();
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
    const bool hide = (state_ == State::Playing);
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
            if (watch_driver_.has_value()) {
                // A learned policy, through the evaluator's own driver — so the
                // agent on screen is the one `md_agent_eval` scores rather than
                // a second implementation that might steer differently.
                action = watch_driver_->act(sim_);
            } else if (ai_driving_) {
                // The agent is just another driver: same Action, same Sim::step,
                // same crosshair and trigger limits a hand is held to.
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
            play_selected_replay();
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
                play_selected_replay();
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
