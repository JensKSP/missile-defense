// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "game_window.hpp"

#include "projection.hpp"
#include "renderer.hpp"

#include <QCursor>
#include <QKeyEvent>
#include <QMouseEvent>
#include <algorithm>
#include <array>
#include <cmath>

namespace md {

GameWindow::GameWindow() {
    sim_.reset(seed_);
    aim_ = Vec2{sim_.config().world_width * 0.5f, sim_.config().world_height * 0.5f};
    highscores_.load();
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

int GameWindow::menu_count() const noexcept {
    return in_progress_ ? 6 : 5;
}

GameWindow::MenuAction GameWindow::action_at(int index) const {
    if (in_progress_) {
        const std::array<MenuAction, 6> acts{MenuAction::Resume,     MenuAction::NewGame,
                                             MenuAction::Help,       MenuAction::Options,
                                             MenuAction::Highscores, MenuAction::Exit};
        return acts[static_cast<std::size_t>(index)];
    }
    const std::array<MenuAction, 5> acts{MenuAction::NewGame, MenuAction::Help, MenuAction::Options,
                                         MenuAction::Highscores, MenuAction::Exit};
    return acts[static_cast<std::size_t>(index)];
}

std::string_view GameWindow::menu_label(int index) const {
    switch (action_at(index)) {
    case MenuAction::Resume:
        return "RESUME";
    case MenuAction::NewGame:
        return in_progress_ ? "NEW GAME" : "START";
    case MenuAction::Help:
        return "HELP";
    case MenuAction::Options:
        return "OPTIONS";
    case MenuAction::Highscores:
        return "HIGHSCORES";
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

float GameWindow::menu_text_px() const noexcept {
    return sim_.config().world_height * 0.015f;
}

float GameWindow::menu_item_top_y(int index) const noexcept {
    // Center the list block vertically between the title and the bottom hint,
    // sized to the item count so it never overlaps them (5-6 menu items or 3
    // options). `first_top` is the top item's top edge.
    const float h = sim_.config().world_height;
    const float spacing = h * 0.09f;
    const float block = static_cast<float>(active_count() - 1) * spacing;
    const float first_top = (h * 0.44f) + (block * 0.5f);
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
    final_score_ = sim_.score() < 0 ? 0 : sim_.score();
    if (highscores_.qualifies(final_score_)) {
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
}

void GameWindow::toggle_audio() {
    audio_on_ = !audio_on_;
    audio_.set_enabled(audio_on_);
}

void GameWindow::toggle_music() {
    music_on_ = !music_on_;
    audio_.set_music_enabled(music_on_);
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
    pending_ = Action::noop();
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
    case MenuAction::Help:
        state_ = State::Help;
        break;
    case MenuAction::Options:
        open_options();
        break;
    case MenuAction::Highscores:
        state_ = State::Highscores;
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
        sim_.step(pending_);
        audio_.handle_events(sim_.events()); // play SFX for this step's events
        pending_ = Action::noop();           // a click fires exactly once
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
        pending_ = Action::fire(nearest_base_with_ammo(aim_), aim_);
        break;
    case State::GameOver:
    case State::Highscores:
    case State::Help:
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
        }
        break;
    case State::GameOver:
    case State::Highscores:
    case State::Help:
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
