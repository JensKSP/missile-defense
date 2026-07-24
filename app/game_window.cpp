#include "game_window.hpp"

#include "projection.hpp"
#include "renderer.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <algorithm>
#include <array>
#include <cmath>

namespace md {

GameWindow::GameWindow() {
    sim_.reset(seed_);
    aim_ = Vec2{sim_.config().world_width * 0.5f, sim_.config().world_height * 0.5f};
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

int GameWindow::menu_count() const noexcept {
    return in_progress_ ? 5 : 4;
}

GameWindow::MenuAction GameWindow::action_at(int index) const {
    if (in_progress_) {
        const std::array<MenuAction, 5> acts{MenuAction::Resume, MenuAction::NewGame,
                                             MenuAction::Help, MenuAction::Highscores,
                                             MenuAction::Exit};
        return acts[static_cast<std::size_t>(index)];
    }
    const std::array<MenuAction, 4> acts{MenuAction::NewGame, MenuAction::Help,
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
    case MenuAction::Highscores:
        return "HIGHSCORES";
    case MenuAction::Exit:
        return "EXIT";
    }
    return "";
}

float GameWindow::menu_text_px() const noexcept {
    return sim_.config().world_height * 0.017f;
}

float GameWindow::menu_item_top_y(int index) const noexcept {
    return sim_.config().world_height * (0.60f - (static_cast<float>(index) * 0.11f));
}

int GameWindow::menu_hit(Vec2 world) const noexcept {
    const float px = menu_text_px();
    const float advance = px * 4.0f; // per-glyph horizontal step (matches draw_text)
    const float center_x = sim_.config().world_width * 0.5f;
    const float pad_x = advance * 0.5f;
    const float pad_y = px * 0.8f;
    for (int i = 0; i < menu_count(); ++i) {
        const float top_y = menu_item_top_y(i);
        const float bottom_y = top_y - (5.0f * px); // glyphs span ~5 rows below top_y
        const auto chars = static_cast<float>(menu_label(i).size());
        const float half_w = (chars * advance * 0.5f) + pad_x;
        if (std::abs(world.x - center_x) <= half_w && world.y <= (top_y + pad_y) &&
            world.y >= (bottom_y - pad_y)) {
            return i;
        }
    }
    return -1;
}

void GameWindow::open_menu() {
    state_ = State::Menu;
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

void GameWindow::select_menu() {
    switch (action_at(menu_index_)) {
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
    case MenuAction::Highscores:
        state_ = State::Highscores;
        break;
    case MenuAction::Exit:
        close();
        break;
    }
}

void GameWindow::advance() {
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

    const double dt = static_cast<double>(sim_.config().dt);
    while (accumulator_ >= dt) {
        sim_.step(pending_);
        audio_.handle_events(sim_.events()); // play SFX for this step's events
        pending_ = Action::noop();           // a click fires exactly once
        accumulator_ -= dt;
        if (sim_.terminated()) {
            state_ = State::GameOver;
            in_progress_ = false;
            accumulator_ = 0.0;
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
    if (state_ == State::Menu) {
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
    case State::Menu: {
        const int hit = menu_hit(aim_);
        if (hit >= 0) {
            menu_index_ = hit;
            select_menu(); // click an item to activate it
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
    }
}

void GameWindow::keyPressEvent(QKeyEvent* event) {
    const int key = event->key();
    switch (state_) {
    case State::Menu:
        if (key == Qt::Key_Up || key == Qt::Key_W) {
            menu_index_ = (menu_index_ + menu_count() - 1) % menu_count();
        } else if (key == Qt::Key_Down || key == Qt::Key_S) {
            menu_index_ = (menu_index_ + 1) % menu_count();
        } else if (key == Qt::Key_Return || key == Qt::Key_Enter) {
            select_menu();
        } else if (key == Qt::Key_Escape && in_progress_) {
            state_ = State::Playing; // Escape resumes the paused game
            started_ = false;
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
