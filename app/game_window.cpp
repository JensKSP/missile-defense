#include "game_window.hpp"

#include "projection.hpp"
#include "renderer.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <algorithm>
#include <cmath>

namespace md {

namespace {
constexpr int menu_item_count = 3; // START, HIGHSCORES, EXIT
}

GameWindow::GameWindow() {
    sim_.reset(seed_);
    aim_ = Vec2{sim_.config().world_width * 0.5f, sim_.config().world_height * 0.5f};
}

QVulkanWindowRenderer* GameWindow::createRenderer() {
    // Ownership passes to QVulkanWindow, which deletes the renderer.
    return new Renderer(this);
}

void GameWindow::start_game() {
    sim_.reset(++seed_);
    state_ = State::Playing;
    started_ = false;
    accumulator_ = 0.0;
    pending_ = Action::noop();
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
        pending_ = Action::noop(); // a click fires exactly once
        accumulator_ -= dt;
        if (sim_.terminated()) {
            state_ = State::GameOver;
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
}

void GameWindow::mousePressEvent(QMouseEvent* event) {
    if (state_ != State::Playing) {
        return;
    }
    update_aim(static_cast<float>(event->position().x()),
               static_cast<float>(event->position().y()));
    pending_ = Action::fire(nearest_base_with_ammo(aim_), aim_);
}

void GameWindow::keyPressEvent(QKeyEvent* event) {
    const int key = event->key();
    switch (state_) {
    case State::Menu:
        if (key == Qt::Key_Up || key == Qt::Key_W) {
            menu_index_ = (menu_index_ + menu_item_count - 1) % menu_item_count;
        } else if (key == Qt::Key_Down || key == Qt::Key_S) {
            menu_index_ = (menu_index_ + 1) % menu_item_count;
        } else if (key == Qt::Key_Return || key == Qt::Key_Enter) {
            if (menu_index_ == 0) {
                start_game();
            } else if (menu_index_ == 1) {
                state_ = State::Highscores;
            } else {
                close();
            }
        } else if (key == Qt::Key_Escape) {
            close();
        }
        break;
    case State::Playing:
        if (key == Qt::Key_P || key == Qt::Key_Space) {
            state_ = State::Paused;
        } else if (key == Qt::Key_Escape) {
            state_ = State::Menu;
        }
        break;
    case State::Paused:
        if (key == Qt::Key_P || key == Qt::Key_Space) {
            state_ = State::Playing;
            started_ = false; // resume without a time jump
        } else if (key == Qt::Key_Escape) {
            state_ = State::Menu;
        }
        break;
    case State::GameOver:
    case State::Highscores:
        if (key == Qt::Key_Return || key == Qt::Key_Enter || key == Qt::Key_Escape) {
            state_ = State::Menu;
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
