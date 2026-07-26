// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "game_window.hpp"

#include <QGuiApplication>
#include <QVulkanInstance>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <print>
#include <string_view>

#ifdef Q_OS_MACOS
#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>
#endif

namespace {

#ifdef Q_OS_MACOS
/// Point the Vulkan loader at the MoltenVK the bundle carries, if it carries one.
///
/// macOS has no Vulkan driver of its own, and the loader finds drivers only
/// through ICD manifests in a fixed set of system directories — none of which is
/// inside an .app. So an installed bundle, which ships its own MoltenVK
/// (app/deploy_macos.cmake.in), has to say where it put it or the loader will
/// enumerate nothing and the game dies on the next line.
///
/// A build tree has no bundled driver and this does nothing, leaving the loader's
/// normal search to find whatever Homebrew installed. An explicit setting always
/// wins, so a developer can still aim it at another ICD.
void use_bundled_vulkan_driver() {
    if (!qEnvironmentVariableIsEmpty("VK_DRIVER_FILES") ||
        !qEnvironmentVariableIsEmpty("VK_ICD_FILENAMES")) {
        return;
    }
    const QString manifest = QDir::cleanPath(QCoreApplication::applicationDirPath() +
                                             "/../Resources/vulkan/icd.d/MoltenVK_icd.json");
    if (!QFileInfo::exists(manifest)) {
        return;
    }
    const QByteArray encoded = QFile::encodeName(manifest);
    qputenv("VK_DRIVER_FILES", encoded);  // the current spelling
    qputenv("VK_ICD_FILENAMES", encoded); // what older loaders read
}
#endif

/// The window's state as one word, for `--report`.
std::string_view state_name(md::GameWindow::State state) {
    using State = md::GameWindow::State;
    switch (state) {
    case State::Menu:
        return "menu";
    case State::Playing:
        return "playing";
    case State::GameOver:
        return "game-over";
    case State::Highscores:
        return "highscores";
    case State::Help:
        return "help";
    case State::About:
        return "about";
    case State::Options:
        return "options";
    case State::EnterScore:
        return "enter-score";
    case State::Replays:
        return "replays";
    }
    return "unknown";
}

/// Which driver was at the controls, for `--report`.
std::string_view mode_name(const md::GameWindow& window) {
    if (window.replaying()) {
        return "replay";
    }
    return window.ai_driving() ? "watch" : "play";
}

/// The main menu's labels, as a JSON array.
///
/// The menu is the one part of the game whose *contents* depend on what else is
/// installed beside it — TRAIN AI appears only where a training console was
/// found — so it is the only way an automated check can tell the game-only
/// package from the full one without a screenshot and a pair of eyes.
std::string menu_json(const md::GameWindow& window) {
    std::string items = "[";
    for (int i = 0; i < window.menu_count(); ++i) {
        if (i != 0) {
            items += ',';
        }
        items += '"';
        items += window.menu_label(i);
        items += '"';
    }
    return items + "]";
}

/// One JSON line on stdout describing how the run ended.
///
/// An exit code alone cannot tell "played a game" from "showed a menu for four
/// seconds", so an automated check of the game needs *something* observable
/// besides not crashing (docs/TESTING.md). This is that surface, and it is
/// deliberately the end state rather than a stream: the questions worth asking
/// of a headless run — did it advance, what did it score, how did it end — are
/// all answered by the last frame.
void write_report(const md::GameWindow& window) {
    const md::Sim& sim = window.sim();
    const auto cities = std::ranges::count_if(sim.cities(), &md::City::alive);
    std::println(R"({{"mode":"{}","state":"{}","frames":{},"ticks":{},)"
                 R"("score":{},"wave":{},"cities_left":{},"terminated":{},)"
                 R"("can_train":{},"menu":{}}})",
                 mode_name(window), state_name(window.state()), window.frames(), sim.tick(),
                 sim.score(), sim.wave(), cities, sim.terminated(), window.can_train(),
                 menu_json(window));
}

int run(int argc, char** argv) {
    QGuiApplication app(argc, argv);
    QGuiApplication::setOrganizationName("MissileDefense");
    QGuiApplication::setApplicationName("MissileDefense"); // stable app-data path for highscores

#ifdef Q_OS_MACOS
    use_bundled_vulkan_driver(); // must precede every Vulkan call — see above
#endif

    QVulkanInstance instance;
#ifdef MD_VULKAN_VALIDATION
    instance.setLayers({"VK_LAYER_KHRONOS_validation"}); // dev builds only (opt-in)
#endif
    if (!instance.create()) {
        qFatal("Failed to create Vulkan instance: %d", static_cast<int>(instance.errorCode()));
        return 1;
    }

    md::GameWindow window;
    window.setVulkanInstance(&instance);
    window.resize(1280, 720);
    window.setTitle("Missile Defense");
    bool report = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--play") {
            window.play_now(); // boot straight into a game (skip the menu)
        } else if (arg == "--watch") {
            window.watch_now(); // boot straight into a game the scripted AI plays
        } else if (arg == "--replay" && (i + 1) < argc) {
            // Watch a recorded run — e.g. an episode a training run dropped on disk.
            if (!window.watch_replay(argv[++i])) {
                qWarning("could not read the recording: %s", argv[i]);
            }
        } else if (arg == "--frames" && (i + 1) < argc) {
            window.set_frame_budget(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--until-done") {
            window.set_exit_when_done(true);
        } else if (arg == "--silent") {
            window.set_silent();
        } else if (arg == "--report") {
            report = true;
        }
    }
    if (window.fullscreen()) { // restore the persisted window mode (see QSettings)
        window.showFullScreen();
    } else {
        window.show();
    }

    const int code = QGuiApplication::exec();
    if (report) {
        write_report(window);
    }
    return code;
}

} // namespace

int main(int argc, char** argv) {
    // Reading a recording touches the filesystem, so main can now be reached by an
    // exception; it must not escape (bugprone-exception-escape). fputs, not
    // std::println, because the handler itself must not be able to throw.
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fputs("missile-defense: ", stderr);
        std::fputs(error.what(), stderr);
        std::fputs("\n", stderr);
        return 1;
    } catch (...) {
        std::fputs("missile-defense: unknown error\n", stderr);
        return 1;
    }
}
