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
#include <format>
#include <print>
#include <string>
#include <string_view>

#ifdef Q_OS_WIN
// For the "no Vulkan driver" dialog — see report_no_vulkan().
#define NOMINMAX
#include <windows.h>
#endif

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

/// Why the Vulkan instance would not open, in words the person can act on.
///
/// `VkResult` is a negative integer, and a negative integer is not a diagnosis.
/// These three cover essentially every real failure: no driver at all, a loader
/// that found one it cannot talk to, and a machine too short of memory to try.
std::string_view explain_vulkan_failure(VkResult code) {
    switch (code) {
    case VK_SUCCESS:
        // Qt reports no error at all when it gave up *before* calling the
        // driver — which is what a platform plugin with no Vulkan support does
        // (`offscreen`, `vnc`, a headless session). A different fault with a
        // different fix, and the one a developer meets rather than a player.
        return "the Qt platform plugin in use offers no Vulkan surface";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "no compatible Vulkan driver was found";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "the Vulkan driver was found but could not be initialised";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "there was not enough memory to create a Vulkan instance";
    default:
        return "the Vulkan loader refused to create an instance";
    }
}

/// Tell the user the game cannot start, and what would fix it.
///
/// This replaces a `qFatal`, which aborts the process: no window, no message,
/// and on Windows a crash dialog naming a module rather than a cause. A missing
/// or too-old GPU driver is the single likeliest reason a *downloaded* build
/// does not start — the machines a release reaches are exactly the ones nobody
/// tested — and it is entirely fixable by the person in front of it, but only if
/// they are told what is wrong. So: an exit code rather than an abort, an
/// explanation rather than a number, and on Windows a dialog as well, because a
/// GUI process there has no console for stderr to arrive in.
void report_no_vulkan(VkResult code) {
    // The numeric code is worth carrying for a bug report but not for a person,
    // so it is a suffix — and omitted entirely when it is VK_SUCCESS, where
    // printing "VkResult 0" beside a failure would just look like a second bug.
    const std::string detail =
        code == VK_SUCCESS ? std::string{} : std::format(" (VkResult {})", static_cast<int>(code));
    const std::string message =
        std::format("Missile Defense needs Vulkan to draw, and {}{}.\n"
                    "\n"
                    "This is almost always the graphics driver:\n"
                    "  Windows  install the latest driver from AMD, Intel or NVIDIA\n"
                    "  Linux    install your vendor's driver, or mesa-vulkan-drivers\n"
                    "           for software rendering (Debian/Ubuntu:\n"
                    "           sudo apt install mesa-vulkan-drivers vulkan-tools)\n"
                    "  macOS    the bundle ships MoltenVK; a build run from a source\n"
                    "           tree needs it installed (brew install molten-vk)\n"
                    "\n"
                    "`vulkaninfo` from vulkan-tools reports what this machine can see.\n"
                    "If QT_QPA_PLATFORM is set, unset it: offscreen and vnc have no Vulkan.",
                    explain_vulkan_failure(code), detail);
    std::fputs(message.c_str(), stderr);
    std::fputs("\n", stderr);
#ifdef Q_OS_WIN
    // stderr from a Windows GUI subsystem process goes nowhere at all, so
    // without this the user who double-clicked the icon gets silence. user32 is
    // already linked by Qt, so this costs no dependency — and QMessageBox is not
    // an option: it lives in QtWidgets, which the game deliberately does not
    // link (docs/PACKAGING.md — the game stays a small, Gui-only binary).
    MessageBoxA(nullptr, message.c_str(), "Missile Defense", MB_OK | MB_ICONERROR);
#endif
}

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
    case State::Watch:
        return "watch-menu";
    case State::Match:
        return "match";
    }
    return "unknown";
}

/// Which driver was at the controls, for `--report`.
std::string_view mode_name(const md::GameWindow& window) {
    if (window.match() != nullptr) {
        return "match";
    }
    if (window.replaying()) {
        return "replay";
    }
    return window.ai_driving() ? "watch" : "play";
}

/// *Which* agent, for `--report`. The same string the HUD shows, so an e2e can
/// assert who was playing instead of a human squinting at a screenshot — which
/// is the whole reason Step 4b put the name in a machine-readable place too.
std::string_view driver_name(const md::GameWindow& window) {
    return window.driver_name();
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
                 R"("can_train":{},"driver":"{}","pretrained":{},"models":{},"replays":{},)"
                 R"("menu":{}}})",
                 mode_name(window), state_name(window.state()), window.frames(), sim.tick(),
                 sim.score(), sim.wave(), cities, sim.terminated(), window.can_train(),
                 driver_name(window), window.has_pretrained(),
                 // How many models this install can actually *run*, which is the
                 // only version of the question worth reporting: a promoted
                 // model that the game silently will not offer is the failure a
                 // packaging test has to be able to see.
                 md::GameWindow::installed_model_count(),
                 // Discovery is the whole of Workstream E: the browser used to
                 // look one directory too high and came up empty for anyone who
                 // had ever used the console. A count is how a test sees that.
                 md::GameWindow::discovered_recording_count(), menu_json(window));
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
        report_no_vulkan(instance.errorCode());
        return 1;
    }

    md::GameWindow window;
    window.setVulkanInstance(&instance);
    window.resize(1280, 720);
    window.setTitle("Missile Defense");
    bool report = false;
    std::string match_left;
    std::string match_right;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--play") {
            window.play_now(); // boot straight into a game (skip the menu)
        } else if (arg == "--watch" || arg == "--watch-scripted") {
            // An optional rung: `--watch-scripted medium`. Bare means the
            // published baseline, so `--watch` keeps meaning exactly what it did.
            if ((i + 1) < argc) {
                if (const auto skill = md::GameWindow::skill_named(argv[i + 1])) {
                    ++i;
                    window.watch_now(*skill);
                    continue;
                }
            }
            window.watch_now(); // boot straight into a game the scripted AI plays
        } else if (arg == "--watch-model" && (i + 1) < argc) {
            // Watch a learned policy. `--watch-scripted` is its twin, spelled
            // out so a packaging test can name both agents explicitly rather
            // than relying on which one `--watch` happens to mean.
            if (!window.watch_model(argv[++i])) {
                return 2; // the reason is already on stderr; do not open a window
            }
        } else if (arg == "--replay" && (i + 1) < argc) {
            // Watch a recorded run — e.g. an episode a training run dropped on disk.
            if (!window.watch_replay(argv[++i])) {
                qWarning("could not read the recording: %s", argv[i]);
            }
        } else if (arg == "--match" && (i + 1) < argc) {
            // Two agents on the same seed, side by side. A manifest, so the
            // scores the tournament measured come with the recordings and the
            // screen can say what it is showing rather than leaving a viewer
            // to assume (docs/API.md, `md.tournament.write_manifest`).
            if (!window.watch_match(std::string{argv[++i]})) {
                return 2; // the reason is already on stderr; do not open a window
            }
        } else if (arg == "--match-left" && (i + 1) < argc) {
            match_left = argv[++i];
        } else if (arg == "--match-right" && (i + 1) < argc) {
            match_right = argv[++i];
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
    // Resolved after the loop: the two halves are one option, and requiring
    // them in a fixed order would be an arbitrary rule to remember.
    if (!match_left.empty() || !match_right.empty()) {
        if (match_left.empty() || match_right.empty()) {
            std::println(stderr, "md_app: --match-left and --match-right go together; "
                                 "a match needs both sides");
            return 2;
        }
        if (!window.watch_match(match_left, match_right)) {
            return 2; // the reason is already on stderr
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
