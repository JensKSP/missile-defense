// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "data_dir.hpp"
#include "game_window.hpp"
#include "instance.hpp"

#include <QGuiApplication>
#include <QVersionNumber>
#include <QVulkanInstance>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#ifdef Q_OS_WIN
// For the "no Vulkan driver" dialog — see report_no_vulkan().
#define NOMINMAX
#include <windows.h>
#endif

#ifdef Q_OS_MACOS
#include <CoreFoundation/CoreFoundation.h> // for the "no Vulkan driver" alert
#include <QByteArray>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>
#endif

namespace {

#ifdef Q_OS_WIN
/// Whether a standard stream already leads somewhere.
///
/// A process started with `> file` or into a pipe inherits a working handle for
/// it whatever subsystem it was linked for. Re-pointing *that* at the console
/// sends the output to the terminal instead of to whoever asked for it, which
/// is not a theory: the first version of this did it unconditionally, and
/// `--report` piped into a reader came back empty every time.
bool stream_already_connected(DWORD stream) {
    const HANDLE handle = GetStdHandle(stream);
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
        return false;
    }
    return GetFileType(handle) != FILE_TYPE_UNKNOWN;
}

/// Print into the terminal that started us, when there is one.
///
/// The game is linked for the GUI subsystem (app/CMakeLists.txt), which is what
/// stops Windows opening a console window behind it. The cost is that a
/// GUI-subsystem process starts with no standard streams at all, so every
/// `--report` line, every `--help` and every error message below would be
/// written into nothing — including the ones an e2e reads back.
///
/// Attaching to the parent's console gets them back for the case that wants
/// them, and does nothing at all when there is no parent console to attach to:
/// launched from Explorer, a shortcut or the installer, this returns
/// immediately. The shell's prompt has already returned by then, so output
/// arrives underneath it — the accepted shape of this on Windows, and still
/// infinitely better than a window that is always there.
void attach_parent_console() {
    if (AttachConsole(ATTACH_PARENT_PROCESS) == 0) {
        return;
    }
    // Only the streams that have nowhere to go. AttachConsole hands the process
    // a console but does not connect the C streams to it, so the ones that were
    // not inherited have to be pointed at it by hand — and the ones that were
    // must be left exactly alone.
    if (!stream_already_connected(STD_OUTPUT_HANDLE)) {
        (void) std::freopen("CONOUT$", "w", stdout);
    }
    if (!stream_already_connected(STD_ERROR_HANDLE)) {
        (void) std::freopen("CONOUT$", "w", stderr);
    }
}
#endif

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

/// Put `message` in front of someone who started this from an icon.
///
/// stderr is the primary channel and always written; this is for the case where
/// nothing is reading it. A GUI process on Windows has no console at all, and a
/// macOS bundle launched from the Finder sends stderr to Console.app, which
/// amounts to the same thing: the game exits, and the person who double-clicked
/// it sees one bounce of the icon and nothing else.
///
/// Neither platform gets a Qt dialog. QMessageBox lives in QtWidgets, which the
/// game deliberately does not link (docs/PACKAGING.md — the game stays a small,
/// Gui-only binary), so each uses what its own system already provides: user32
/// on Windows and CoreFoundation on macOS, both already linked by Qt.
void report_to_the_desktop([[maybe_unused]] const std::string& message) {
#ifdef Q_OS_WIN
    MessageBoxA(nullptr, message.c_str(), "Missile Defense", MB_OK | MB_ICONERROR);
#elifdef Q_OS_MACOS
    // CFUserNotification and not `osascript`: this runs when the app is already
    // failing to start, and spawning a shell to draw an error box adds a second
    // thing that can be missing. Every string has to be released, including on
    // the path where the call itself fails.
    CFStringRef title =
        CFStringCreateWithCString(nullptr, "Missile Defense", kCFStringEncodingUTF8);
    CFStringRef body = CFStringCreateWithCString(nullptr, message.c_str(), kCFStringEncodingUTF8);
    if (title != nullptr && body != nullptr) {
        CFUserNotificationDisplayAlert(0.0, kCFUserNotificationStopAlertLevel, nullptr, nullptr,
                                       nullptr, title, body, nullptr, nullptr, nullptr, nullptr);
    }
    if (title != nullptr) {
        CFRelease(title);
    }
    if (body != nullptr) {
        CFRelease(body);
    }
#endif
}

/// Tell the user the game cannot start, and what would fix it.
///
/// This replaces a `qFatal`, which aborts the process: no window, no message,
/// and on Windows a crash dialog naming a module rather than a cause. A missing
/// or too-old GPU driver is the single likeliest reason a *downloaded* build
/// does not start — the machines a release reaches are exactly the ones nobody
/// tested — and it is entirely fixable by the person in front of it, but only if
/// they are told what is wrong. So: an exit code rather than an abort, an
/// explanation rather than a number, and a desktop alert as well wherever
/// stderr has nowhere to go.
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
    report_to_the_desktop(message);
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
    case State::TrainNotice:
        return "train-notice";
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
                 R"("can_train":{},"driver":"{}","audible":{},"pretrained":{},"models":{},)"
                 R"("replays":{},"menu":{}}})",
                 mode_name(window), state_name(window.state()), window.frames(), sim.tick(),
                 sim.score(), sim.wave(), cities, sim.terminated(), window.can_train(),
                 driver_name(window),
                 // `--silent` is a promise made to whoever is at the machine,
                 // and it was quietly broken for the first few hundred
                 // milliseconds of every run. Reported so a test can hold it.
                 window.audible(), window.has_pretrained(),
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
    // One window per thing (app/instance.hpp): if a twin is already showing
    // what this command line asks for — the bare game, this replay, this
    // model on this seed — hand it the activation and exit instead of opening
    // it again. The probe runs before Qt and before Vulkan, because the
    // duplicate's whole job is to exit and it should cost milliseconds; the
    // *claim* waits until the window exists, below. Disengaged by any
    // automation flag (`key` returns "") and by MD_SINGLE_INSTANCE=0, so the
    // e2e harness and the capture tool never meet it.
    const std::vector<std::string> arguments{argv + 1, argv + argc};
    const char* solo = std::getenv("MD_SINGLE_INSTANCE");
    const std::string identity = solo != nullptr && std::string_view{solo} == "0"
                                     ? std::string{}
                                     : md::instance::key(arguments);
    std::string channel;
    if (!identity.empty()) {
        channel = md::instance::machine_endpoint(identity);
        if (md::instance::forward(channel) == md::instance::Forward::Raised) {
            return 0;
        }
    }

    // No platform is chosen here. The session picks: Wayland where the user runs
    // Wayland, X11 where they run X11, which is what anyone would expect and for
    // a while was not what they got.
    //
    // This used to force xcb on Wayland sessions, because `QVulkanWindow` cannot
    // survive Qt's own window teardown there — QTBUG-123214, still untriaged, and
    // `QWindowPrivate::destroy()` is byte-identical through Qt's dev branch, so no
    // release available today changes it. That fallback cost every Wayland user a
    // tearing window, NVIDIA implementing no implicit sync under XWayland.
    //
    // `GameWindow::event` now steps out of the way of the defect instead; the long
    // note there explains how, and `app/tests/wayland_teardown.cpp` holds both
    // halves of the claim to evidence.
    QGuiApplication app(argc, argv);
    // The organisation name routes QSettings to its platform store and nothing
    // else — a Qt application without one files settings under "Unknown
    // Organization" and an access error. Where *data* lives is `md::app_data_dir`,
    // deliberately independent of both names; app/data_dir.hpp says why.
    QGuiApplication::setOrganizationName("MissileDefense");
    QGuiApplication::setApplicationName("MissileDefense");
    // Before anything reads a file: the high-score table and the trainer record
    // used to live one directory deeper, under a path Qt built from the
    // organisation name as well.
    md::migrate_legacy_data_dir();

#ifdef Q_OS_MACOS
    use_bundled_vulkan_driver(); // must precede every Vulkan call — see above
#endif

    QVulkanInstance instance;
    // Say which Vulkan this program is written against. Left unset, Qt sends
    // `VkApplicationInfo::apiVersion = 0`, and the validation layer then reports
    // every shader module as declaring the `Shader` capability without the
    // VK_VERSION_1_0 that satisfies it — a real complaint about a real omission,
    // however harmless the driver's behaviour happens to be.
    //
    // 1.0 and not something newer: this renderer is instanced quads under an
    // orthographic transform and uses nothing above the base feature set, and
    // asking for more would refuse to start on hardware that can run the game.
    instance.setApiVersion(QVersionNumber(1, 0));
#ifdef MD_VULKAN_VALIDATION
    instance.setLayers({"VK_LAYER_KHRONOS_validation"}); // dev builds only (opt-in)
#endif
    if (!instance.create()) {
        report_no_vulkan(instance.errorCode());
        return 1;
    }

    // Read before the window exists, and not in the loop below with everything
    // else. `GameWindow`'s constructor starts the audio device and then probes
    // the machine for interpreters, which launches processes — so a `--silent`
    // noticed after construction is noticed several hundred milliseconds of
    // music too late. Worse in the loop: `--match` and `--replay` are handled
    // there and load their files first, and the harness writes `--silent` last.
    const bool silent = std::any_of(
        argv + 1, argv + argc, [](const char* arg) { return std::string_view{arg} == "--silent"; });
    md::GameWindow window{silent};

    // Declared after the window on purpose: scope order is the whole safety
    // argument. The server's destructor detaches — joins its thread, clears
    // its callback — and running *before* the window's destructor on every
    // path out of this function, the early `return 2`s included, is what
    // keeps an activation from poking a window that is half torn down.
    md::instance::Server twin;
    if (!channel.empty()) {
        if (!twin.start(channel) &&
            md::instance::forward(channel) == md::instance::Forward::Raised) {
            return 0; // lost the claim to a twin born during our own startup
        }
        twin.on_activate([&window] {
            // On the server's thread here; the window may only be touched on
            // the GUI thread, so the raise is posted, not performed —
            // GameWindow::event answers with raise_to_person(). postEvent is
            // one of the few Qt calls documented thread-safe, and it owns the
            // event from the moment it takes it.
            // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
            QCoreApplication::postEvent(&window, new QEvent(md::GameWindow::activation_event()));
        });
    }
    window.setVulkanInstance(&instance);
    window.resize(1280, 720);
    window.setTitle("Missile Defense");
    bool report = false;
    std::string match_left;
    std::string match_right;
    // Seeds are read in a pass of their own, because `--watch-model` starts the
    // episode as it is parsed: a `--seed` written after it on the command line
    // would otherwise arrive too late to be the seed of the thing being watched.
    for (int i = 1; (i + 1) < argc; ++i) {
        if (std::string_view(argv[i]) == "--seed") {
            window.set_seed(std::strtoull(argv[i + 1], nullptr, 10));
        }
    }
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
        } else if (arg == "--seed" && (i + 1) < argc) {
            ++i; // already applied above, before anything could start a game
        } else if (arg == "--replay" && (i + 1) < argc) {
            // Watch a recorded run — e.g. an episode a training run dropped on disk.
            if (!window.watch_replay(argv[++i])) {
                qWarning("could not read the recording: %s", argv[i]);
            }
        } else if (arg == "--match" && (i + 1) < argc) {
            // Two agents on the same seed, side by side. A manifest, so the
            // scores the tournament measured come with the recordings and the
            // screen can say what it is showing rather than leaving a viewer
            // to assume (docs/API.md, `missile_defense.runs.tournament.write_manifest`).
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
            window.set_silent(); // already applied above; kept so the list is the options
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
#ifdef Q_OS_WIN
    // Before anything can want to print — including the catch blocks below.
    attach_parent_console();
#endif
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
