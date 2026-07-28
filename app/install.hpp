// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// What TRAIN AI does when there is no trainer yet.
///
/// The game ships the trainer's wheel beside itself and installs it on request;
/// it does not ship a Python. That split is the whole design
/// (docs/superpowers/plans/2026-07-28-python-dependency-and-installation.md): a
/// bundled interpreter plus PySide6 is two hundred megabytes on a download that
/// is otherwise a game, and the audience that wants to train an agent can
/// install Python.
///
/// Everything here is a decision *about* the machine rather than an action on
/// it, for the same reason `trainer.hpp` is: the interesting part is which of
/// four answers a given machine gets, and a test of that should not need a
/// window, a network or a package manager. :struct:`Machine` carries what was
/// probed; :func:`decide` is a pure function of it.
namespace md::install {

/// The four answers, and the only four.
enum class Offer : std::uint8_t {
    /// A trainer resolved and is the version this game shipped. Start it.
    Start,
    /// A trainer resolved, but an older one than the wheel beside the game — an
    /// upgrade replaced the game and left the trainer behind. Offer to reinstall
    /// *before* the two disagree about a policy file, which is the same fault
    /// arriving later and looking like a broken model.
    Update,
    /// No trainer, but a wheel beside the game and an interpreter new enough to
    /// take it. This is the one that spawns anything.
    Install,
    /// A wheel, but no interpreter this wheel can be installed into. The message
    /// has to name a version and a place to get it.
    NeedsPython,
    /// No wheel beside the game, so this build cannot install anything itself.
    /// Two machines land here and the answer is the same for both: read the
    /// instructions. On a distribution build the trainer is
    /// `missile-defense-trainer` and apt owns both halves — never offer to pip
    /// into a distribution's interpreter, which is what PEP 668 exists to stop.
    /// On a Windows or macOS build that shipped without a wheel it is
    /// `pip install missile-defense[trainer]`, which the page also covers.
    NeedsPackage,
};

/// An interpreter that was found, and what it said its version was.
struct Interpreter {
    std::filesystem::path path;
    int major = 0;
    int minor = 0;

    /// Whether the shipped wheel will install here.
    ///
    /// 3.12 and not 3.11, even though `requires-python` says 3.11. The wheel is
    /// tagged `cp312-abi3` because that is where CPython's stable ABI starts
    /// (bindings/CMakeLists.txt), so pip refuses it below that and the only
    /// alternative is an sdist, which needs a C++23 compiler. Offering an
    /// install that cannot succeed is worse than saying which version is needed.
    [[nodiscard]] bool usable() const noexcept { return major > 3 || (major == 3 && minor >= 12); }
};

/// Everything the decision needs from the machine it is running on.
struct Machine {
    /// Whether this path is a file that exists.
    std::function<bool(const std::filesystem::path&)> executable;
    /// The wheel this build ships beside itself, or empty when it ships none.
    std::filesystem::path wheel;
    /// Its version, read out of the filename by :func:`wheel_version`.
    std::string wheel_version;
    /// What `trainer.conf` says was installed, or empty when nothing was. Only
    /// meaningful together with `wheel_version`: two empties mean "no opinion",
    /// which is every developer build and every distribution one.
    std::string installed_version;
    /// Every interpreter that was found, in no particular order.
    std::vector<Interpreter> interpreters;
};

/// The newest usable interpreter, or nothing when none is.
///
/// Newest rather than first: a machine with 3.11 and 3.13 on it should get the
/// one that works, and "first on PATH" would hand it whichever the shell happens
/// to resolve. The choice is reported to the user before anything is installed,
/// because a silent pick is impossible to correct.
[[nodiscard]] std::optional<Interpreter> best(const Machine& machine);

/// Which of the four answers this machine gets.
[[nodiscard]] Offer decide(const Machine& machine, bool trainer_found);

/// `script`, wrapped so it runs in a terminal window the user can watch.
///
/// A visible terminal and not a progress bar in the game: this downloads about
/// 150 MB of PySide6, and a terminal gives progress, scrollback and a copyable
/// error for free — none of which the game has, because it deliberately does not
/// link QtWidgets.
///
/// **No script file.** `app/trainer.cpp` already recorded what happens to one:
/// Smart App Control blocks `.cmd` outright on a stock Windows 11. `cmd.exe` is
/// a system binary and everything after it is an argument, so there is nothing
/// for it to block. macOS gets `open -a Terminal`, and elsewhere this is never
/// reached — a distribution build offers `NeedsPackage` instead.
[[nodiscard]] std::vector<std::string> terminal_command(const std::string& script);

/// The interpreters a real machine has, newest first.
///
/// Runs each candidate to ask its version, which is the only way to know: a
/// binary called `python3.12` may be a symlink to anything, and on Windows the
/// `python.exe` in `WindowsApps` is not an interpreter at all but a stub that
/// opens the Microsoft Store.
[[nodiscard]] std::vector<Interpreter> probe_interpreters(const std::string& search_path);

/// The wheel this build shipped beside itself, or empty.
///
/// One directory, not a search: an installer puts it next to the binary, and
/// looking further up would start finding other people's wheels.
[[nodiscard]] std::filesystem::path shipped_wheel(const std::filesystem::path& beside);

/// The version out of a wheel filename, or empty when it is not one.
///
/// `missile_defense-0.1.0-cp312-abi3-win_amd64.whl` -> `0.1.0`. PEP 427 fixes
/// the shape — distribution, version, then the tags — so the second
/// hyphen-separated field is the version by specification rather than by luck.
[[nodiscard]] std::string wheel_version(const std::filesystem::path& wheel);

/// The whole install, as one command line for a shell.
///
/// One line and not two, joined by `&&`, because the record must only be written
/// when pip *succeeded* — and the game cannot tell: it spawns this detached into
/// a terminal and returns to the menu. So the child does it, with the package it
/// has just installed, in the interpreter it was installed into. A record left
/// behind by a failed install would point at an interpreter that exists and
/// cannot import the trainer, which is exactly the "menu entry that launches
/// nothing" this whole lookup is built to avoid.
[[nodiscard]] std::string install_script(const Interpreter& interpreter,
                                         const std::filesystem::path& wheel);

} // namespace md::install
