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
    /// A trainer resolved. Nothing to install; start it.
    Start,
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

/// `pip install --user <wheel>[trainer]`, as an argv.
///
/// `--user`, so no administrator and no write into the install directory. PEP
/// 668 would refuse this on a distribution interpreter, which is exactly the
/// case :enum:`Offer::NeedsPackage` never reaches.
///
/// The extra is part of the path argument — `pip install "<path>.whl[trainer]"`
/// — because PySide6 is optional to the package and required by the window.
[[nodiscard]] std::vector<std::string> pip_command(const Interpreter& interpreter,
                                                   const std::filesystem::path& wheel);

/// `inner`, wrapped so it runs in a terminal window the user can watch.
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
[[nodiscard]] std::vector<std::string> terminal_command(const std::vector<std::string>& inner);

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

} // namespace md::install
