// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// Finding the trainer from inside the game.
///
/// **This lookup is the boundary between the two products.**
/// `missile_defense.runs.runner.trainer_executable()` searches the same four
/// places in the same order, and a disagreement between them is either a menu
/// entry that launches nothing or a trainer that is installed and unreachable.
///
/// Four places, most specific first:
///
///   1. `MD_TRAINER` — someone named one, and nothing else is consulted.
///   2. The interpreter this game recorded when it installed the trainer itself.
///   3. `missile-defense-trainer` on `PATH`, then where a distribution puts it.
///   4. A checkout, run through any interpreter that can be found.
///
/// **Two is the one that carries its weight**, and it replaced a search for a
/// payload directory beside the game. Guessing where a trainer ended up cannot
/// be made to work: pip writes its scripts to `~/Library/Python/3.x/bin` on
/// macOS or `%APPDATA%\Python\...\Scripts` on Windows, neither is on `PATH`,
/// and a macOS app launched from the Finder inherits almost no `PATH` at all —
/// launchd gives it `/usr/bin:/bin:/usr/sbin:/sbin`, which contains neither
/// Homebrew nor anything pip touched. So the game stops guessing and writes down
/// what it did: the *interpreter*, not the script, because
/// `<python> -m missile_defense.ui` needs no scripts directory, no `PATH` and no
/// `.exe`-versus-`.cmd` rule. It is the same command case four already uses.
///
/// Nothing here touches Qt, and that is deliberate — the search order is the
/// part worth testing, and a test of it should not need a window, a display or
/// an event loop. :struct:`Lookup` injects the two things that would otherwise
/// make it untestable: reading the environment, and asking whether a file is
/// there. A test that had to write into `/usr/bin` to prove `/usr/bin` is
/// searched *after* `PATH` could not be written at all.
namespace md::trainer {

/// How `Lookup::search_path` is split, as the platform spells it.
///
/// Public because `search_path` is: a caller that supplies the string has to
/// know how it will be read. A test that joined two directories with a colon
/// on Windows built one nonsense directory instead of two, and the four
/// resulting failures looked like the search was broken rather than the fixture.
#ifdef _WIN32
inline constexpr char path_separator = ';';
#else
inline constexpr char path_separator = ':';
#endif

/// What the search appends before asking `Lookup::executable` about a name.
///
/// Public for the same reason: the callback is handed `missile-defense-trainer.exe` on
/// Windows and `missile-defense-trainer` everywhere else, and an implementation that does
/// not expect that answers no to every candidate.
#ifdef _WIN32
inline constexpr std::string_view executable_suffix = ".exe";
#else
inline constexpr std::string_view executable_suffix{};
#endif

/// Everything the search needs from the machine it is running on.
struct Lookup {
    /// An environment variable's value, or empty when it is unset.
    std::function<std::string(std::string_view)> variable;
    /// Whether this path is an executable file that exists.
    std::function<bool(const std::filesystem::path&)> executable;
    /// `PATH`, as the platform spells it. Held separately from `variable`
    /// because it is split rather than read whole.
    std::string search_path;
    /// The checkout this binary was built in, or empty when it was installed.
    std::filesystem::path checkout_root;
    /// The interpreter this game recorded after installing the trainer into it,
    /// or empty when it never has. Read from disk by :func:`machine_lookup` and
    /// held as a value here so the search itself touches no files — a test can
    /// describe a machine that has one without creating it.
    std::filesystem::path recorded_interpreter;
};

/// How to start the trainer: an argv, plus the import path it needs.
struct Command {
    std::vector<std::string> argv;
    /// `PYTHONPATH` for the checkout case, where `missile_defense` is not installed anywhere
    /// the interpreter would find on its own. Empty for an installed launcher,
    /// which carries its own.
    std::filesystem::path python_path;
};

/// Where the recorded interpreter is written, inside the game's data directory.
///
/// A `key=value` text file rather than anything structured: it holds two facts,
/// a person may need to read or delete it to un-wedge an install, and the game
/// must be able to parse it without taking on a dependency it has no other use
/// for.
inline constexpr std::string_view record_file = "trainer.conf";

/// The key under which :data:`record_file` holds the interpreter's path.
inline constexpr std::string_view record_interpreter_key = "interpreter";

/// A :struct:`Lookup` bound to the real machine, given the running binary.
///
/// `own_executable` is where the checkout root comes from: the game is built to
/// `build/<preset>/app/md_app`, so walking up from it until `python/missile_defense/ui`
/// appears finds the checkout without hard-coding how deep the build tree is.
///
/// `data_dir` is where :data:`record_file` is looked for. Passed in rather than
/// asked of `QStandardPaths`, because everything in this file is deliberately
/// Qt-free: the search order is the part worth testing, and a test of it should
/// not need a `QCoreApplication` to exist.
[[nodiscard]] Lookup machine_lookup(const std::filesystem::path& own_executable,
                                    const std::filesystem::path& data_dir);

/// The interpreter recorded in `data_dir`, or empty when there is no record.
///
/// Separate from :func:`machine_lookup` so the installer can write the file and
/// read it back without building a whole `Lookup`.
[[nodiscard]] std::filesystem::path recorded_interpreter(const std::filesystem::path& data_dir);

/// The trainer's executable, or nothing when this install does not have one.
[[nodiscard]] std::optional<std::filesystem::path> find(const Lookup& lookup);

/// The whole command line, or nothing. Adds `-m missile_defense.ui` for the checkout case.
///
/// Split from :func:`find` because the menu only needs to know *whether* there
/// is a trainer to decide whether to offer training, while starting one needs
/// the rest — and only one of the three answers is not self-contained.
[[nodiscard]] std::optional<Command> command(const Lookup& lookup);

} // namespace md::trainer
