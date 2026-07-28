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
/// **This lookup is the boundary between the two products.** The game adds its
/// TRAIN AI entry only when it resolves, so on a game-only install — no Python,
/// no `missile_defense` package, no `missile-defense-trainer` — it must find nothing and the
/// menu simply does not offer training. `missile_defense.ui.runner.trainer_executable()`
/// searches the same four places in the same order for exactly that reason: a
/// disagreement between them is either a menu entry that launches nothing, or a
/// trainer that is installed and unreachable.
///
/// The fourth place is the Windows one, and it was missing until 2026-07-27:
/// there, the trainer's payload is installed *beside the game* — `missile_defense/ui/` next
/// to `missile-defense.exe`, under `C:\Program Files\Missile Defense` or
/// wherever the portable ZIP was unpacked. That directory is on nobody's `PATH`,
/// and the two system directories searched before it are Unix paths that cannot
/// exist. So every Windows install, installer and ZIP alike, resolved to nothing
/// and offered no way into training at all — while
/// `missile-defense-trainer.cmd` sat unreachable in the same folder as the
/// binary that could not find it.
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
    /// The directory holding the trainer's Python payload an installer left
    /// beside the game (`missile_defense/ui/__main__.py` in it), or empty when there is
    /// none. Distinct from `checkout_root`, which holds the *sources* one level
    /// further down in `python/`; these are two different layouts and looking
    /// for one in the other finds nothing.
    std::filesystem::path payload_root;
};

/// How to start the trainer: an argv, plus the import path it needs.
struct Command {
    std::vector<std::string> argv;
    /// `PYTHONPATH` for the checkout case, where `missile_defense` is not installed anywhere
    /// the interpreter would find on its own. Empty for an installed launcher,
    /// which carries its own.
    std::filesystem::path python_path;
};

/// A :struct:`Lookup` bound to the real machine, given the running binary.
///
/// `own_executable` is where the checkout root comes from: the game is built to
/// `build/<preset>/app/md_app`, so walking up from it until `python/missile_defense/ui`
/// appears finds the checkout without hard-coding how deep the build tree is.
[[nodiscard]] Lookup machine_lookup(const std::filesystem::path& own_executable);

/// The trainer's executable, or nothing when this install does not have one.
[[nodiscard]] std::optional<std::filesystem::path> find(const Lookup& lookup);

/// The whole command line, or nothing. Adds `-m missile_defense.ui` for the checkout case.
///
/// Split from :func:`find` because the menu only needs to know *whether* there
/// is a trainer to decide whether to offer training, while starting one needs
/// the rest — and only one of the three answers is not self-contained.
[[nodiscard]] std::optional<Command> command(const Lookup& lookup);

} // namespace md::trainer
