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

/// Finding the training console from inside the game.
///
/// **This lookup is the boundary between the two products.** The game adds its
/// TRAIN AI entry only when it resolves, so on a game-only install — no Python,
/// no `md` package, no `md-console` — it must find nothing and the menu simply
/// does not offer training. `md.ui.runner.console_executable()` searches the
/// same three places in the same order for exactly that reason: a disagreement
/// between them is either a menu entry that launches nothing, or a console that
/// is installed and unreachable.
///
/// Nothing here touches Qt, and that is deliberate — the search order is the
/// part worth testing, and a test of it should not need a window, a display or
/// an event loop. :struct:`Lookup` injects the two things that would otherwise
/// make it untestable: reading the environment, and asking whether a file is
/// there. A test that had to write into `/usr/bin` to prove `/usr/bin` is
/// searched *after* `PATH` could not be written at all.
namespace md::console {

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
};

/// How to start the console: an argv, plus the import path it needs.
struct Command {
    std::vector<std::string> argv;
    /// `PYTHONPATH` for the checkout case, where `md` is not installed anywhere
    /// the interpreter would find on its own. Empty for an installed launcher,
    /// which carries its own.
    std::filesystem::path python_path;
};

/// A :struct:`Lookup` bound to the real machine, given the running binary.
///
/// `own_executable` is where the checkout root comes from: the game is built to
/// `build/<preset>/app/md_app`, so walking up from it until `python/md/ui`
/// appears finds the checkout without hard-coding how deep the build tree is.
[[nodiscard]] Lookup machine_lookup(const std::filesystem::path& own_executable);

/// The console's executable, or nothing when this install does not have one.
[[nodiscard]] std::optional<std::filesystem::path> find(const Lookup& lookup);

/// The whole command line, or nothing. Adds `-m md.ui` for the checkout case.
///
/// Split from :func:`find` because the menu only needs to know *whether* there
/// is a console to decide whether to offer training, while starting one needs
/// the rest — and only one of the three answers is not self-contained.
[[nodiscard]] std::optional<Command> command(const Lookup& lookup);

} // namespace md::console
