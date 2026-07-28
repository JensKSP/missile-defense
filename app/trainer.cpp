// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "trainer.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace md::trainer {

namespace {

/// What the trainer is called once installed. The Debian package and the
/// pyproject entry point agree on the name, so one search finds either.
constexpr std::array<std::string_view, 1> trainer_names{"missile-defense-trainer"};

/// Where an installer leaves it. `/usr/games` is deliberately absent: the
/// trainer is not a game and its Debian package puts it in `/usr/bin`.
constexpr std::array<std::string_view, 2> system_directories{"/usr/bin", "/usr/local/bin"};

/// Anything that could run `-m missile_defense.ui`, most specific first.
///
/// Windows takes the two in the other order, to agree with
/// `packaging/launcher.cmd.in`, which runs plain `python`. That is the same
/// decision made twice, and on a machine where the two names resolve to
/// different interpreters the menu entry and the shipped script would start
/// different trainers.
///
/// `python3` is the riskier name to prefer there. Windows ships app execution
/// aliases for *both* names in `WindowsApps`, and what they lead to depends on
/// how Python was installed — measured on one machine here, both resolved to a
/// real 3.14; on a machine with no Python at all they open the Microsoft Store
/// instead of running anything. Neither can be told apart from an interpreter
/// without executing it, which a search that only asks "is this file there?"
/// must not do, so it takes the name the rest of the packaging already picked.
#ifdef _WIN32
constexpr std::array<std::string_view, 2> interpreter_names{"python", "python3"};
#else
constexpr std::array<std::string_view, 2> interpreter_names{"python3", "python"};
#endif

/// The marker that says a directory is this project's checkout rather than any
/// other. The trainer's own entry module, because that is precisely the thing
/// the checkout fallback would go on to run.
constexpr std::string_view trainer_module = "python/missile_defense/ui/__main__.py";

/// Which of the four answers the search gave, so the caller does not have to
/// re-derive it from the path. An installed `missile-defense-trainer` can perfectly well sit
/// in the same directory as a `python3`, so the path alone does not say — and
/// the two interpreter answers are told apart by nothing else at all.
enum class Origin : std::uint8_t {
    Named,     ///< MD_TRAINER
    Recorded,  ///< the interpreter this game installed the trainer into
    Installed, ///< a launcher on PATH or in a system directory
    Checkout,  ///< an interpreter, to be handed `-m missile_defense.ui`
};

std::string with_suffix(std::string_view name) {
    return std::string{name} + std::string{executable_suffix};
}

/// The first `name` found in `PATH`, mirroring what a shell would resolve.
std::optional<std::filesystem::path> on_search_path(const Lookup& lookup, std::string_view name) {
    const std::string wanted = with_suffix(name);
    std::string_view rest{lookup.search_path};
    while (!rest.empty()) {
        const std::size_t split = rest.find(path_separator);
        const std::string_view directory = rest.substr(0, split);
        rest = split == std::string_view::npos ? std::string_view{} : rest.substr(split + 1);
        if (directory.empty()) {
            continue; // an empty PATH element means "here", which is not searched
        }
        std::filesystem::path candidate = std::filesystem::path{directory} / wanted;
        if (lookup.executable(candidate)) {
            return candidate;
        }
    }
    return std::nullopt;
}

/// The first interpreter on `PATH` that could be handed `-m missile_defense.ui`.
///
/// Shared by the two answers that need one. They differ only in *which*
/// directory goes on the import path, and having one of them find an
/// interpreter the other would not is a difference nobody would ever want.
std::optional<std::filesystem::path> interpreter(const Lookup& lookup) {
    for (const std::string_view name : interpreter_names) {
        if (auto found = on_search_path(lookup, name)) {
            return found;
        }
    }
    return std::nullopt;
}

/// The trainer's executable and how it was found. The order is the contract.
std::optional<std::pair<std::filesystem::path, Origin>> resolve(const Lookup& lookup) {
    // 1. Someone said which one. A path that does not exist is *nothing* rather
    //    than a fallback: falling back would start a different trainer than the
    //    one that was named, and the variable exists to pin exactly that.
    if (const std::string named = lookup.variable("MD_TRAINER"); !named.empty()) {
        const std::filesystem::path candidate{named};
        if (lookup.executable(candidate)) {
            return std::pair{candidate, Origin::Named};
        }
        return std::nullopt;
    }
    // 2. What this game did itself. Checked before `PATH` on purpose: if the
    //    trainer was installed from here, that interpreter is the one it was
    //    installed *into*, and a `missile-defense-trainer` that happens to be on
    //    PATH belongs to some other Python and may not import the package at
    //    all. A record pointing at an interpreter that has since been removed is
    //    skipped rather than fatal — unlike MD_TRAINER, nobody asked for this
    //    one by name, so falling through to the rest is the helpful answer.
    if (!lookup.recorded_interpreter.empty() && lookup.executable(lookup.recorded_interpreter)) {
        return std::pair{lookup.recorded_interpreter, Origin::Recorded};
    }
    // 3. An installed launcher: PATH first, then where a distribution puts it —
    //    a desktop session's PATH is not a login shell's, so both are needed.
    //    This is the Debian case, where apt owns both halves.
    for (const std::string_view name : trainer_names) {
        if (auto found = on_search_path(lookup, name)) {
            return std::pair{*found, Origin::Installed};
        }
    }
    for (const std::string_view directory : system_directories) {
        for (const std::string_view name : trainer_names) {
            std::filesystem::path candidate = std::filesystem::path{directory} / with_suffix(name);
            if (lookup.executable(candidate)) {
                return std::pair{candidate, Origin::Installed};
            }
        }
    }
    // 4. A checkout, run through an interpreter. Without one there is nothing to
    //    run `-m missile_defense.ui` with, and offering the entry anyway would put an item on
    //    screen that does nothing when it is chosen.
    if (lookup.checkout_root.empty() || !lookup.executable(lookup.checkout_root / trainer_module)) {
        return std::nullopt;
    }
    if (auto found = interpreter(lookup)) {
        return std::pair{*found, Origin::Checkout};
    }
    return std::nullopt;
}

} // namespace

std::filesystem::path recorded_interpreter(const std::filesystem::path& data_dir) {
    if (data_dir.empty()) {
        return {};
    }
    std::ifstream record{data_dir / record_file};
    if (!record) {
        return {}; // never installed from here, which is the ordinary case
    }
    // Hand-parsed because the file has two keys and the game links no parser it
    // would otherwise want. Unknown keys and blank lines are skipped rather than
    // rejected: a newer trainer writing a third key must not stop an older game
    // from reading the interpreter out of it.
    std::string line;
    while (std::getline(record, line)) {
        const std::size_t split = line.find('=');
        if (split == std::string::npos) {
            continue;
        }
        if (std::string_view{line}.substr(0, split) == record_interpreter_key) {
            return std::filesystem::path{line.substr(split + 1)};
        }
    }
    return {};
}

Lookup machine_lookup(const std::filesystem::path& own_executable,
                      const std::filesystem::path& data_dir) {
    Lookup lookup;
    lookup.variable = [](std::string_view name) {
        const char* value = std::getenv(std::string{name}.c_str());
        return value == nullptr ? std::string{} : std::string{value};
    };
    // `is_regular_file` rather than `exists`, and the error_code overload rather
    // than the throwing one: an unreadable directory somewhere on PATH is a
    // reason to keep looking, not a reason for the menu to fail to draw.
    lookup.executable = [](const std::filesystem::path& candidate) {
        std::error_code ec;
        return std::filesystem::is_regular_file(candidate, ec);
    };
    lookup.search_path = lookup.variable("PATH");

    // Walk up from the binary looking for the trainer's sources. An installed
    // game finds nothing and stops at the filesystem root; a build tree finds
    // the checkout three or four levels up without this having to know which.
    std::error_code ec;
    const std::filesystem::path own_directory =
        std::filesystem::absolute(own_executable, ec).parent_path();

    lookup.recorded_interpreter = recorded_interpreter(data_dir);

    std::filesystem::path directory = own_directory;
    while (!directory.empty()) {
        std::error_code probe;
        if (std::filesystem::is_regular_file(directory / trainer_module, probe)) {
            lookup.checkout_root = directory;
            break;
        }
        const std::filesystem::path parent = directory.parent_path();
        if (parent == directory) {
            break; // the root is its own parent; nothing above it to try
        }
        directory = parent;
    }
    return lookup;
}

std::optional<std::filesystem::path> find(const Lookup& lookup) {
    if (auto found = resolve(lookup)) {
        return found->first;
    }
    return std::nullopt;
}

std::optional<Command> command(const Lookup& lookup) {
    auto found = resolve(lookup);
    if (!found.has_value()) {
        return std::nullopt;
    }
    if (found->second == Origin::Checkout) {
        return Command{{found->first.string(), "-m", "missile_defense.ui"},
                       lookup.checkout_root / "python"};
    }
    if (found->second == Origin::Recorded) {
        // No import path: pip installed the package *into* this interpreter, so
        // it is already on that interpreter's own `sys.path`. Setting PYTHONPATH
        // here would be guessing at a directory we deliberately stopped
        // guessing at.
        return Command{{found->first.string(), "-m", "missile_defense.ui"}, {}};
    }
    return Command{{found->first.string()}, {}};
}

} // namespace md::trainer
