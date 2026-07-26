// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "console.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace md::console {

namespace {

#ifdef _WIN32
constexpr char path_separator = ';';
constexpr std::string_view executable_suffix = ".exe";
#else
constexpr char path_separator = ':';
constexpr std::string_view executable_suffix{};
#endif

/// What the console is called once installed. The Debian package and the
/// pyproject entry point agree on the name, so one search finds either.
constexpr std::array<std::string_view, 1> console_names{"md-console"};

/// Where an installer leaves it. `/usr/games` is deliberately absent: the
/// console is not a game and its Debian package puts it in `/usr/bin`.
constexpr std::array<std::string_view, 2> system_directories{"/usr/bin", "/usr/local/bin"};

/// Anything that could run `-m md.ui` from a checkout, most specific first.
constexpr std::array<std::string_view, 2> interpreter_names{"python3", "python"};

/// The marker that says a directory is this project's checkout rather than any
/// other. The console's own entry module, because that is precisely the thing
/// the checkout fallback would go on to run.
constexpr std::string_view console_module = "python/md/ui/__main__.py";

/// Which of the three answers the search gave, so the caller does not have to
/// re-derive it from the path. An installed `md-console` can perfectly well sit
/// in the same directory as a `python3`, so the path alone does not say.
enum class Origin : std::uint8_t {
    Named,     ///< MD_CONSOLE
    Installed, ///< a launcher on PATH or in a system directory
    Checkout,  ///< an interpreter, to be handed `-m md.ui`
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

/// The console's executable and how it was found. The order is the contract.
std::optional<std::pair<std::filesystem::path, Origin>> resolve(const Lookup& lookup) {
    // 1. Someone said which one. A path that does not exist is *nothing* rather
    //    than a fallback: falling back would start a different console than the
    //    one that was named, and the variable exists to pin exactly that.
    if (const std::string named = lookup.variable("MD_CONSOLE"); !named.empty()) {
        const std::filesystem::path candidate{named};
        if (lookup.executable(candidate)) {
            return std::pair{candidate, Origin::Named};
        }
        return std::nullopt;
    }
    // 2. An installed launcher: PATH first, then where an installer puts it —
    //    a desktop session's PATH is not a login shell's, so both are needed.
    for (const std::string_view name : console_names) {
        if (auto found = on_search_path(lookup, name)) {
            return std::pair{*found, Origin::Installed};
        }
    }
    for (const std::string_view directory : system_directories) {
        for (const std::string_view name : console_names) {
            std::filesystem::path candidate = std::filesystem::path{directory} / with_suffix(name);
            if (lookup.executable(candidate)) {
                return std::pair{candidate, Origin::Installed};
            }
        }
    }
    // 3. A checkout, run through an interpreter. Without one there is nothing to
    //    run `-m md.ui` with, and offering the entry anyway would put an item on
    //    screen that does nothing when it is chosen.
    if (lookup.checkout_root.empty() || !lookup.executable(lookup.checkout_root / console_module)) {
        return std::nullopt;
    }
    for (const std::string_view name : interpreter_names) {
        if (auto found = on_search_path(lookup, name)) {
            return std::pair{*found, Origin::Checkout};
        }
    }
    return std::nullopt;
}

} // namespace

Lookup machine_lookup(const std::filesystem::path& own_executable) {
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

    // Walk up from the binary looking for the console's sources. An installed
    // game finds nothing and stops at the filesystem root; a build tree finds
    // the checkout three or four levels up without this having to know which.
    std::error_code ec;
    std::filesystem::path directory = std::filesystem::absolute(own_executable, ec).parent_path();
    while (!directory.empty()) {
        std::error_code probe;
        if (std::filesystem::is_regular_file(directory / console_module, probe)) {
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
        return Command{{found->first.string(), "-m", "md.ui"}, lookup.checkout_root / "python"};
    }
    return Command{{found->first.string()}, {}};
}

} // namespace md::console
