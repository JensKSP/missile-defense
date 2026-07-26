// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// The lookup that decides whether the menu offers TRAIN AI at all. It is the
// boundary between the two products, so what is tested here is the *search
// order* rather than any one answer: `md.ui.runner.console_executable()` walks
// the same three places in the same sequence, and a disagreement between them is
// either a menu entry that launches nothing or a console nobody can reach.
#include "console.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>

namespace {

/// A machine that exists only in this file: named variables, named executables.
///
/// Injecting both halves is what makes the order assertable at all. The real
/// lookup asks the filesystem, and a test that had to create files in /usr/bin
/// to prove /usr/bin is searched after PATH could not be written.
struct Machine {
    std::map<std::string, std::string> variables;
    std::set<std::string> executables; // absolute paths that "exist"
    std::string path;                  // the PATH variable's directories

    [[nodiscard]] md::console::Lookup lookup(std::filesystem::path root = {}) const {
        md::console::Lookup probe;
        probe.variable = [this](std::string_view name) -> std::string {
            const auto found = variables.find(std::string{name});
            return found == variables.end() ? std::string{} : found->second;
        };
        probe.executable = [this](const std::filesystem::path& candidate) {
            return executables.contains(candidate.generic_string());
        };
        probe.search_path = path;
        probe.checkout_root = std::move(root);
        return probe;
    }
};

constexpr std::string_view checkout = "/home/dev/missile-defense";

} // namespace

TEST_CASE("MD_CONSOLE names the console and nothing else is consulted", "[unit][app][console]") {
    Machine machine;
    machine.variables["MD_CONSOLE"] = "/opt/build/md-console";
    machine.executables = {"/opt/build/md-console", "/usr/bin/md-console"};
    machine.path = "/usr/bin";

    const auto found = md::console::find(machine.lookup());
    REQUIRE(found.has_value());
    CHECK(found->generic_string() == "/opt/build/md-console");
}

TEST_CASE("A named console that does not exist resolves to nothing, not a fallback",
          "[unit][app][console]") {
    // Falling back would start a *different* console than the one that was
    // named, which is worse than not starting one: the flag exists precisely to
    // pin which build is used.
    Machine machine;
    machine.variables["MD_CONSOLE"] = "/opt/build/md-console";
    machine.executables = {"/usr/bin/md-console"};
    machine.path = "/usr/bin";

    CHECK_FALSE(md::console::find(machine.lookup()).has_value());
}

TEST_CASE("PATH is searched before the directories an installer writes to",
          "[unit][app][console]") {
    Machine machine;
    machine.executables = {"/home/dev/.local/bin/md-console", "/usr/bin/md-console"};
    machine.path = "/home/dev/.local/bin:/usr/games";

    const auto found = md::console::find(machine.lookup());
    REQUIRE(found.has_value());
    CHECK(found->generic_string() == "/home/dev/.local/bin/md-console");
}

TEST_CASE("An installed console off PATH is still found in the system directories",
          "[unit][app][console]") {
    // Started from a desktop entry, a session's PATH is not a login shell's.
    Machine machine;
    machine.executables = {"/usr/bin/md-console"};
    machine.path = "/nowhere";

    const auto found = md::console::find(machine.lookup());
    REQUIRE(found.has_value());
    CHECK(found->generic_string() == "/usr/bin/md-console");
}

TEST_CASE("A checkout offers its own console through the interpreter", "[unit][app][console]") {
    // The developer case: no installed launcher, but `python -m md.ui` is right
    // there. Python's lookup answers `sys.executable` here; this side has to go
    // and find an interpreter, which is the one place the two differ in
    // mechanism while agreeing on the answer.
    Machine machine;
    machine.executables = {std::string{checkout} + "/python/md/ui/__main__.py", "/usr/bin/python3"};
    machine.path = "/usr/bin";

    const auto command = md::console::command(machine.lookup(checkout));
    REQUIRE(command.has_value());
    CHECK(command->argv == std::vector<std::string>{"/usr/bin/python3", "-m", "md.ui"});
    CHECK(command->python_path.generic_string() == std::string{checkout} + "/python");
}

TEST_CASE("A game-only install offers no console at all", "[unit][app][console]") {
    // The negative half of the packaging promise, at the level the menu reads
    // it: no launcher, no interpreter, no checkout — so no TRAIN AI entry.
    Machine machine;
    machine.executables = {"/usr/games/missile-defense"};
    machine.path = "/usr/games:/usr/bin";

    CHECK_FALSE(md::console::find(machine.lookup()).has_value());
    CHECK_FALSE(md::console::command(machine.lookup()).has_value());
}

TEST_CASE("A checkout without an interpreter offers nothing", "[unit][app][console]") {
    // `-m md.ui` needs something to run it. Offering the entry anyway would put
    // a menu item on screen that does nothing when chosen.
    Machine machine;
    machine.executables = {std::string{checkout} + "/python/md/ui/__main__.py"};
    machine.path = "/usr/bin";

    CHECK_FALSE(md::console::command(machine.lookup(checkout)).has_value());
}

TEST_CASE("An installed launcher is run directly, with no interpreter and no path",
          "[unit][app][console]") {
    Machine machine;
    machine.executables = {"/usr/bin/md-console"};
    machine.path = "/usr/bin";

    const auto command = md::console::command(machine.lookup());
    REQUIRE(command.has_value());
    CHECK(command->argv == std::vector<std::string>{"/usr/bin/md-console"});
    CHECK(command->python_path.empty());
}
