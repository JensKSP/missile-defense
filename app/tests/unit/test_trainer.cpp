// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// The lookup that decides whether the menu offers TRAIN AI at all. It is the
// boundary between the two products, so what is tested here is the *search
// order* rather than any one answer: `missile_defense.ui.runner.trainer_executable()` walks
// the same four places in the same sequence, and a disagreement between them is
// either a menu entry that launches nothing or a trainer nobody can reach.
#include "trainer.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>

namespace {

/// Join directories the way the platform's PATH does.
///
/// Not a literal `:`. On Windows the separator is `;`, so a hard-coded colon
/// produced a single directory named `a:b` — and the search correctly found
/// nothing in it, which read as four broken tests rather than one broken
/// fixture.
std::string join(std::initializer_list<std::string_view> directories) {
    std::string joined;
    for (const std::string_view directory : directories) {
        if (!joined.empty()) {
            joined += md::trainer::path_separator;
        }
        joined += directory;
    }
    return joined;
}

/// A path as the search will report it, suffix and all.
///
/// `find()` returns what it probed, so on Windows that is `missile-defense-trainer.exe`.
/// The fixtures name plain executables — `join()`'s counterpart on the way out.
/// Compared through `generic_string()`, hence forward slashes here.
std::string exe(std::string_view path) {
    return std::string{path} + std::string{md::trainer::executable_suffix};
}

/// `argv[0]`, spelled the way the search spells it.
///
/// Built with `operator/` and `string()` because that is what `command()` does,
/// so the separator is the *platform's* and not the fixture's: on Windows the
/// real answer is `/usr/bin\python3.exe`, mixed slashes and all, and asserting
/// a tidier spelling only tests that the fixture and the code disagree.
std::string launcher(std::string_view directory, std::string_view name) {
    return (std::filesystem::path{directory} /
            (std::string{name} + std::string{md::trainer::executable_suffix}))
        .string();
}

/// A machine that exists only in this file: named variables, named executables.
///
/// Injecting both halves is what makes the order assertable at all. The real
/// lookup asks the filesystem, and a test that had to create files in /usr/bin
/// to prove /usr/bin is searched after PATH could not be written.
struct Machine {
    std::map<std::string, std::string> variables;
    /// Absolute paths that "exist", spelled *without* the platform's executable
    /// suffix. `lookup()` adds it, so every case below reads the same on every
    /// platform instead of sprouting `.exe` in a dozen string literals.
    std::set<std::string> executables;
    /// The PATH variable's directories, joined by `join()` with the separator
    /// the search actually splits on — `;` on Windows, `:` elsewhere.
    std::string path;
    /// The install directory an installer dropped the trainer's payload into,
    /// beside the game. A member rather than a `lookup()` argument because the
    /// checkout already has that seat and the two are never both the answer.
    std::filesystem::path payload;

    [[nodiscard]] md::trainer::Lookup lookup(std::filesystem::path root = {}) const {
        md::trainer::Lookup probe;
        probe.variable = [this](std::string_view name) -> std::string {
            const auto found = variables.find(std::string{name});
            return found == variables.end() ? std::string{} : found->second;
        };
        probe.executable = [this](const std::filesystem::path& candidate) {
            // The search appends `.exe` on Windows before asking. Strip it back
            // off so the fixtures can name plain `missile-defense-trainer` and still be
            // answered on a platform that looks for `missile-defense-trainer.exe`.
            std::string name = candidate.generic_string();
            constexpr std::string_view suffix = md::trainer::executable_suffix;
            if (!suffix.empty() && name.size() > suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                name.resize(name.size() - suffix.size());
            }
            return executables.contains(name);
        };
        probe.search_path = path;
        probe.checkout_root = std::move(root);
        probe.payload_root = payload;
        return probe;
    }
};

constexpr std::string_view checkout = "/home/dev/missile-defense";

/// Where a Windows installer leaves the game and the trainer's payload together.
/// Spelled with forward slashes so the fixtures compare through
/// `generic_string()` like every other path here; the separator is not what
/// these cases are about.
constexpr std::string_view install_dir = "/c/Program Files/Missile Defense";

} // namespace

TEST_CASE("MD_TRAINER names the trainer and nothing else is consulted", "[unit][app][trainer]") {
    Machine machine;
    machine.variables["MD_TRAINER"] = "/opt/build/missile-defense-trainer";
    machine.executables = {"/opt/build/missile-defense-trainer",
                           "/usr/bin/missile-defense-trainer"};
    machine.path = "/usr/bin";

    const auto found = md::trainer::find(machine.lookup());
    REQUIRE(found.has_value());
    CHECK(found->generic_string() == "/opt/build/missile-defense-trainer");
}

TEST_CASE("A named trainer that does not exist resolves to nothing, not a fallback",
          "[unit][app][trainer]") {
    // Falling back would start a *different* trainer than the one that was
    // named, which is worse than not starting one: the flag exists precisely to
    // pin which build is used.
    Machine machine;
    machine.variables["MD_TRAINER"] = "/opt/build/missile-defense-trainer";
    machine.executables = {"/usr/bin/missile-defense-trainer"};
    machine.path = "/usr/bin";

    CHECK_FALSE(md::trainer::find(machine.lookup()).has_value());
}

TEST_CASE("PATH is searched before the directories an installer writes to",
          "[unit][app][trainer]") {
    Machine machine;
    machine.executables = {"/home/dev/.local/bin/missile-defense-trainer",
                           "/usr/bin/missile-defense-trainer"};
    machine.path = join({"/home/dev/.local/bin", "/usr/games"});

    const auto found = md::trainer::find(machine.lookup());
    REQUIRE(found.has_value());
    CHECK(found->generic_string() == exe("/home/dev/.local/bin/missile-defense-trainer"));
}

TEST_CASE("An installed trainer off PATH is still found in the system directories",
          "[unit][app][trainer]") {
    // Started from a desktop entry, a session's PATH is not a login shell's.
    Machine machine;
    machine.executables = {"/usr/bin/missile-defense-trainer"};
    machine.path = "/nowhere";

    const auto found = md::trainer::find(machine.lookup());
    REQUIRE(found.has_value());
    CHECK(found->generic_string() == exe("/usr/bin/missile-defense-trainer"));
}

TEST_CASE("A checkout offers its own trainer through the interpreter", "[unit][app][trainer]") {
    // The developer case: no installed launcher, but `python -m missile_defense.ui` is right
    // there. Python's lookup answers `sys.executable` here; this side has to go
    // and find an interpreter, which is the one place the two differ in
    // mechanism while agreeing on the answer.
    Machine machine;
    machine.executables = {std::string{checkout} + "/python/missile_defense/ui/__main__.py",
                           "/usr/bin/python3"};
    machine.path = "/usr/bin";

    const auto command = md::trainer::command(machine.lookup(checkout));
    REQUIRE(command.has_value());
    CHECK(command->argv ==
          std::vector<std::string>{launcher("/usr/bin", "python3"), "-m", "missile_defense.ui"});
    CHECK(command->python_path.generic_string() == std::string{checkout} + "/python");
}

TEST_CASE("An installed payload beside the game is run through the interpreter",
          "[unit][app][trainer]") {
    // The Windows case, and the one that was missing: the installer writes
    // `missile_defense\ui\` next to `md_app.exe` and nothing onto PATH, so before this stage
    // existed every Windows install — installer and portable ZIP alike —
    // resolved to nothing and the menu never offered training at all.
    Machine machine;
    machine.payload = install_dir;
    machine.executables = {std::string{install_dir} + "/missile_defense/ui/__main__.py",
                           "/usr/bin/python3"};
    machine.path = "/usr/bin";

    const auto command = md::trainer::command(machine.lookup());
    REQUIRE(command.has_value());
    CHECK(command->argv ==
          std::vector<std::string>{launcher("/usr/bin", "python3"), "-m", "missile_defense.ui"});
    // The payload's own directory, not a `python/` below it: that is what the
    // installed layout looks like, and what launcher.cmd.in sets from `%~dp0`.
    CHECK(command->python_path.generic_string() == std::string{install_dir});
}

TEST_CASE("An installed launcher still wins over the payload beside the game",
          "[unit][app][trainer]") {
    // Someone who pip-installed the package has an `missile-defense-trainer` of their own on
    // PATH. It is the more explicit answer of the two, so the order that put
    // PATH first has to survive the new stage being added underneath it.
    Machine machine;
    machine.payload = install_dir;
    machine.executables = {std::string{install_dir} + "/missile_defense/ui/__main__.py",
                           "/home/dev/.local/bin/missile-defense-trainer", "/usr/bin/python3"};
    machine.path = join({"/home/dev/.local/bin", "/usr/bin"});

    const auto command = md::trainer::command(machine.lookup());
    REQUIRE(command.has_value());
    CHECK(command->argv ==
          std::vector<std::string>{launcher("/home/dev/.local/bin", "missile-defense-trainer")});
    CHECK(command->python_path.empty());
}

TEST_CASE("A payload with no interpreter offers nothing", "[unit][app][trainer]") {
    // The same rule the checkout case keeps, and it matters more here: a
    // Windows install carries the payload whether or not the machine has any
    // Python, so without this the menu would offer training on every game-only
    // install that happened to tick the trainer component.
    Machine machine;
    machine.payload = install_dir;
    machine.executables = {std::string{install_dir} + "/missile_defense/ui/__main__.py"};
    machine.path = "/usr/bin";

    CHECK_FALSE(md::trainer::find(machine.lookup()).has_value());
    CHECK_FALSE(md::trainer::command(machine.lookup()).has_value());
}

TEST_CASE("A game-only install offers no trainer at all", "[unit][app][trainer]") {
    // The negative half of the packaging promise, at the level the menu reads
    // it: no launcher, no interpreter, no checkout — so no TRAIN AI entry.
    Machine machine;
    machine.executables = {"/usr/games/missile-defense"};
    machine.path = join({"/usr/games", "/usr/bin"});

    CHECK_FALSE(md::trainer::find(machine.lookup()).has_value());
    CHECK_FALSE(md::trainer::command(machine.lookup()).has_value());
}

TEST_CASE("A checkout without an interpreter offers nothing", "[unit][app][trainer]") {
    // `-m missile_defense.ui` needs something to run it. Offering the entry anyway would put
    // a menu item on screen that does nothing when chosen.
    Machine machine;
    machine.executables = {std::string{checkout} + "/python/missile_defense/ui/__main__.py"};
    machine.path = "/usr/bin";

    CHECK_FALSE(md::trainer::command(machine.lookup(checkout)).has_value());
}

TEST_CASE("An installed launcher is run directly, with no interpreter and no path",
          "[unit][app][trainer]") {
    Machine machine;
    machine.executables = {"/usr/bin/missile-defense-trainer"};
    machine.path = "/usr/bin";

    const auto command = md::trainer::command(machine.lookup());
    REQUIRE(command.has_value());
    CHECK(command->argv ==
          std::vector<std::string>{launcher("/usr/bin", "missile-defense-trainer")});
    CHECK(command->python_path.empty());
}
