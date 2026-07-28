// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// Which of four answers a machine gets when TRAIN AI is pressed. The decision is
// a pure function of what was probed, so every case here is a machine described
// in four lines rather than a machine that has to exist.
#include "install.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace {

md::install::Interpreter python(int major, int minor, std::string_view path = "/usr/bin/python3") {
    return md::install::Interpreter{std::filesystem::path{path}, major, minor};
}

constexpr std::string_view wheel_path =
    "/opt/missile-defense/missile_defense-0.1.0-cp312-abi3-linux_x86_64.whl";

} // namespace

TEST_CASE("A resolved trainer is started, whatever else is on the machine",
          "[unit][app][install]") {
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.interpreters = {python(3, 13)};
    CHECK(md::install::decide(machine, true) == md::install::Offer::Start);
}

TEST_CASE("A wheel and a new enough Python is the one case that installs", "[unit][app][install]") {
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.interpreters = {python(3, 12)};
    CHECK(md::install::decide(machine, false) == md::install::Offer::Install);
}

TEST_CASE("Python 3.11 is not enough, because the wheel is cp312-abi3", "[unit][app][install]") {
    // `requires-python` says 3.11 and that is true of the sdist. The shipped
    // wheel is tagged for the stable ABI, which starts at 3.12, so pip refuses
    // it below that — and offering an install that cannot succeed is worse than
    // naming the version that would work.
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.interpreters = {python(3, 11)};
    CHECK(md::install::decide(machine, false) == md::install::Offer::NeedsPython);
}

TEST_CASE("No Python at all asks for one rather than for a package", "[unit][app][install]") {
    md::install::Machine machine;
    machine.wheel = wheel_path;
    CHECK(md::install::decide(machine, false) == md::install::Offer::NeedsPython);
}

TEST_CASE("A build that ships no wheel points at the distribution's package",
          "[unit][app][install]") {
    // The Debian case. Offering pip here would be offering PEP 668, which
    // refuses by design and is right to: the interpreter belongs to apt, and so
    // does the trainer.
    md::install::Machine machine;
    machine.interpreters = {python(3, 13)};
    CHECK(md::install::decide(machine, false) == md::install::Offer::NeedsPackage);
}

TEST_CASE("The newest usable interpreter wins, not the first one found", "[unit][app][install]") {
    // A machine with several Pythons is the ordinary case, and "first on PATH"
    // would hand it whichever the shell resolves — which is usually the
    // distribution's, and usually the oldest.
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.interpreters = {python(3, 12, "/usr/bin/python3"), python(3, 14, "/opt/py/bin/python3"),
                            python(3, 13, "/usr/local/bin/python3")};

    const auto chosen = md::install::best(machine);
    REQUIRE(chosen.has_value());
    CHECK(chosen->minor == 14);
    CHECK(chosen->path.generic_string() == "/opt/py/bin/python3");
}

TEST_CASE("An unusable interpreter is never chosen, however new it looks", "[unit][app][install]") {
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.interpreters = {python(2, 7), python(3, 11)};
    CHECK_FALSE(md::install::best(machine).has_value());
}

TEST_CASE("The install command is a user install of the local wheel with its extra",
          "[unit][app][install]") {
    // `--user`, so no administrator and no write into the install directory. The
    // extra rides on the path because PySide6 is optional to the package and
    // required by the window, and the wheel is local because the game and the
    // trainer share a policy version — a trainer from PyPI could be a different
    // one (see `policy_container_version`).
    const auto command = md::install::pip_command(python(3, 13), std::filesystem::path{wheel_path});
    REQUIRE(command.size() == 7);
    CHECK(command[0] == "/usr/bin/python3");
    CHECK(command[1] == "-m");
    CHECK(command[2] == "pip");
    CHECK(command[3] == "install");
    CHECK(command[4] == "--user");
    CHECK(command.back() == std::string{wheel_path} + "[trainer]");
}

TEST_CASE("Nothing in the spawned command is a script file", "[unit][app][install]") {
    // The rule app/trainer.cpp learned the hard way: Smart App Control blocks
    // `.cmd` and `.bat` outright on a stock Windows 11, so a helper script would
    // be a helper that never runs. Everything here is an argument to a system
    // binary instead.
    const auto command =
        md::install::terminal_command(md::install::pip_command(python(3, 13), wheel_path));
    for (const std::string& argument : command) {
        CHECK_FALSE(argument.ends_with(".cmd"));
        CHECK_FALSE(argument.ends_with(".bat"));
        CHECK_FALSE(argument.ends_with(".command"));
        CHECK_FALSE(argument.ends_with(".sh"));
    }
}
