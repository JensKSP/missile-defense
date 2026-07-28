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
    const std::string script =
        md::install::install_script(python(3, 13), std::filesystem::path{wheel_path});
    CHECK(script.find("-m pip install --user --upgrade") != std::string::npos);
    CHECK(script.find(std::string{wheel_path} + "[trainer]") != std::string::npos);
}

TEST_CASE("The record is written by the install, and only if it succeeded",
          "[unit][app][install]") {
    // Chained with `&&`, and the second half imports the package the first half
    // installed. The game cannot check: it spawns this detached into a terminal
    // and goes back to the menu. A record written optimistically would name an
    // interpreter that exists and cannot import the trainer — a menu entry that
    // launches nothing, which is the failure this lookup exists to prevent.
    const std::string script = md::install::install_script(python(3, 13), wheel_path);
    const std::size_t pip = script.find("pip install");
    const std::size_t chain = script.find("&&");
    const std::size_t remember = script.find("record_interpreter");
    REQUIRE(pip != std::string::npos);
    REQUIRE(chain != std::string::npos);
    REQUIRE(remember != std::string::npos);
    CHECK(pip < chain);
    CHECK(chain < remember);
}

TEST_CASE("A trainer older than the wheel beside the game offers an update",
          "[unit][app][install]") {
    // Caught here rather than by `policy_container_version` when someone opens a
    // model, which is the same fault arriving later and looking like a broken
    // model instead of a stale install.
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.wheel_version = "0.2.0";
    machine.installed_version = "0.1.0";
    machine.interpreters = {python(3, 13)};
    CHECK(md::install::decide(machine, true) == md::install::Offer::Update);
}

TEST_CASE("Matching versions just start, and an unknown one is not a mismatch",
          "[unit][app][install]") {
    md::install::Machine machine;
    machine.wheel = wheel_path;
    machine.wheel_version = "0.2.0";
    machine.installed_version = "0.2.0";
    CHECK(md::install::decide(machine, true) == md::install::Offer::Start);
    // A distribution build has no wheel and a checkout has no record. Neither is
    // out of date; both would be if "different" included "unknown".
    machine.installed_version.clear();
    CHECK(md::install::decide(machine, true) == md::install::Offer::Start);
    machine.wheel_version.clear();
    machine.installed_version = "0.1.0";
    CHECK(md::install::decide(machine, true) == md::install::Offer::Start);
}

TEST_CASE("The version comes out of the wheel filename, where PEP 427 puts it",
          "[unit][app][install]") {
    CHECK(md::install::wheel_version(wheel_path) == "0.1.0");
    CHECK(md::install::wheel_version("missile_defense-2.11.3-cp312-abi3-win_amd64.whl") ==
          "2.11.3");
    CHECK(md::install::wheel_version("something-else-1.0.whl").empty());
    CHECK(md::install::wheel_version("missile_defense-0.1.0.tar.gz").empty());
}

TEST_CASE("Nothing in the spawned command is a script file", "[unit][app][install]") {
    // The rule app/trainer.cpp learned the hard way: Smart App Control blocks
    // `.cmd` and `.bat` outright on a stock Windows 11, so a helper script would
    // be a helper that never runs. Everything here is an argument to a system
    // binary instead.
    const auto command =
        md::install::terminal_command(md::install::install_script(python(3, 13), wheel_path));
    for (const std::string& argument : command) {
        CHECK_FALSE(argument.ends_with(".cmd"));
        CHECK_FALSE(argument.ends_with(".bat"));
        CHECK_FALSE(argument.ends_with(".command"));
        CHECK_FALSE(argument.ends_with(".sh"));
    }
}
