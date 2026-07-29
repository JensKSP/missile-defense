// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// One window per thing: the identity a launch derives from its command line
// decides whether it becomes a window or a raised twin, so what is tested
// here is the *decision* — which lines engage, which stay out of the way, and
// that two spawns of the same thing derive the same identity. The channel
// itself gets one live round trip, in-process, because "the twin answered and
// the window was asked to raise" is the whole feature and no amount of key
// equality proves it.
#include "instance.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {

std::string absolute(const std::string& path) {
    return std::filesystem::weakly_canonical(path).generic_string();
}

/// An endpoint no other test run — or stray twin — can be sitting on: the
/// key embeds this process's pid, which is exactly how it will never collide
/// with a real game and never dedupe against a parallel test shard.
std::string private_endpoint(const std::string& suffix) {
    const auto pid =
#ifdef _WIN32
        GetCurrentProcessId();
#else
        getpid();
#endif
    return md::instance::machine_endpoint(std::format("test\n{}\n{}", pid, suffix));
}

/// A flag a server thread can raise and a test can wait on, with a deadline —
/// a test that can hang is worse than a test that fails.
class Raised {
  public:
    void set() {
        const std::lock_guard<std::mutex> lock{mutex_};
        count_ += 1;
        signal_.notify_all();
    }

    [[nodiscard]] bool wait(int at_least = 1) {
        std::unique_lock<std::mutex> lock{mutex_};
        return signal_.wait_for(lock, std::chrono::seconds{5}, [&] { return count_ >= at_least; });
    }

    [[nodiscard]] int count() {
        const std::lock_guard<std::mutex> lock{mutex_};
        return count_;
    }

  private:
    std::mutex mutex_;
    std::condition_variable signal_;
    int count_ = 0;
};

} // namespace

TEST_CASE("the bare game is one thing") {
    CHECK(md::instance::key({}) == "game");
}

TEST_CASE("each target is its own thing, spelled the same from any directory") {
    const std::string replay = md::instance::key({"--replay", "runs/e1.mdr"});
    CHECK(replay == "replay\n" + absolute("runs/e1.mdr"));
    CHECK(md::instance::key({"--replay", "runs/e2.mdr"}) != replay);

    // The seed is part of *which episode* a watched model is showing.
    const std::string pinned = md::instance::key({"--watch-model", "m.mdp", "--seed", "7"});
    CHECK(pinned == "watch\n" + absolute("m.mdp") + "\n7");
    CHECK(md::instance::key({"--seed", "7", "--watch-model", "m.mdp"}) == pinned);
    CHECK(md::instance::key({"--watch-model", "m.mdp"}) != pinned);

    CHECK(md::instance::key({"--match", "t/manifest.json"}) ==
          "match\n" + absolute("t/manifest.json"));
    CHECK(md::instance::key({"--match-left", "a.mdr", "--match-right", "b.mdr"}) ==
          "pair\n" + absolute("a.mdr") + "\n" + absolute("b.mdr"));
}

TEST_CASE("automation lines stay out of the machinery") {
    // The e2e harness writes --frames and --silent on every line it spawns,
    // and tools/capture.py writes --silent; a screenshot run that forwarded
    // itself to the game somebody was playing would be a debugging session.
    CHECK(md::instance::key({"--silent"}).empty());
    CHECK(md::instance::key({"--frames", "60"}).empty());
    CHECK(md::instance::key({"--report"}).empty());
    CHECK(md::instance::key({"--until-done"}).empty());
    CHECK(md::instance::key({"--play"}).empty());
    CHECK(md::instance::key({"--watch"}).empty());
    CHECK(md::instance::key({"--watch-scripted", "medium"}).empty());
    CHECK(md::instance::key({"--replay", "e.mdr", "--frames", "60"}).empty());
    // A flag newer than key() must not start deduplicating by accident on
    // the day it is added — unknown means disengage, not "probably fine".
    CHECK(md::instance::key({"--some-future-flag", "x"}).empty());
    // Half a pair is a line main.cpp exits over; nothing to dedupe.
    CHECK(md::instance::key({"--match-left", "a.mdr"}).empty());
    // A value-taking flag without its value is a line main.cpp ignores too.
    CHECK(md::instance::key({"--replay"}).empty());
    CHECK(md::instance::key({"--seed", "7"}).empty());
}

TEST_CASE("endpoints separate users and identities, deterministically") {
    const std::string mine = md::instance::endpoint("game", "jens", "/run/user/1000");
    CHECK(mine == md::instance::endpoint("game", "jens", "/run/user/1000"));
    CHECK(mine != md::instance::endpoint("game", "someone-else", "/run/user/1001"));
    CHECK(mine != md::instance::endpoint("replay\n/tmp/e.mdr", "jens", "/run/user/1000"));
#ifdef _WIN32
    CHECK(mine.starts_with(R"(\\.\pipe\missile-defense-)"));
#else
    CHECK(mine.starts_with("/run/user/1000/missile-defense-"));
    // No runtime dir is the bare-Unix case; /tmp is shared, the digest's user
    // component is what keeps two people's games out of each other's way.
    CHECK(md::instance::endpoint("game", "jens", "").starts_with("/tmp/missile-defense-"));
#endif
}

TEST_CASE("a forwarded activation reaches the twin's callback") {
    const std::string channel = private_endpoint("round-trip");
    CHECK(md::instance::forward(channel) == md::instance::Forward::NoInstance);

    Raised raised;
    md::instance::Server server;
    REQUIRE(server.start(channel));
    server.on_activate([&raised] { raised.set(); });

    CHECK(md::instance::forward(channel) == md::instance::Forward::Raised);
    CHECK(raised.wait());

    // The claim is exclusive: a second server is refused, which is what sends
    // a real duplicate down the forward path instead.
    md::instance::Server usurper;
    CHECK_FALSE(usurper.start(channel));

    server.detach();
    CHECK(md::instance::forward(channel) == md::instance::Forward::NoInstance);
}

TEST_CASE("an activation that beats the callback is delivered, not dropped") {
    // The real window comes up whole seconds after the claim; a person who
    // double-launched during that gap still deserves a raised window.
    const std::string channel = private_endpoint("early-bird");
    Raised raised;
    md::instance::Server server;
    REQUIRE(server.start(channel));
    CHECK(md::instance::forward(channel) == md::instance::Forward::Raised);
    CHECK(raised.count() == 0);
    server.on_activate([&raised] { raised.set(); });
    CHECK(raised.wait());
    server.detach();
}

#ifndef _WIN32
TEST_CASE("a crashed twin's socket is swept, not obeyed") {
    // A corpse: the path exists, nobody listens. A crash cannot be staged in
    // a unit test, but what it leaves behind can — anything on the path that
    // refuses connections looks exactly like a dead socket to the claim.
    // Windows has no such state; its pipes die with their process.
    const std::string channel = private_endpoint("corpse");
    {
        std::ofstream corpse{channel};
    }
    REQUIRE(std::filesystem::exists(channel));

    Raised raised;
    md::instance::Server server;
    REQUIRE(server.start(channel));
    server.on_activate([&raised] { raised.set(); });
    CHECK(md::instance::forward(channel) == md::instance::Forward::Raised);
    CHECK(raised.wait());
    server.detach();
    CHECK_FALSE(std::filesystem::exists(channel)); // detach cleans up after itself
}
#endif
