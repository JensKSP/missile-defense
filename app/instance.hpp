// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace md::instance {

/// One window per thing.
///
/// Launching what is already open used to open it again: the trainer's Play
/// button and the game's TRAIN AI entry could each stack an arbitrary number of
/// the other program, and a double-clicked desktop icon did the same to the
/// game itself. The desired behaviour — raise the window that is already
/// showing this thing, start one only when none is — cannot be done by the
/// *launcher*, because no portable way exists to bring another process's window
/// forward (Wayland forbids it outright). It can be done by the launched
/// program: a second launch that finds a twin already serving the same identity
/// hands its activation over and exits, and the twin raises itself, which every
/// platform allows.
///
/// The identity is per *target*, not per program: the plain game is one thing,
/// each replay, watched model and match is its own thing. Opening two different
/// recordings side by side therefore still works; opening the same one twice
/// raises the window that has it.
///
/// The channel is a named pipe on Windows and a Unix socket elsewhere, spoken
/// raw rather than through QtNetwork — the game links Qt6::Gui and nothing
/// else of Qt, and a whole module (plus a DLL in every installer) is a lot to
/// pay for one line of traffic between two copies of the same binary.

/// This launch's identity for single-instance purposes, or `""` to stay out of
/// the way entirely.
///
/// `args` is the command line without `argv[0]`. Only the launch shapes a
/// person produces — the bare game, `--replay`, `--watch-model` (with or
/// without `--seed`), `--match`, `--match-left/--match-right` — get a key.
/// Anything else on the line (`--frames`, `--report`, `--silent`, `--play`,
/// a flag this function has never heard of) disengages the machinery: those
/// lines are written by the e2e harness, the capture tool and developers, and
/// a screenshot run that quietly forwarded itself to the game somebody was
/// playing would be a debugging session nobody deserves. Unknown flags
/// disengage for the same reason — a future automation flag must not start
/// deduplicating by accident on the day it is added.
[[nodiscard]] std::string key(const std::vector<std::string>& args);

/// Where a launch with this `key` rendezvouses with its twin, as a full
/// platform address: `\\.\pipe\...` on Windows, a socket path elsewhere.
///
/// `user` and `runtime_dir` are injected so this is testable as itself;
/// `machine_endpoint` fills them from the environment. The user is folded into
/// the digest because the pipe namespace on Windows and `/tmp` on a bare Unix
/// are machine-wide, and two people's games deduplicating against each other
/// would be both wrong and a question of who may open whose window.
[[nodiscard]] std::string endpoint(const std::string& key, const std::string& user,
                                   const std::string& runtime_dir);

/// `endpoint` with this machine's answers: USERNAME/USER and XDG_RUNTIME_DIR.
[[nodiscard]] std::string machine_endpoint(const std::string& key);

/// What became of handing this launch to a twin.
enum class Forward : std::uint8_t {
    NoInstance, ///< nobody serving this identity — go ahead and be it
    Raised      ///< a twin answered and will raise its window; exit quietly
};

/// Offer this launch's activation to whoever already serves `endpoint`.
///
/// On Windows this also grants the twin the right to take the foreground
/// (`AllowSetForegroundWindow`): this process was just started by whatever the
/// user is looking at, so it holds the privilege, and the twin — idle in the
/// background — does not. Without the grant the raise degrades to a taskbar
/// flash, which is the OS answering a question nobody asked.
///
/// A twin that exists but does not answer within a couple of seconds is
/// treated as absent: opening a second window beats hanging a launch on a
/// wedged process.
[[nodiscard]] Forward forward(const std::string& endpoint);

/// The serving half: claims `endpoint` and turns each "activate" a future twin
/// sends into one call of the installed callback.
///
/// The callback is installed separately from `start` because the claim wants
/// to happen as early as possible — before Qt, before Vulkan — while the thing
/// an activation must poke (the window) exists only much later. An activation
/// that arrives in between is remembered and delivered when the callback is
/// installed, so a twin launched during our own startup is not lost.
///
/// The callback runs on the server's own thread: whoever installs it marshals
/// to their event loop themselves (main.cpp posts it with
/// `QMetaObject::invokeMethod`). `detach()` clears the callback and stops the
/// thread, and must run before whatever the callback touches is destroyed;
/// the destructor calls it too.
class Server {
  public:
    Server() = default;
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    Server(Server&&) = delete;
    Server& operator=(Server&&) = delete;

    /// Claim `endpoint` and start answering. False when a live twin already
    /// holds it (the caller should `forward` instead); a socket file a crashed
    /// twin left behind is swept up rather than treated as a twin.
    [[nodiscard]] bool start(const std::string& endpoint);

    /// Install (or replace) what an activation does. Delivers immediately, on
    /// the caller's thread, if one arrived before there was a callback.
    void on_activate(std::function<void()> callback);

    /// Stop answering: clear the callback, unblock and join the thread, close
    /// and (on Unix) unlink the channel. Idempotent, and safe to call without
    /// a successful `start`.
    void detach();

  private:
    void serve();
    void notify();

    std::mutex mutex_;
    std::function<void()> callback_;
    bool pending_ = false;
    std::atomic<bool> stop_{false};
    std::atomic<bool> done_{false};
    std::thread thread_;
    std::string endpoint_;
#ifdef _WIN32
    void* first_pipe_ = nullptr; ///< HANDLE, spelled without windows.h
#else
    int listen_fd_ = -1;
#endif
};

} // namespace md::instance
