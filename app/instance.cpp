// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// The wire is one round trip, spoken raw so the game keeps linking only
// Qt6::Gui: the server greets with "md1 <pid>\n", the client grants that pid
// the foreground (Windows) and answers "activate\n", the server raises its
// window. Every read is bounded by a deadline — the peer is by construction
// another copy of this program, but a wedged one must cost a launch two
// seconds, not forever, and must never be able to wedge *this* copy's
// shutdown: `detach()` has to be able to join the thread on every path, which
// is why the loops below poll instead of blocking.
#include "instance.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace md::instance {

namespace {

using namespace std::chrono_literals;

/// How long a peer gets to say its piece before it is treated as absent.
constexpr auto reply_deadline = 2s;
/// How often the bounded reads look again.
constexpr auto poll_step = 25ms;

constexpr std::string_view greeting_prefix = "md1 ";
constexpr std::string_view activate_message = "activate\n";

/// FNV-1a, 64 bit. Not std::hash, whose value for the same string is allowed
/// to differ between the running game and the newer build just launched — and
/// two spellings of one identity is exactly a window that fails to dedupe.
std::uint64_t fnv1a(std::string_view text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

/// A target path as both spawns of the same thing will spell it.
///
/// `weakly_canonical` so `runs/x.mdr` from the checkout and the absolute path
/// a later launcher sends meet in the middle; `generic_string` so the
/// separators cannot disagree. A path the filesystem refuses to resolve is
/// used as written — both twins will fail to resolve it identically, and a
/// deterministic wrong answer still dedupes.
std::string normalised(const std::string& path) {
    std::error_code ec;
    std::filesystem::path canon = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        canon = std::filesystem::absolute(path, ec);
        if (ec) {
            return path;
        }
    }
    return canon.generic_string();
}

std::string variable(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string{value};
}

#ifdef _WIN32

/// One bounded read: whatever arrives within the deadline, or empty.
///
/// PeekNamedPipe under a poll loop rather than a blocking ReadFile, because a
/// synchronous pipe read has no timeout and the serve/stop contract above
/// needs every wait in this file to end on its own.
std::string bounded_read(HANDLE pipe) {
    const auto deadline = std::chrono::steady_clock::now() + reply_deadline;
    while (std::chrono::steady_clock::now() < deadline) {
        DWORD available = 0;
        if (PeekNamedPipe(pipe, nullptr, 0, nullptr, &available, nullptr) == 0) {
            return {}; // the peer went away
        }
        if (available > 0) {
            std::array<char, 128> buffer{};
            DWORD read = 0;
            if (ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr) ==
                0) {
                return {};
            }
            return std::string{buffer.data(), read};
        }
        Sleep(static_cast<DWORD>(std::chrono::milliseconds{poll_step}.count()));
    }
    return {};
}

bool write_all(HANDLE pipe, std::string_view text) {
    DWORD written = 0;
    return WriteFile(pipe, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) != 0 &&
           written == text.size();
}

#else

/// One bounded read from a socket: whatever arrives in time, or empty.
std::string bounded_read(int fd) {
    pollfd waiting{fd, POLLIN, 0};
    const int ready =
        poll(&waiting, 1,
             static_cast<int>(
                 std::chrono::duration_cast<std::chrono::milliseconds>(reply_deadline).count()));
    if (ready <= 0) {
        return {};
    }
    std::array<char, 128> buffer{};
    const ssize_t got = recv(fd, buffer.data(), buffer.size(), 0);
    if (got <= 0) {
        return {};
    }
    return std::string{buffer.data(), static_cast<std::size_t>(got)};
}

/// send() that cannot raise SIGPIPE: the peer closing early is an ordinary
/// event here, and the default disposition would take the whole game down.
bool write_all(int fd, std::string_view text) {
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0; // macOS: SO_NOSIGPIPE is set where the fd is made
#endif
    return send(fd, text.data(), text.size(), flags) == static_cast<ssize_t>(text.size());
}

void quiet_pipes([[maybe_unused]] int fd) {
#ifndef MSG_NOSIGNAL
    const int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
}

/// A connected client socket for `endpoint`, or -1.
int connect_to(const std::string& endpoint) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (endpoint.size() >= sizeof(address.sun_path)) {
        close(fd);
        return -1;
    }
    std::memcpy(address.sun_path, endpoint.c_str(), endpoint.size() + 1);
    if (connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close(fd);
        return -1;
    }
    quiet_pipes(fd);
    return fd;
}

#endif

/// The pid out of "md1 <pid>\n", or 0 when the greeting is not ours.
///
/// A stranger on this endpoint is possible — the name is derived, not
/// reserved — and answering a foreign protocol with "activate" is exactly the
/// kind of thing that turns into a bug report from another program entirely.
std::uint32_t greeted_pid(std::string_view reply) {
    if (!reply.starts_with(greeting_prefix)) {
        return 0;
    }
    std::uint32_t pid = 0;
    for (const char c : reply.substr(greeting_prefix.size())) {
        if (c < '0' || c > '9') {
            break;
        }
        pid = pid * 10 + static_cast<std::uint32_t>(c - '0');
    }
    return pid;
}

} // namespace

std::string key(const std::vector<std::string>& args) {
    std::string replay;
    std::string model;
    std::string seed;
    std::string match;
    std::string left;
    std::string right;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        // Every recognised flag here takes a value; one written without its
        // value is a line main.cpp ignores too, and disengaging is safer than
        // guessing what was meant.
        if (i + 1 >= args.size()) {
            return {};
        }
        const std::string& value = args[i + 1];
        ++i;
        if (arg == "--replay") {
            replay = normalised(value);
        } else if (arg == "--watch-model") {
            model = normalised(value);
        } else if (arg == "--seed") {
            seed = value;
        } else if (arg == "--match") {
            match = normalised(value);
        } else if (arg == "--match-left") {
            left = normalised(value);
        } else if (arg == "--match-right") {
            right = normalised(value);
        } else {
            return {}; // automation, or a flag newer than this function
        }
    }
    // '\n' as the joint: it cannot appear in a flag and is vanishingly odd in
    // a path, so "watch\na\nb" and a model literally called "a\nb" cannot be
    // made to collide by accident.
    if (!match.empty()) {
        return "match\n" + match;
    }
    if (!left.empty() || !right.empty()) {
        if (left.empty() || right.empty()) {
            return {}; // main.cpp exits over this line; nothing to dedupe
        }
        return "pair\n" + left + "\n" + right;
    }
    if (!model.empty()) {
        return "watch\n" + model + "\n" + seed;
    }
    if (!replay.empty()) {
        return "replay\n" + replay;
    }
    if (!seed.empty()) {
        return {}; // a seed with nothing to apply it to is nobody's launcher
    }
    return "game";
}

std::string endpoint(const std::string& key, const std::string& user,
                     const std::string& runtime_dir) {
    const std::string tag = std::format("{:016x}", fnv1a(user + '\0' + key));
#ifdef _WIN32
    (void) runtime_dir; // pipes have a namespace, not a directory
    return R"(\\.\pipe\missile-defense-)" + tag;
#else
    const std::string directory = runtime_dir.empty() ? std::string{"/tmp"} : runtime_dir;
    return directory + "/missile-defense-" + tag + ".sock";
#endif
}

std::string machine_endpoint(const std::string& key) {
#ifdef _WIN32
    return endpoint(key, variable("USERNAME"), {});
#else
    return endpoint(key, variable("USER"), variable("XDG_RUNTIME_DIR"));
#endif
}

#ifdef _WIN32

Forward forward(const std::string& endpoint) {
    const std::wstring name{endpoint.begin(), endpoint.end()}; // ASCII by construction
    HANDLE pipe = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                              0, nullptr);
    if (pipe == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PIPE_BUSY &&
        WaitNamedPipeW(name.c_str(), 500) != 0) {
        pipe = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0,
                           nullptr);
    }
    if (pipe == INVALID_HANDLE_VALUE) {
        return Forward::NoInstance;
    }
    const std::uint32_t pid = greeted_pid(bounded_read(pipe));
    if (pid == 0) {
        CloseHandle(pipe);
        return Forward::NoInstance;
    }
    // The privilege transfer that makes the raise a raise. This process was
    // started by whatever holds the foreground — the trainer's button, the
    // desktop — so it may pass the right along; the twin, backgrounded for
    // who knows how long, could not have taken it alone.
    AllowSetForegroundWindow(pid);
    const bool delivered = write_all(pipe, activate_message);
    FlushFileBuffers(pipe);
    CloseHandle(pipe);
    return delivered ? Forward::Raised : Forward::NoInstance;
}

namespace {
/// One listening instance of the pipe. FIRST_PIPE_INSTANCE on the first is
/// what turns "somebody already serves this identity" into a clean refusal
/// instead of two servers quietly sharing one name.
HANDLE pipe_instance(const std::wstring& name, bool first) {
    const DWORD open_mode =
        PIPE_ACCESS_DUPLEX | (first ? DWORD{FILE_FLAG_FIRST_PIPE_INSTANCE} : DWORD{0});
    return CreateNamedPipeW(name.c_str(), open_mode,
                            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                            PIPE_UNLIMITED_INSTANCES, 512, 512, 0, nullptr);
}
} // namespace

bool Server::start(const std::string& endpoint) {
    endpoint_ = endpoint;
    // The claiming instance is made *here*, not on the thread: the caller's
    // next move on refusal is to forward to whoever holds the name, and an
    // answer that arrived asynchronously would arrive after that decision.
    const HANDLE claimed = pipe_instance({endpoint_.begin(), endpoint_.end()}, true);
    if (claimed == INVALID_HANDLE_VALUE) {
        return false; // a twin holds the name
    }
    first_pipe_ = claimed;
    stop_ = false;
    done_ = false;
    thread_ = std::thread([this] { serve(); });
    return true;
}

void Server::serve() {
    const std::wstring name{endpoint_.begin(), endpoint_.end()};
    HANDLE pipe = static_cast<HANDLE>(first_pipe_);
    while (!stop_) {
        if (pipe == INVALID_HANDLE_VALUE) {
            pipe = pipe_instance(name, false);
            if (pipe == INVALID_HANDLE_VALUE) {
                break; // torn down under us: done serving
            }
        }
        const BOOL connected = ConnectNamedPipe(pipe, nullptr);
        if (connected == 0 && GetLastError() != ERROR_PIPE_CONNECTED) {
            CloseHandle(pipe);
            pipe = INVALID_HANDLE_VALUE;
            continue;
        }
        if (!stop_) {
            write_all(pipe, std::format("{}{}\n", greeting_prefix, GetCurrentProcessId()));
            if (bounded_read(pipe).starts_with("activate")) {
                notify();
            }
        }
        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
        pipe = INVALID_HANDLE_VALUE;
    }
    if (pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(pipe);
    }
    done_ = true;
}

namespace {
/// Unblock a ConnectNamedPipe by being, briefly, the client it waits for.
void nudge(const std::string& endpoint) {
    const std::wstring name{endpoint.begin(), endpoint.end()};
    const HANDLE pipe = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                    OPEN_EXISTING, 0, nullptr);
    if (pipe != INVALID_HANDLE_VALUE) {
        CloseHandle(pipe);
    }
}
} // namespace

#else

Forward forward(const std::string& endpoint) {
    const int fd = connect_to(endpoint);
    if (fd < 0) {
        return Forward::NoInstance;
    }
    const std::uint32_t pid = greeted_pid(bounded_read(fd));
    if (pid == 0) {
        close(fd);
        return Forward::NoInstance;
    }
    const bool delivered = write_all(fd, activate_message);
    close(fd);
    return delivered ? Forward::Raised : Forward::NoInstance;
}

bool Server::start(const std::string& endpoint) {
    endpoint_ = endpoint;
    if (endpoint_.size() >= sizeof(sockaddr_un::sun_path)) {
        return false; // XDG_RUNTIME_DIR from another dimension; live without
    }
    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        return false;
    }
    quiet_pipes(listen_fd_);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, endpoint_.c_str(), endpoint_.size() + 1);
    if (bind(listen_fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        // A socket file, but whose? A live twin answers a connect; a crashed
        // one left a corpse that refuses it, and a corpse is swept, not obeyed.
        const int probe = connect_to(endpoint_);
        if (probe >= 0) {
            close(probe);
            close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }
        unlink(endpoint_.c_str());
        if (bind(listen_fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
            close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }
    }
    if (listen(listen_fd_, 4) != 0) {
        close(listen_fd_);
        unlink(endpoint_.c_str());
        listen_fd_ = -1;
        return false;
    }
    stop_ = false;
    done_ = false;
    thread_ = std::thread([this] { serve(); });
    return true;
}

void Server::serve() {
    while (!stop_) {
        const int client = accept(listen_fd_, nullptr, nullptr);
        if (client < 0) {
            if (stop_ || (errno != EINTR && errno != ECONNABORTED)) {
                break;
            }
            continue;
        }
        quiet_pipes(client);
        if (!stop_) {
            write_all(client, std::format("{}{}\n", greeting_prefix, getpid()));
            if (bounded_read(client).starts_with("activate")) {
                notify();
            }
        }
        close(client);
    }
    done_ = true;
}

namespace {
/// Unblock an accept() by being, briefly, the client it waits for.
void nudge(const std::string& endpoint) {
    const int fd = connect_to(endpoint);
    if (fd >= 0) {
        close(fd);
    }
}
} // namespace

#endif

void Server::notify() {
    std::function<void()> callback;
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        if (callback_) {
            callback = callback_;
        } else {
            // Landed during our own startup — between the claim and the
            // window. Remembered, not dropped: the person double-launched and
            // deserves a raised window, not a swallowed click.
            pending_ = true;
        }
    }
    if (callback) {
        callback();
    }
}

void Server::on_activate(std::function<void()> callback) {
    bool fire = false;
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        callback_ = std::move(callback);
        fire = pending_ && static_cast<bool>(callback_);
        pending_ = false;
    }
    if (fire) {
        callback_(); // deliberately outside the lock; installer's own thread
    }
}

void Server::detach() {
    {
        const std::lock_guard<std::mutex> lock{mutex_};
        callback_ = nullptr; // from here no activation can touch the window
    }
    if (!thread_.joinable()) {
        return;
    }
    stop_ = true;
    // Nudge until the thread confirms it is out of its wait. A single nudge
    // can slip into the gap where the server is between pipe instances; every
    // read in serve() is deadline-bounded, so this loop is too.
    while (!done_) {
        nudge(endpoint_);
        std::this_thread::sleep_for(poll_step);
    }
    thread_.join();
#ifndef _WIN32
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        unlink(endpoint_.c_str());
    }
#endif
}

Server::~Server() {
    detach();
}

} // namespace md::instance
