// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "install.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <memory>
#include <system_error>

namespace md::install {

namespace {

/// Names worth asking about, most specific first.
///
/// The versioned ones are listed because an unversioned `python3` is often the
/// distribution's, and on a machine where someone installed 3.13 beside it the
/// versioned name is the only way to reach the newer one.
constexpr std::array<std::string_view, 6> interpreter_names{
    "python3.14", "python3.13", "python3.12", "python3", "python", "py",
};

#ifdef _WIN32
constexpr std::string_view exe_suffix = ".exe";
constexpr char path_separator = ';';
#else
constexpr std::string_view exe_suffix{};
constexpr char path_separator = ':';
#endif

/// What a candidate says `sys.version_info` is, or nothing.
///
/// Running it is the whole point: a name proves nothing. Windows ships app
/// execution aliases called `python.exe` that open the Microsoft Store and exit
/// 9009, and a `python3` on PATH may be a wrapper, a symlink or a shim. The only
/// question that can be answered honestly is "does this thing run and what does
/// it say it is".
/// Closes whatever `popen` opened, under the name this platform gives it.
///
/// A named type rather than `decltype(&pclose)`: on glibc that function carries
/// attributes, and GCC rejects a template argument that has them under
/// `-Werror=ignored-attributes`. clang accepts it, which is exactly why the
/// release build compiles with GCC — it is the only build that does.
struct PipeCloser {
    void operator()(std::FILE* pipe) const noexcept {
        if (pipe != nullptr) {
#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif
        }
    }
};

/// What a candidate says `sys.version_info` is, or nothing.
///
/// Running it is the whole point: a name proves nothing. Windows ships app
/// execution aliases called `python.exe` that open the Microsoft Store and exit
/// 9009, and a `python3` on PATH may be a wrapper, a symlink or a shim. The only
/// question that can be answered honestly is "does this thing run and what does
/// it say it is".
std::optional<Interpreter> ask_version(const std::filesystem::path& candidate) {
    std::string command = '"' + candidate.string() +
                          "\" -c \"import sys; print(sys.version_info[0], sys.version_info[1])\"";
#ifdef _WIN32
    command = '"' + command + '"'; // cmd.exe strips one layer of quoting
    const std::unique_ptr<std::FILE, PipeCloser> pipe{_popen(command.c_str(), "r")};
#else
    command += " 2>/dev/null";
    const std::unique_ptr<std::FILE, PipeCloser> pipe{popen(command.c_str(), "r")};
#endif
    if (!pipe) {
        return std::nullopt;
    }
    std::array<char, 64> buffer{};
    if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) == nullptr) {
        return std::nullopt;
    }
    // Parsed by hand rather than with `sscanf`, which is a vararg function and
    // rejected by cppcoreguidelines-pro-type-vararg — rightly, since the format
    // string and the arguments are unchecked. Two integers separated by a space
    // is not worth a vararg call: anything that does not parse is simply not an
    // interpreter, which is the answer this function exists to give.
    const std::string_view line{buffer.data()};
    const std::size_t space = line.find(' ');
    if (space == std::string_view::npos) {
        return std::nullopt;
    }
    Interpreter found;
    found.path = candidate;
    const auto number = [](std::string_view text, int& out) {
        const char* begin = text.data();
        const char* end = begin + text.size();
        const auto [stop, ec] = std::from_chars(begin, end, out);
        return ec == std::errc{} && stop != begin;
    };
    if (!number(line.substr(0, space), found.major) ||
        !number(line.substr(space + 1), found.minor)) {
        return std::nullopt;
    }
    return found;
}

} // namespace

std::optional<Interpreter> best(const Machine& machine) {
    const Interpreter* winner = nullptr;
    for (const Interpreter& candidate : machine.interpreters) {
        if (!candidate.usable()) {
            continue;
        }
        if (winner == nullptr || candidate.major > winner->major ||
            (candidate.major == winner->major && candidate.minor > winner->minor)) {
            winner = &candidate;
        }
    }
    return winner == nullptr ? std::nullopt : std::optional{*winner};
}

Offer decide(const Machine& machine, const bool trainer_found) {
    if (trainer_found) {
        return Offer::Start;
    }
    // No wheel means this build cannot install anything itself. Usually that is
    // a distribution build, where pointing at pip would be pointing at PEP 668 —
    // which refuses by design and is right to. It is also a Windows or macOS
    // build that shipped without one, where the answer is a pip install from
    // PyPI. Both are "read the instructions", which is why they share an answer
    // and why the screen names neither package manager.
    if (machine.wheel.empty()) {
        return Offer::NeedsPackage;
    }
    return best(machine).has_value() ? Offer::Install : Offer::NeedsPython;
}

std::vector<std::string> pip_command(const Interpreter& interpreter,
                                     const std::filesystem::path& wheel) {
    return {interpreter.path.string(),   "-m", "pip", "install", "--user", "--upgrade",
            wheel.string() + "[trainer]"};
}

std::vector<std::string> terminal_command(const std::vector<std::string>& inner) {
    std::vector<std::string> command;
#ifdef _WIN32
    // `/k` and not `/c`: the window stays open on failure, which is the case
    // where its contents are the whole point.
    command = {"cmd.exe", "/k"};
    command.insert(command.end(), inner.begin(), inner.end());
#elifdef __APPLE__
    // `open -a Terminal` takes a *file* to open, not a command, so the command
    // goes through `sh -c` and Terminal is asked to run that. Quoting is left to
    // the caller's argv rather than rebuilt into one string here.
    command = {"/usr/bin/open", "-a", "Terminal"};
    command.insert(command.end(), inner.begin(), inner.end());
#else
    // Never reached: a Linux build ships no wheel, so `decide` answers
    // NeedsPackage before anything gets here. Returned unchanged rather than
    // empty so a caller that ignores the platform still has something runnable.
    command = inner;
#endif
    return command;
}

std::vector<Interpreter> probe_interpreters(const std::string& search_path) {
    std::vector<Interpreter> found;
    std::string_view rest{search_path};
    while (!rest.empty()) {
        const std::size_t split = rest.find(path_separator);
        const std::string_view directory = rest.substr(0, split);
        rest = split == std::string_view::npos ? std::string_view{} : rest.substr(split + 1);
        if (directory.empty()) {
            continue;
        }
        for (const std::string_view name : interpreter_names) {
            std::filesystem::path candidate =
                std::filesystem::path{directory} / (std::string{name} + std::string{exe_suffix});
            std::error_code ec;
            if (!std::filesystem::is_regular_file(candidate, ec)) {
                continue;
            }
            const bool seen = std::ranges::any_of(
                found, [&](const Interpreter& other) { return other.path == candidate; });
            if (!seen) {
                if (auto version = ask_version(candidate)) {
                    found.push_back(*version);
                }
            }
        }
    }
    std::ranges::sort(found, [](const Interpreter& a, const Interpreter& b) {
        return a.major != b.major ? a.major > b.major : a.minor > b.minor;
    });
    return found;
}

std::filesystem::path shipped_wheel(const std::filesystem::path& beside) {
    if (beside.empty()) {
        return {};
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(beside, ec)) {
        return {};
    }
    // The newest by filename, so a directory left holding two versions after an
    // upgrade resolves rather than picking whichever the filesystem lists first.
    std::filesystem::path newest;
    for (const auto& entry : std::filesystem::directory_iterator{beside, ec}) {
        const std::filesystem::path& path = entry.path();
        if (path.extension() != ".whl") {
            continue;
        }
        if (!path.filename().string().starts_with("missile_defense-")) {
            continue;
        }
        if (newest.empty() || path.filename() > newest.filename()) {
            newest = path;
        }
    }
    return newest;
}

} // namespace md::install
