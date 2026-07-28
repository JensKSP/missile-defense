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

#ifdef _WIN32
// CreateProcessW, for both process launches this file makes. Neither can go
// through the C runtime's `_popen` or Qt's QProcess — each needs a creation
// flag those wrappers either hardcode wrong or cannot pass at all. The two
// long comments below say which flag and what it cost.
#ifndef NOMINMAX // MinGW's libstdc++ predefines it; MSVC does not
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

/// The Python source both probes run: two integers on one line.
constexpr std::string_view version_probe =
    "import sys; print(sys.version_info[0], sys.version_info[1])";

/// Two integers out of what the probe printed, or nothing.
///
/// Parsed by hand rather than with `sscanf`, which is a vararg function and
/// rejected by cppcoreguidelines-pro-type-vararg — rightly, since the format
/// string and the arguments are unchecked. Two integers separated by a space
/// is not worth a vararg call: anything that does not parse is simply not an
/// interpreter, which is the answer this function exists to give. `from_chars`
/// stops at the first non-digit, so the line's own terminator — `\n` here,
/// `\r\n` from a Windows pipe — needs no trimming to be ignored.
std::optional<Interpreter> parse_version(std::string_view line,
                                         const std::filesystem::path& candidate) {
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

#ifdef _WIN32

/// UTF-8 to UTF-16, which is the encoding the W API speaks.
///
/// UTF-8 is what the rest of this flow already travels in — the game hands
/// paths over as `QString::...::toStdString()` — so it is the one honest choice
/// here, even though `std::filesystem::path::string()` on Windows is only its
/// equal while the path is ASCII. Every path this touches is an install prefix
/// or an interpreter location, which in practice is exactly that.
std::wstring widen(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const int size = static_cast<int>(text.size());
    const int needed = MultiByteToWideChar(CP_UTF8, 0, text.data(), size, nullptr, 0);
    if (needed <= 0) {
        return {};
    }
    std::wstring wide(static_cast<std::size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), size, wide.data(), needed);
    return wide;
}

/// `command_line`'s output, gathered with no console window of its own.
///
/// Not `_popen`, which is how this used to be asked. `_popen` routes the
/// command through `cmd.exe /c`, and a console child of a GUI-subsystem
/// process gets a brand-new console *window* — so every interpreter probed
/// from the game's constructor flashed one, before the game window even
/// existed. An installed build probes several (it is the build that ships a
/// wheel), and starting it from Explorer opened as a stutter of black windows;
/// a checkout ships no wheel, probes nothing, and never showed a developer any
/// of this (reported from a real install, 2026-07-29). CREATE_NO_WINDOW is the
/// flag `_popen` has no way to pass — and running the candidate directly also
/// retires `cmd.exe` and its quoting rules from the probe entirely.
std::optional<std::string> read_windowless(std::wstring command_line) {
    SECURITY_ATTRIBUTES inheritable{};
    inheritable.nLength = sizeof(inheritable);
    inheritable.bInheritHandle = TRUE;
    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    if (CreatePipe(&read_end, &write_end, &inheritable, 0) == 0) {
        return std::nullopt;
    }
    // The child must inherit only its own end: a read end open in the child
    // would keep the pipe alive past its exit, and the loop below would never
    // see EOF.
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_end;
    // stderr into the same pipe rather than onto a console it does not have. A
    // candidate that writes anything there — the Microsoft Store stub, say —
    // fails the parse and is dropped, which is the answer it earned.
    startup.hStdError = write_end;
    PROCESS_INFORMATION process{};
    const BOOL started = CreateProcessW(nullptr, command_line.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(write_end); // the parent's copy; the child holds its own
    if (started == 0) {
        CloseHandle(read_end);
        return std::nullopt;
    }
    CloseHandle(process.hThread);
    std::string output;
    std::array<char, 256> chunk{};
    DWORD got = 0;
    while (ReadFile(read_end, chunk.data(), static_cast<DWORD>(chunk.size()), &got, nullptr) != 0 &&
           got != 0) {
        output.append(chunk.data(), got);
    }
    CloseHandle(read_end);
    // EOF means the child closed its stdout, which for an interpreter that just
    // printed and exited is the exit itself; the wait is bookkeeping, not a
    // pause.
    WaitForSingleObject(process.hProcess, INFINITE);
    CloseHandle(process.hProcess);
    return output;
}

/// What a candidate says `sys.version_info` is, or nothing.
///
/// Running it is the whole point: a name proves nothing. Windows ships app
/// execution aliases called `python.exe` that open the Microsoft Store and exit
/// 9009, and a `python3` on PATH may be a wrapper, a symlink or a shim. The only
/// question that can be answered honestly is "does this thing run and what does
/// it say it is".
std::optional<Interpreter> ask_version(const std::filesystem::path& candidate) {
    const std::wstring command_line =
        L'"' + candidate.wstring() + L"\" -c \"" + widen(version_probe) + L'"';
    const std::optional<std::string> output = read_windowless(command_line);
    if (!output.has_value()) {
        return std::nullopt;
    }
    return parse_version(*output, candidate);
}

#else

/// Closes whatever `popen` opened.
///
/// A named type rather than `decltype(&pclose)`: on glibc that function carries
/// attributes, and GCC rejects a template argument that has them under
/// `-Werror=ignored-attributes`. clang accepts it, which is exactly why the
/// release build compiles with GCC — it is the only build that does.
struct PipeCloser {
    void operator()(std::FILE* pipe) const noexcept {
        if (pipe != nullptr) {
            pclose(pipe);
        }
    }
};

/// What a candidate says `sys.version_info` is, or nothing. See the Windows
/// half above for why the question has to be asked by running it.
std::optional<Interpreter> ask_version(const std::filesystem::path& candidate) {
    const std::string command = '"' + candidate.string() + "\" -c \"" +
                                std::string{version_probe} + "\" 2>/dev/null";
    const std::unique_ptr<std::FILE, PipeCloser> pipe{popen(command.c_str(), "r")};
    if (!pipe) {
        return std::nullopt;
    }
    std::array<char, 64> buffer{};
    if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) == nullptr) {
        return std::nullopt;
    }
    return parse_version(buffer.data(), candidate);
}

#endif

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
        // Both empty is "no opinion" — a developer build, or a distribution one
        // where apt keeps the two in step. Only a machine that has both a
        // recorded install and a wheel to compare it against can be out of date.
        const bool comparable =
            !machine.wheel_version.empty() && !machine.installed_version.empty();
        if (comparable && machine.installed_version != machine.wheel_version) {
            return Offer::Update;
        }
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

std::string wheel_version(const std::filesystem::path& wheel) {
    const std::string name = wheel.filename().string();
    if (!name.starts_with("missile_defense-") || !name.ends_with(".whl")) {
        return {};
    }
    const std::size_t first = name.find('-');
    const std::size_t second = name.find('-', first + 1);
    if (second == std::string::npos) {
        return {};
    }
    return name.substr(first + 1, second - first - 1);
}

std::string install_script(const Interpreter& interpreter, const std::filesystem::path& wheel) {
    const std::string python = '"' + interpreter.path.string() + '"';
    // The recording half runs the *installed* package, so it cannot succeed
    // unless the install did — which is the whole reason it is chained rather
    // than done here.
    const std::string remember = "from missile_defense.runs.runner import record_interpreter; "
                                 "import sys, missile_defense; "
                                 "record_interpreter(sys.executable, missile_defense.__version__)";
    return python + " -m pip install --user --upgrade \"" + wheel.string() + "[trainer]\"" +
           " && " + python + " -c \"" + remember + '"';
}

std::vector<std::string> terminal_command(const std::string& script) {
#ifdef _WIN32
    // `/k` and not `/c`: the window stays open afterwards, which is the case
    // where its contents are the whole point. `cmd.exe` is a system binary and
    // the script is an argument to it, so there is no script *file* — which
    // Smart App Control would block (app/trainer.hpp records that lesson).
    return {"cmd.exe", "/k", script};
#elifdef __APPLE__
    // Terminal.app opens a *file*, not a command, so the script goes to `sh -c`
    // through `osascript`, which is the one way to get a visible window.
    return {"/usr/bin/osascript", "-e",
            "tell application \"Terminal\" to do script \"" + script + "\""};
#else
    // Never reached: a Linux build ships no wheel, so `decide` answers
    // NeedsPackage before anything gets here. `sh -c` anyway, so a caller that
    // ignores the platform still has something runnable rather than nothing.
    return {"/bin/sh", "-c", script};
#endif
}

#ifdef _WIN32
bool spawn_terminal(const std::string& script) {
    // `cmd.exe /S /K "<script>"`. With /S, cmd strips exactly the outer pair
    // of quotes and runs what stood between them verbatim — the documented
    // idiom for handing cmd a line that itself contains quotes. The outer pair
    // is added here and nothing inside is escaped, because for cmd there is no
    // escaping: between the stripped quotes is the command, exactly as a
    // person would have typed it. /K rather than /C so the window survives the
    // install and its output stays readable, which is the reason a terminal
    // was chosen over a progress bar in the first place (install.hpp).
    //
    // Win32 directly rather than `QProcess::startDetached`, which cannot make
    // this window exist. Its portable quoting writes every embedded quote as
    // \" — an escape sh reads and cmd.exe does not — and it passes
    // CREATE_NO_WINDOW whenever the parent has no console, which a
    // GUI-subsystem game never has. Together that opened an *invisible* cmd
    // sitting on a mangled command line forever: ENTER on the install notice
    // did nothing anyone could see, while a fresh orphaned cmd.exe joined Task
    // Manager per press (reported 2026-07-29). CREATE_NEW_CONSOLE is the
    // window; the hand-built line is the command.
    // The window dressing rides in the command itself rather than in
    // STARTUPINFO, because two different hosts may draw this window — classic
    // conhost, or Windows Terminal where it is the system default — and the
    // cmd builtins are the one interface both honour. `title` names the
    // window, which is also what lets it be found and raised below; `mode con`
    // sizes it, because an unsized console gets the user's console defaults,
    // and those were tuned for whatever the user does in a terminal, not for
    // this — on the machine that reported it, a window bigger than the screen.
    constexpr const wchar_t window_title[] = L"Missile Defense - trainer install";
    std::wstring command_line = L"cmd.exe /S /K \"title " + std::wstring{window_title} +
                                L" && mode con: cols=100 lines=30 && " + widen(script) + L'"';
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const BOOL started = CreateProcessW(nullptr, command_line.data(), nullptr, nullptr, FALSE,
                                        CREATE_NEW_CONSOLE, nullptr, nullptr, &startup, &process);
    if (started == 0) {
        return false;
    }
    // Raise it. A CREATE_NEW_CONSOLE window opens *behind* the game: Windows
    // leaves foreground with the process that has it, and that is the game —
    // which is also exactly what licenses this call, since only the foreground
    // process may hand foreground to another window. The keypress that led
    // here is at most a moment old, so the licence holds. Polled briefly
    // because the window exists only once the console host has drawn it;
    // best-effort, and a window that never appears just stays where it was.
    for (int attempt = 0; attempt < 40; ++attempt) {
        if (HWND window = FindWindowW(nullptr, window_title); window != nullptr) {
            SetForegroundWindow(window);
            break;
        }
        Sleep(50);
    }
    // Detached is the contract (the game neither waits nor reports), so the
    // handles close now and the terminal outlives whoever opened it.
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return true;
}
#endif

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
