# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Screenshot / video capture of the running game window, on any of the three.

    poe shot                 # -> shot.png
    poe shot -- --launch     # start the game first, capture it, close it
    poe rec                  # -> clip.mp4
    poe contact clip.mp4     # -> contact.png, a grid of frames

There is no portable way to photograph a window, so there is one backend per
platform and each uses what that platform already ships:

| Platform | Screenshot | Video |
|---|---|---|
| Linux/X11 | ImageMagick `import -window` | frame loop + ffmpeg, PulseAudio for sound |
| Windows | PowerShell + `Graphics.CopyFromScreen` | ffmpeg `gdigrab` |
| macOS | `screencapture -R` over the window's rect | ffmpeg `avfoundation` |

None of them is a new dependency of the project: they are the OS's own tools,
plus the ffmpeg and ImageMagick this file already wanted.

**Two traps, both of which produce a plausible but wrong picture rather than an
error**, and both from the game being a Vulkan window:

* A **fullscreen** swapchain bypasses the compositor, so a screen-region grab
  gets the desktop *behind* it and `PrintWindow` returns black. Capture in
  windowed mode; this warns when the window covers the whole screen rather than
  silently handing back the wrong image.
* A **scaled display** (1280x800 logical over 2560x1600 physical) makes a
  DPI-unaware capture take the top-left quadrant at 1:1. The Windows backend
  declares itself DPI-aware before it reads a single coordinate.

And one nothing here can prevent: on Windows and macOS the pixels come from the
*screen*, so a notification or a task switcher that happens to be over the
window lands in the picture. The window is raised first, which removes the
common case of another application on top of it, but a toast that arrives during
the grab is in the file. **Look at the image before drawing conclusions from
it** — the failure mode of all three of these is a plausible picture, not an
error.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from . import _util

T = TypeVar("T")

#: What the game calls its window. A substring match, so `--title` can aim the
#: same tool at the training console without knowing its full caption.
WINDOW_TITLE = "Missile Defense"

#: Windows: the game, and the interpreter the console runs in. Consulted only
#: when the title search comes up empty — a `QVulkanWindow` is not always
#: findable by caption, but its process always knows its own main window.
FALLBACK_PROCESSES = ("md_app", "missile-defense", "python", "pythonw")


class CaptureError(RuntimeError):
    """Something the user can fix — the message says what."""


# ---- Windows -----------------------------------------------------------------

#: Finds a top-level window by title substring and writes a PNG of its screen
#: region. Two choices in here are the difference between a picture and a
#: puzzle, and both come from the game being a Vulkan window:
#:
#: * ``CopyFromScreen`` rather than ``PrintWindow``, which returns black for a
#:   swapchain the compositor is not redrawing into.
#: * ``Get-Process``'s ``MainWindowTitle`` rather than ``EnumWindows``.
#:   ``FindWindowW`` does not find a ``QVulkanWindow`` even given its exact
#:   caption, and driving ``EnumWindows`` from PowerShell means marshalling a
#:   callback into a script block — which crashes the host outright
#:   (``STATUS_STACK_BUFFER_OVERRUN``). .NET already walks that list natively
#:   and hands back both the handle and the caption.
#:
#: .NET is always present on Windows, so none of this is a dependency.
_POWERSHELL_GRAB = r"""
param([string]$Needle, [string]$Out, [string[]]$Processes, [double]$Wait = 0)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class MdCap {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern int GetSystemMetrics(int index);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
# Before any coordinate is read: on a scaled display an unaware process is told
# about a virtual 1280x800 desktop and grabs a quarter of the real one.
[void][MdCap]::SetProcessDPIAware()

$deadline = (Get-Date).AddSeconds($Wait)
do {
  $windowed = Get-Process | Where-Object { $_.MainWindowHandle -ne 0 }
  $found = $windowed | Where-Object { $_.MainWindowTitle -like "*$Needle*" } |
           Select-Object -First 1
  if (-not $found) {
    # A window whose caption has not been set yet still belongs to a process
    # whose name we know.
    $found = $windowed | Where-Object { $Processes -contains $_.ProcessName } |
             Select-Object -First 1
  }
  if ($found) { break }
  Start-Sleep -Milliseconds 250
} while ((Get-Date) -lt $deadline)
if (-not $found) { Write-Error "no window matching '$Needle'"; exit 1 }

$title = $found.MainWindowTitle
# CopyFromScreen photographs the *screen* at the window's rectangle, so anything
# covering it is what comes back — an editor over the game, and a screenshot
# that looks like a screenshot of something else. Raise it first. SW_RESTORE
# also un-minimises, which is the other way the rectangle can be a lie.
[void][MdCap]::ShowWindow($found.MainWindowHandle, 9)
[void][MdCap]::SetForegroundWindow($found.MainWindowHandle)
Start-Sleep -Milliseconds 400

$r = New-Object MdCap+RECT
[void][MdCap]::GetWindowRect($found.MainWindowHandle, [ref]$r)
$w = $r.Right - $r.Left; $h = $r.Bottom - $r.Top
if ($w -le 0 -or $h -le 0) { Write-Error "window '$title' has no area (minimised?)"; exit 1 }

$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($r.Left, $r.Top, 0, 0, $bmp.Size)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# SM_CXSCREEN / SM_CYSCREEN, for the fullscreen warning on the other side.
Write-Output ("{0}|{1}|{2}|{3}|{4}" -f $title, $w, $h,
  [MdCap]::GetSystemMetrics(0), [MdCap]::GetSystemMetrics(1))
"""


def _powershell() -> str:
    return _util.tool("powershell", "pwsh")


def _shot_windows(out: Path, title: str, *, wait: float = 0.0) -> None:
    """Grab the window, waiting up to ``wait`` seconds for one to appear.

    The wait is inside the script rather than a poll from here: each PowerShell
    start-up costs a third of a second, and a window that is simply not there
    should still be reported at once rather than after ten of them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "grab.ps1"
        script.write_text(_POWERSHELL_GRAB, encoding="utf-8")
        done = subprocess.run(
            [
                _powershell(),
                *("-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"),
                *("-File", str(script)),
                *("-Needle", title),
                *("-Out", str(out.resolve())),
                # One argv token: PowerShell's own parser splits it into the
                # [string[]] the script declares.
                *("-Processes", ",".join(FALLBACK_PROCESSES)),
                *("-Wait", f"{wait:g}"),
            ],
            capture_output=True,
            text=True,
        )
    if done.returncode != 0:
        raise CaptureError(_no_window(title, done.stderr.strip() or "PowerShell found nothing"))
    _warn_if_fullscreen(done.stdout.strip())


def _warn_if_fullscreen(report: str) -> None:
    """``title|w|h|screen_w|screen_h`` — the last four are the fullscreen check.

    A window the size of the screen is almost certainly a fullscreen Vulkan
    surface, and a screen grab of one of those is the desktop behind it. Said
    out loud, because the picture that comes back otherwise looks fine until you
    try to read the score off it.
    """
    parts = report.rsplit("|", 4)
    if len(parts) != 5:
        return
    title, width, height, screen_width, screen_height = parts
    print(f"captured '{title}' at {width}x{height}", file=sys.stderr)
    if (width, height) == (screen_width, screen_height):
        print(
            "warning: that window fills the screen. A fullscreen Vulkan surface "
            "bypasses the compositor, so this image may be the desktop behind it "
            "— switch the game to windowed and take it again.",
            file=sys.stderr,
        )


def _record_windows(
    out: Path, seconds: float, fps: float, title: str, *, audio: bool = True
) -> None:
    """gdigrab, which ffmpeg aims at a window by caption all by itself.

    Silent whatever ``audio`` says: Windows has no loopback capture device
    unless somebody installed one, and a flag that needs `virtual-audio-capturer`
    to exist is a flag that usually fails.
    """
    del audio
    _util.run(
        [
            _util.tool("ffmpeg"),
            *("-y", "-hide_banner", "-loglevel", "error"),
            *("-f", "gdigrab", "-framerate", f"{fps:.0f}"),
            *("-i", f"title={title}"),
            *("-t", f"{seconds:g}"),
            *("-vf", CROP, "-c:v", "libx264", "-pix_fmt", "yuv420p"),
            str(out),
        ]
    )


# ---- macOS -------------------------------------------------------------------

#: The window's position and size, via the accessibility API — the only way to
#: ask for one window's rectangle without a compiled bridge. `screencapture`
#: itself can only take a window *interactively* (`-w` waits for a click), so
#: the region comes from here and the pixels from `screencapture -R`.
_APPLESCRIPT_RECT = """
tell application "System Events"
    set matches to (every process whose name contains "%s")
    if (count of matches) is 0 then error "no process"
    tell item 1 of matches
        set {x, y} to position of window 1
        set {w, h} to size of window 1
    end tell
end tell
return (x as text) & "," & (y as text) & "," & (w as text) & "," & (h as text)
"""


def _macos_rect(process: str) -> str:
    done = subprocess.run(
        ["osascript", "-e", _APPLESCRIPT_RECT % process], capture_output=True, text=True
    )
    if done.returncode != 0 or not done.stdout.strip():
        raise CaptureError(
            f"could not find a '{process}' window. Start it with `poe app`, or pass "
            "--launch.\nIf it *is* running, macOS needs your terminal ticked under "
            "System Settings → Privacy & Security → Accessibility before it may ask "
            "another app where its windows are."
        )
    return done.stdout.strip()


def _shot_macos(out: Path, title: str, *, wait: float = 0.0) -> None:
    del title  # the accessibility API matches on the process, not the caption
    rect = _retry(lambda: _macos_rect(MACOS_PROCESS), wait)
    _util.run([_util.tool("screencapture"), "-x", "-R", rect, str(out)])


def _record_macos(out: Path, seconds: float, fps: float, title: str, *, audio: bool = True) -> None:
    """avfoundation captures a *screen*, so the window is cropped out of it.

    Silent whatever ``audio`` says: macOS routes no system output to a capture
    device without a loopback driver such as BlackHole installed.
    """
    del title, audio
    left, top, width, height = (int(float(part)) for part in _macos_rect(MACOS_PROCESS).split(","))
    _util.run(
        [
            _util.tool("ffmpeg"),
            *("-y", "-hide_banner", "-loglevel", "error"),
            *("-f", "avfoundation", "-framerate", f"{fps:.0f}", "-capture_cursor", "0"),
            *("-i", f"{MACOS_SCREEN}:none"),
            *("-t", f"{seconds:g}"),
            *("-vf", f"crop={width}:{height}:{left}:{top},{CROP}"),
            *("-c:v", "libx264", "-pix_fmt", "yuv420p"),
            str(out),
        ]
    )


#: What the bundle's executable is called — `md_app.app` (app/CMakeLists.txt).
MACOS_PROCESS = "md_app"
#: avfoundation numbers capture devices; screen 1 is the first display on a Mac
#: with no capture cards. `ffmpeg -f avfoundation -list_devices true -i ""` says.
MACOS_SCREEN = "1"


# ---- Linux / X11 -------------------------------------------------------------

WINDOW_ID_FIELD = 3  # 'xwininfo: Window id: 0x320000f "Missile Defense"'


def _env(display: str) -> dict[str, str]:
    return dict(os.environ, DISPLAY=display)


def _display() -> str:
    return os.environ.get("DISPLAY", ":0")


def _window_id(display: str, title: str) -> str | None:
    result = _util.run(
        [_util.tool("xwininfo"), "-name", title],
        env=_env(display),
        check=False,
        capture=True,
        quiet=True,
    )
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if "Window id:" in line:
            return line.split()[WINDOW_ID_FIELD]
    return None


def _require_x11_window(title: str, wait: float = 0.0) -> tuple[str, str]:
    display = _display()

    def find() -> str:
        window = _window_id(display, title)
        if window is None:
            raise CaptureError(_no_window(title, f"nothing on DISPLAY={display}"))
        return window

    return display, _retry(find, wait)


def _shot_x11(out: Path, title: str, *, wait: float = 0.0) -> None:
    display, window = _require_x11_window(title, wait)
    _util.run([_util.tool("import"), "-window", window, str(out)], env=_env(display))


def _record_x11(out: Path, seconds: float, fps: float, title: str, *, audio: bool = True) -> None:
    """A frame loop rather than `x11grab`, because the sound has to line up.

    `import` is driven at a measured rate and the real rate is handed to ffmpeg
    afterwards, so a machine that could not keep up produces a clip that plays
    at the right *speed* rather than one that drifts against its own audio.
    """
    display, window = _require_x11_window(title)
    ffmpeg = _util.tool("ffmpeg")
    import_bin = _util.tool("import")
    recorder: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            wav = frames_dir / "audio.wav"
            monitor = _default_monitor() if audio else None
            if monitor is not None:
                recorder = subprocess.Popen(
                    [
                        ffmpeg,
                        *("-y", "-hide_banner", "-loglevel", "error"),
                        *("-f", "pulse", "-i", monitor, "-t", str(seconds)),
                        str(wav),
                    ],
                    env=dict(os.environ, LC_ALL="C"),
                )

            start = time.monotonic()
            count = 0
            while time.monotonic() - start < seconds:
                count += 1
                frame = frames_dir / f"f_{count:06d}.png"
                grab = subprocess.run(
                    [import_bin, "-window", window, str(frame)], env=_env(display), check=False
                )
                if grab.returncode != 0:
                    break
                time.sleep(1.0 / fps)
            actual = count / max(time.monotonic() - start, 1e-6)
            if recorder is not None:
                recorder.wait()

            common = [
                ffmpeg,
                *("-y", "-hide_banner", "-loglevel", "error"),
                *("-framerate", f"{actual:.3f}", "-start_number", "1"),
                *("-i", str(frames_dir / "f_%06d.png")),
            ]
            encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
            if wav.exists() and wav.stat().st_size > 0:
                cmd = [*common, "-i", str(wav), "-vf", CROP, *encode]
                cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest", str(out)]
            else:
                cmd = [*common, "-vf", CROP, *encode, str(out)]
            _util.run(cmd, env=dict(os.environ, LC_ALL="C"))
            print(f"{count} frames, {actual:.2f} fps effective", file=sys.stderr)
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()


def _default_monitor() -> str | None:
    pactl = _util.tool_optional("pactl")
    if pactl is None:
        return None
    sink = _util.run([pactl, "get-default-sink"], check=False, capture=True, quiet=True)
    name = sink.stdout.strip()
    return f"{name}.monitor" if name else None


# ---- the parts that are the same everywhere ----------------------------------

#: libx264 refuses an odd dimension, and a window is whatever size it is.
CROP = "crop=trunc(iw/2)*2:trunc(ih/2)*2"

#: How long `--launch` gives a freshly started game to put a window up. Generous,
#: because it is only ever spent when the window is genuinely not there yet.
LAUNCH_WAIT_S = 15.0


def _retry(find: Callable[[], T], seconds: float) -> T:
    """Call ``find`` until it stops raising, or for ``seconds`` and then let it.

    A window that is not there is the normal answer when nothing was launched,
    so a zero wait must cost nothing: the first call is made either way and its
    exception is the one that reaches the user.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            return find()
        except CaptureError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def _no_window(title: str, detail: str) -> str:
    return (
        f"no window matching '{title}' ({detail}).\n"
        "Start the game with `poe app`, or pass --launch to have this start and "
        "close one for you."
    )


def _launch_environ() -> dict[str, str]:
    """The environment the game is started in — the same two quirks as the console.

    Kept in step with `md.ui.runner.launch_environ` by hand rather than by
    import: `tools` is developer scaffolding and must not need the package on
    its path to run.
    """
    env = dict(os.environ)
    if sys.platform == "linux":
        env.setdefault("QT_QPA_PLATFORM", "xcb")  # an X11 window, so it is grabbable
    elif sys.platform == "win32":
        msys = Path(env.get("MSYS2_ROOT", "C:/msys64")) / "clang64/bin"
        if msys.is_dir():
            # Or the MinGW build dies looking for libc++.dll, with no window and
            # a modal error box nobody is there to dismiss.
            env["PATH"] = f"{msys}{os.pathsep}{env.get('PATH', '')}"
    return env


#: A window exists before it has drawn anything, and a capture of a half-drawn
#: one is worse than a slow capture. Waiting for the *window* is each backend's
#: job (see `_retry`); this is the frame after it.
LAUNCH_SETTLE_S = 1.5


def _launch() -> subprocess.Popen[bytes]:
    binary = _util.app_binary()
    if not binary.exists():
        _util.run(["cmake", "--build", "--preset", "release"], capture=True)
    return subprocess.Popen([str(binary)], env=_launch_environ())


def _wait_for_window(title: str, seconds: float) -> None:
    """Block until something matching ``title`` is on screen. Screenshot-free."""
    if sys.platform == "linux":
        _require_x11_window(title, seconds)
    elif sys.platform == "darwin":
        _retry(lambda: _macos_rect(MACOS_PROCESS), seconds)
    else:
        # No cheap query on Windows that is not another PowerShell start-up, and
        # gdigrab fails loudly on a missing title anyway.
        time.sleep(min(seconds, LAUNCH_SETTLE_S * 2))


#: One screenshot and one recorder per platform. Adding a platform — Wayland is
#: the obvious next one — is a function and a line here, not a refactor.
#: `(out, title, *, wait)` and `(out, seconds, fps, title, *, audio)` — spelled
#: loosely because a backend documents ignoring what its platform cannot give it,
#: and a keyword with a default is how it says so.
Shot = Callable[..., None]
Recorder = Callable[..., None]

_SHOT: dict[str, Shot] = {"linux": _shot_x11, "win32": _shot_windows, "darwin": _shot_macos}
_RECORD: dict[str, Recorder] = {
    "linux": _record_x11,
    "win32": _record_windows,
    "darwin": _record_macos,
}


def _backend(table: dict[str, T], what: str) -> T:
    backend = table.get(sys.platform)
    if backend is None:
        raise CaptureError(
            f"'{what}' has no backend for {sys.platform}. Linux/X11, Windows and "
            "macOS are covered; adding one is a function in tools/capture.py."
        )
    return backend


def screenshot(out: str = "shot.png", *, launch: bool = False, title: str = WINDOW_TITLE) -> int:
    process = _launch() if launch else None
    try:
        _backend(_SHOT, "screenshot")(Path(out), title, wait=LAUNCH_WAIT_S if launch else 0.0)
    except CaptureError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if process is not None:
            process.terminate()
    print(f"wrote {out}")
    return 0


def record(
    out: str = "clip.mp4",
    seconds: float = 6.0,
    fps: float = 20.0,
    *,
    launch: bool = False,
    audio: bool = True,
    title: str = WINDOW_TITLE,
) -> int:
    process = _launch() if launch else None
    try:
        if process is not None:
            # A recorder starts capturing immediately, so unlike a screenshot it
            # cannot wait for the window from inside — the first second would be
            # the desktop.
            _wait_for_window(title, LAUNCH_WAIT_S)
            time.sleep(LAUNCH_SETTLE_S)
        # Only X11 has a sound source this can find without being told one, so
        # the flag reaches every backend and two of them document ignoring it.
        _backend(_RECORD, "record")(Path(out), seconds, fps, title, audio=audio)
    except CaptureError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if process is not None:
            process.terminate()
    print(f"wrote {out}")
    return 0


def frames(source: str, out: str = "contact.png", fps: float = 3.0) -> int:
    """A grid of stills from a clip — the one command that needs no window."""
    ffmpeg = _util.tool("ffmpeg")
    montage = _util.tool("montage")
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%04d.png")
        _util.run(
            [
                ffmpeg,
                *("-y", "-hide_banner", "-loglevel", "error"),
                *("-i", source, "-vf", f"fps={fps}"),
                pattern,
            ]
        )
        tiles = sorted(str(path) for path in Path(tmp).glob("f_*.png"))
        _util.run(
            [montage, *tiles, "-tile", "5x", "-geometry", "384x+3+3", "-background", "#111827", out]
        )
    print(f"wrote {out}")
    return 0


def _bool_flag(args: list[str], name: str) -> bool:
    if name in args:
        args.remove(name)
        return True
    return False


def _value_flag(args: list[str], name: str, default: str) -> str:
    if name not in args:
        return default
    index = args.index(name)
    if index + 1 >= len(args):
        raise SystemExit(f"error: {name} needs a value")
    value = args[index + 1]
    del args[index : index + 2]
    return value


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args else ""
    launch = _bool_flag(args, "--launch")
    title = _value_flag(args, "--title", WINDOW_TITLE)

    if command == "screenshot":
        return screenshot(args[0] if args else "shot.png", launch=launch, title=title)
    if command == "record":
        no_audio = _bool_flag(args, "--no-audio")
        out = args[0] if len(args) > 0 else "clip.mp4"
        seconds = float(args[1]) if len(args) > 1 else 6.0
        fps = float(args[2]) if len(args) > 2 else 20.0
        return record(out, seconds, fps, launch=launch, audio=not no_audio, title=title)
    if command == "frames":
        if not args:
            print("usage: python -m tools.capture frames INPUT [OUT] [FPS]", file=sys.stderr)
            return 2
        out = args[1] if len(args) > 1 else "contact.png"
        fps = float(args[2]) if len(args) > 2 else 3.0
        return frames(args[0], out, fps)

    print(
        "usage: python -m tools.capture {screenshot|record|frames} [OUT] "
        "[--launch] [--title SUBSTRING]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
