# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Screenshot / video capture of the running game window.

``screenshot`` and ``record`` use an X11 backend (ImageMagick ``import`` +
``xwininfo``), so they run on Linux/X11 only — on other platforms use the OS's
own capture tools. ``frames`` (video -> contact sheet) is cross-platform
(ffmpeg + ImageMagick ``montage``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import _util

WINDOW_TITLE = "Missile Defense"


def _require_x11(what: str) -> None:
    if sys.platform != "linux":
        raise SystemExit(
            f"error: '{what}' has an X11/Linux backend only; on {sys.platform} "
            "use your OS screen-capture tools."
        )


def _env(display: str) -> dict[str, str]:
    return dict(os.environ, DISPLAY=display)


def _window_id(display: str) -> str | None:
    result = _util.run(
        [_util.tool("xwininfo"), "-name", WINDOW_TITLE],
        env=_env(display),
        check=False,
        capture=True,
        quiet=True,
    )
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if "Window id:" in line:  # e.g. 'xwininfo: Window id: 0x320000f "Missile Defense"'
            return line.split()[3]
    return None


def _launch(display: str) -> subprocess.Popen[bytes]:
    binary = _util.app_binary()
    if not binary.exists():
        _util.run(["cmake", "--build", "--preset", "release"], capture=True)
    env = _env(display)
    env["QT_QPA_PLATFORM"] = "xcb"
    proc = subprocess.Popen([str(binary)], env=env)
    for _ in range(100):
        if _window_id(display) is not None:
            break
        time.sleep(0.1)
    time.sleep(0.4)
    return proc


def screenshot(out: str = "shot.png", *, launch: bool = False) -> int:
    _require_x11("screenshot")
    display = os.environ.get("DISPLAY", ":0")
    proc = _launch(display) if launch else None
    try:
        wid = _window_id(display)
        if wid is None:
            print(
                f"error: no '{WINDOW_TITLE}' window on DISPLAY={display}. "
                "Start it with 'poe app', or pass --launch.",
                file=sys.stderr,
            )
            return 1
        _util.run([_util.tool("import"), "-window", wid, out], env=_env(display))
        print(f"wrote {out}")
        return 0
    finally:
        if proc is not None:
            proc.terminate()


def record(
    out: str = "clip.mp4",
    seconds: float = 6.0,
    fps: float = 20.0,
    *,
    launch: bool = False,
    audio: bool = True,
) -> int:
    _require_x11("record")
    display = os.environ.get("DISPLAY", ":0")
    ffmpeg = _util.tool("ffmpeg")
    proc = _launch(display) if launch else None
    audio_proc: subprocess.Popen[bytes] | None = None
    try:
        wid = _window_id(display)
        if wid is None:
            print(f"error: no '{WINDOW_TITLE}' window on DISPLAY={display}.", file=sys.stderr)
            return 1

        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            wav = frames_dir / "audio.wav"

            if audio:
                monitor = _default_monitor()
                if monitor is not None:
                    audio_proc = subprocess.Popen(
                        [
                            ffmpeg,
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-f",
                            "pulse",
                            "-i",
                            monitor,
                            "-t",
                            str(seconds),
                            str(wav),
                        ],
                        env=dict(os.environ, LC_ALL="C"),
                    )

            import_bin = _util.tool("import")
            start = time.monotonic()
            count = 0
            while time.monotonic() - start < seconds:
                count += 1
                frame = frames_dir / f"f_{count:06d}.png"
                grab = subprocess.run(
                    [import_bin, "-window", wid, str(frame)],
                    env=_env(display),
                    check=False,
                )
                if grab.returncode != 0:
                    break
                time.sleep(1.0 / fps)
            elapsed = max(time.monotonic() - start, 1e-6)
            actual_fps = count / elapsed
            if audio_proc is not None:
                audio_proc.wait()

            pattern = str(frames_dir / "f_%06d.png")
            crop = "crop=trunc(iw/2)*2:trunc(ih/2)*2"
            common = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{actual_fps:.3f}",
                "-start_number",
                "1",
                "-i",
                pattern,
            ]
            if audio and wav.exists() and wav.stat().st_size > 0:
                cmd = [
                    *common,
                    "-i",
                    str(wav),
                    "-vf",
                    crop,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-shortest",
                    out,
                ]
            else:
                cmd = [*common, "-vf", crop, "-c:v", "libx264", "-pix_fmt", "yuv420p", out]
            _util.run(cmd, env=dict(os.environ, LC_ALL="C"))
            print(f"wrote {out} ({count} frames, {actual_fps:.2f} fps effective)")
            return 0
    finally:
        if audio_proc is not None and audio_proc.poll() is None:
            audio_proc.terminate()
        if proc is not None:
            proc.terminate()


def _default_monitor() -> str | None:
    pactl = _util.tool_optional("pactl")
    if pactl is None:
        return None
    sink = _util.run([pactl, "get-default-sink"], check=False, capture=True, quiet=True)
    name = sink.stdout.strip()
    return f"{name}.monitor" if name else None


def frames(source: str, out: str = "contact.png", fps: float = 3.0) -> int:
    ffmpeg = _util.tool("ffmpeg")
    montage = _util.tool("montage")
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%04d.png")
        _util.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                source,
                "-vf",
                f"fps={fps}",
                pattern,
            ]
        )
        tiles = sorted(str(p) for p in Path(tmp).glob("f_*.png"))
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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args else ""
    launch = _bool_flag(args, "--launch")

    if command == "screenshot":
        return screenshot(args[0] if args else "shot.png", launch=launch)
    if command == "record":
        no_audio = _bool_flag(args, "--no-audio")
        out = args[0] if len(args) > 0 else "clip.mp4"
        seconds = float(args[1]) if len(args) > 1 else 6.0
        fps = float(args[2]) if len(args) > 2 else 20.0
        return record(out, seconds, fps, launch=launch, audio=not no_audio)
    if command == "frames":
        if not args:
            print("usage: python -m tools.capture frames INPUT [OUT] [FPS]", file=sys.stderr)
            return 2
        out = args[1] if len(args) > 1 else "contact.png"
        fps = float(args[2]) if len(args) > 2 else 3.0
        return frames(args[0], out, fps)

    print("usage: python -m tools.capture {screenshot|record|frames} ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
