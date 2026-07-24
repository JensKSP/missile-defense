#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Record the running app's window to an H.264 mp4.
#
# Grabs frames with ImageMagick `import -window <id>` in a loop, then encodes.
# (ffmpeg x11grab reads the X root framebuffer, which is BLACK for Xwayland
# windows on a Wayland compositor; per-window `import` works.) The video is
# encoded at the *actual* capture rate so playback runs at real-time speed.
#
# Audio (the game's sound) is captured in parallel from the default PipeWire/
# Pulse sink monitor and muxed in as AAC. Pass --no-audio for a silent clip.
#
# Usage: scripts/record.sh [--launch] [--no-audio] [out.mp4] [seconds] [fps]
set -euo pipefail
export LC_ALL=C # force '.' decimal separator for awk/printf/sleep/ffmpeg
cd "$(git rev-parse --show-toplevel)"

LAUNCH=0
AUDIO=1
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --launch) LAUNCH=1 ;;
    --no-audio) AUDIO=0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done
OUT="${1:-clip.mp4}"
DURATION="${2:-6}"
FPS="${3:-20}"
DISPLAY_ID="${DISPLAY:-:1}"
BIN="build/release/app/md_app"

if [[ "$LAUNCH" == "1" ]]; then
  [[ -x "$BIN" ]] || cmake --build --preset release >/dev/null
  DISPLAY="$DISPLAY_ID" QT_QPA_PLATFORM=xcb "$BIN" &
  app_pid=$!
  trap 'kill "$app_pid" 2>/dev/null || true' EXIT
  for _ in $(seq 1 100); do
    DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" >/dev/null 2>&1 && break
    sleep 0.1
  done
  sleep 0.4
fi

wid=$(DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" 2>/dev/null | awk '/Window id:/{print $4}')
if [[ -z "$wid" ]]; then
  echo "error: no 'Missile Defense' window on DISPLAY=$DISPLAY_ID." >&2
  echo "       Start it with 'poe app', or pass --launch to spawn a throwaway one." >&2
  exit 1
fi

frames_dir="$(mktemp -d)"
audio_wav=""
trap 'rm -rf "$frames_dir" "${audio_wav:-}"; kill "${app_pid:-}" 2>/dev/null || true' EXIT

# Capture audio in parallel (background) for the duration.
if [[ "$AUDIO" == "1" ]]; then
  mon="$(pactl get-default-sink 2>/dev/null).monitor"
  if pactl list short sources 2>/dev/null | grep -q "$mon"; then
    audio_wav="$(mktemp --suffix=.wav)"
    ffmpeg -y -hide_banner -loglevel error -f pulse -i "$mon" -t "$DURATION" "$audio_wav" &
    audio_pid=$!
  fi
fi

# Grab frames until the duration elapses.
start=$(date +%s.%N)
n=0
while awk "BEGIN{exit !(($(date +%s.%N) - $start) < $DURATION)}"; do
  n=$((n + 1))
  DISPLAY="$DISPLAY_ID" import -window "$wid" "$frames_dir/f_$(printf '%06d' "$n").png" 2>/dev/null || break
  sleep "$(awk "BEGIN{print 1.0/$FPS}")"
done
elapsed=$(awk "BEGIN{print $(date +%s.%N) - $start}")
actual_fps=$(awk "BEGIN{printf \"%.3f\", $n / $elapsed}")
[[ -n "${audio_pid:-}" ]] && wait "$audio_pid" 2>/dev/null || true

if [[ -n "$audio_wav" && -s "$audio_wav" ]]; then
  ffmpeg -y -hide_banner -loglevel error \
    -framerate "$actual_fps" -start_number 1 -i "$frames_dir/f_%06d.png" -i "$audio_wav" \
    -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k -shortest "$OUT"
else
  ffmpeg -y -hide_banner -loglevel error \
    -framerate "$actual_fps" -start_number 1 -i "$frames_dir/f_%06d.png" \
    -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$OUT"
fi
echo "wrote $OUT ($n frames, ${actual_fps} fps effective, audio=$([[ -n "$audio_wav" ]] && echo 1 || echo 0))"
