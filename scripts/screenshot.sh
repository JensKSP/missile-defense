#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Capture one frame of the *already-running* app window to a PNG.
#
# By default it attaches to a running "Missile Defense" window and grabs it, so
# you can shoot the app any time while it's up. Pass --launch to spawn a throwaway
# instance first (used for headless/CI capture when nothing is running).
#
# The app must be an X11 window for ImageMagick `import` to grab it — run it via
# `poe app` (which forces Qt's xcb platform) on this Wayland box.
#
# Usage: scripts/screenshot.sh [--launch] [output.png]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LAUNCH=0
if [[ "${1:-}" == "--launch" ]]; then LAUNCH=1; shift; fi
OUT="${1:-shot.png}"
DISPLAY_ID="${DISPLAY:-:1}"
BIN="build/release/app/md_app"

if [[ "$LAUNCH" == "1" ]]; then
  [[ -x "$BIN" ]] || cmake --build --preset release >/dev/null
  DISPLAY="$DISPLAY_ID" QT_QPA_PLATFORM=xcb "$BIN" &
  trap 'kill "$!" 2>/dev/null || true' EXIT
  for _ in $(seq 1 100); do
    DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" >/dev/null 2>&1 && break
    sleep 0.1
  done
  sleep 0.4
fi

WID=$(DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" 2>/dev/null | awk '/Window id:/{print $4}')
if [[ -z "$WID" ]]; then
  echo "error: no 'Missile Defense' window on DISPLAY=$DISPLAY_ID." >&2
  echo "       Start it with 'poe app', or pass --launch to spawn a throwaway one." >&2
  exit 1
fi
DISPLAY="$DISPLAY_ID" import -window "$WID" "$OUT"
echo "wrote $OUT"
