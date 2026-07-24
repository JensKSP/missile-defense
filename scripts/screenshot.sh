#!/usr/bin/env bash
# Launch the app, capture one frame of its window to a PNG, then exit.
#
# Works on an X11 or Xwayland display by forcing Qt's xcb platform so the window
# is a real X window that ImageMagick's `import` can grab. Deterministic golden
# images should instead use an offscreen render path (planned) — this is the
# interactive "let me see it" tool.
#
# Usage: scripts/screenshot.sh [output.png] [seconds-to-let-it-render]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT="${1:-shot.png}"
SETTLE="${2:-0.6}"
DISPLAY_ID="${DISPLAY:-:1}"
BIN="build/release/app/md_app"

[[ -x "$BIN" ]] || cmake --build --preset release >/dev/null

DISPLAY="$DISPLAY_ID" QT_QPA_PLATFORM=xcb "$BIN" &
APP_PID=$!
trap 'kill "$APP_PID" 2>/dev/null || true' EXIT

# Wait for the window to appear (up to ~10s), then let it render a few frames.
for _ in $(seq 1 100); do
  if DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" >/dev/null 2>&1; then break; fi
  sleep 0.1
done
sleep "$SETTLE"

WID=$(DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" 2>/dev/null | awk '/Window id:/{print $4}')
if [[ -z "$WID" ]]; then
  echo "error: Missile Defense window never appeared on DISPLAY=$DISPLAY_ID" >&2
  exit 1
fi
DISPLAY="$DISPLAY_ID" import -window "$WID" "$OUT"
echo "wrote $OUT ($(DISPLAY="$DISPLAY_ID" xwininfo -id "$WID" | awk '/Width|Height/{printf "%s ", $2}'))"
