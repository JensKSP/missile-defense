#!/usr/bin/env bash
# Record the *already-running* app window to an H.264 mp4 via ffmpeg x11grab.
#
# By default it attaches to a running "Missile Defense" window and records it, so
# you can capture a clip any time while playing. Pass --launch to spawn a throwaway
# instance first (headless/CI use). The app must be an X11 window (`poe app`).
#
# Usage: scripts/record.sh [--launch] [out.mp4] [seconds] [fps]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LAUNCH=0
if [[ "${1:-}" == "--launch" ]]; then LAUNCH=1; shift; fi
OUT="${1:-clip.mp4}"
DURATION="${2:-6}"
FPS="${3:-30}"
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

if ! DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" >/dev/null 2>&1; then
  echo "error: no 'Missile Defense' window on DISPLAY=$DISPLAY_ID." >&2
  echo "       Start it with 'poe app', or pass --launch to spawn a throwaway one." >&2
  exit 1
fi

eval "$(DISPLAY="$DISPLAY_ID" xwininfo -name "Missile Defense" | awk '
  /Absolute upper-left X/ { x = $4 }
  /Absolute upper-left Y/ { y = $4 }
  /Width/                 { w = $2 }
  /Height/                { h = $2 }
  END { printf "X=%d Y=%d W=%d H=%d\n", x, y, w, h }')"
# H.264 needs even dimensions.
W=$((W - W % 2))
H=$((H - H % 2))

ffmpeg -y -hide_banner -loglevel error \
  -f x11grab -framerate "$FPS" -video_size "${W}x${H}" -i "${DISPLAY_ID}+${X},${Y}" \
  -t "$DURATION" -c:v libx264 -pix_fmt yuv420p -preset veryfast "$OUT"
echo "wrote $OUT (${W}x${H}, ${DURATION}s @ ${FPS}fps)"
