#!/usr/bin/env bash
# Turn a video into a tiled contact-sheet PNG, so a clip can be reviewed at a
# glance as a single image (e.g. by the agent, which reads images not video).
#
# Usage: scripts/frames.sh input.(mp4|webm|...) [out.png] [fps]
set -euo pipefail
IN="${1:?usage: frames.sh input.mp4 [out.png] [fps]}"
OUT="${2:-contact.png}"
FPS="${3:-3}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

ffmpeg -y -hide_banner -loglevel error -i "$IN" -vf "fps=$FPS" "$TMP/f_%04d.png"
montage "$TMP"/f_*.png -tile 5x -geometry 384x+3+3 -background '#111827' "$OUT"
echo "wrote $OUT"
