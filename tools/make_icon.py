# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Generate the application icon from code (no external art assets).

Draws a bold Missile-Command emblem — a fireball inside a targeting reticle over
three cities — at high resolution, then downscales to the sizes each platform
needs. Run once to (re)generate the committed assets:

    python -m tools.make_icon           # from the repo root

Outputs (committed; the build consumes these, so Pillow is dev-only):
    packaging/missile-defense.ico                         (Windows: 16..256)
    packaging/icons/hicolor/<size>x<size>/apps/missile-defense.png  (Linux)
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
SIZES = [16, 32, 48, 64, 128, 256]
R = 1024  # master render resolution (downscaled per target size)

# Palette
NAVY_TOP = (10, 14, 26)
NAVY_BOT = (20, 34, 62)
CORE = (255, 255, 246)
YELLOW = (255, 214, 96)
ORANGE = (255, 138, 42)
RED = (214, 52, 40)
CYAN = (96, 222, 236)
TEAL = (74, 205, 194)


def _lerp(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, ...]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))


def _fire_ramp(f: float) -> tuple[int, int, int]:
    """f in [0,1]: 0 = outer edge (red) -> 1 = white-hot core."""
    if f < 0.34:
        return _lerp(RED, ORANGE, f / 0.34)  # type: ignore[return-value]
    if f < 0.68:
        return _lerp(ORANGE, YELLOW, (f - 0.34) / 0.34)  # type: ignore[return-value]
    return _lerp(YELLOW, CORE, (f - 0.68) / 0.32)  # type: ignore[return-value]


def _rounded_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def render_master() -> Image.Image:
    img = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Background: vertical navy gradient, rounded-square.
    for y in range(R):
        d.line([(0, y), (R, y)], fill=(*_lerp(NAVY_TOP, NAVY_BOT, y / (R - 1)), 255))
    # Faint starfield.
    for sx, sy, sr in [
        (0.16, 0.20, 3),
        (0.82, 0.16, 4),
        (0.30, 0.34, 2),
        (0.72, 0.30, 3),
        (0.12, 0.46, 2),
        (0.88, 0.44, 2),
    ]:
        x, y, r = sx * R, sy * R, sr
        d.ellipse([x - r, y - r, x + r, y + r], fill=(150, 170, 210, 120))

    cx, cy = 0.5 * R, 0.45 * R

    # Cities + ground line near the bottom.
    ground = 0.83 * R
    d.line(
        [(0.14 * R, ground), (0.86 * R, ground)], fill=(70, 96, 140, 255), width=max(2, R // 200)
    )
    for i, bx in enumerate((0.30, 0.50, 0.70)):
        bw, bh = 0.075 * R, (0.05 + 0.015 * (i % 2)) * R
        x = bx * R
        d.rounded_rectangle(
            [x - bw / 2, ground - bh, x + bw / 2, ground], radius=R // 90, fill=(*TEAL, 255)
        )

    body = 0.24 * R

    # Soft outer glow: a blurred orange disk composited under the body.
    halo = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse(
        [cx - body * 1.15, cy - body * 1.15, cx + body * 1.15, cy + body * 1.15],
        fill=(*ORANGE, 200),
    )
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(R * 0.045)))

    # Fireball body + starburst spikes on their own layer, then composited.
    glow = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for k in range(8):  # spikes first, so the body sits on top of their roots
        a = k * math.pi / 4
        r0, r1, w = body * 0.6, body * 1.55, body * 0.15
        tip = (cx + math.cos(a) * r1, cy + math.sin(a) * r1)
        ox, oy = -math.sin(a) * w, math.cos(a) * w
        bx, by = cx + math.cos(a) * r0, cy + math.sin(a) * r0
        gd.polygon([(bx + ox, by + oy), (bx - ox, by - oy), tip], fill=(*YELLOW, 220))
    for i in range(80):  # solid concentric disks: red rim -> white-hot core
        f = i / 79
        rad = body * (1.0 - f)
        gd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(*_fire_ramp(f), 255))
    img.alpha_composite(glow)

    # Targeting reticle: cyan ring with four gaps + inward ticks.
    ring = 0.40 * R
    lw = max(6, R // 42)
    box = [cx - ring, cy - ring, cx + ring, cy + ring]
    for start in (30, 120, 210, 300):
        d.arc(box, start=start, end=start + 60, fill=(*CYAN, 255), width=lw)
    for a in (0, 90, 180, 270):
        rad = math.radians(a)
        x0, y0 = cx + math.cos(rad) * (ring - lw), cy + math.sin(rad) * (ring - lw)
        x1, y1 = cx + math.cos(rad) * (ring - 3.2 * lw), cy + math.sin(rad) * (ring - 3.2 * lw)
        d.line([(x0, y0), (x1, y1)], fill=(*CYAN, 255), width=lw)

    return img


def main() -> None:
    master = render_master()
    mask = _rounded_mask(R, R * 0.18)
    master.putalpha(mask)

    variants = {s: master.resize((s, s), Image.LANCZOS) for s in SIZES}

    PKG.mkdir(parents=True, exist_ok=True)
    ico = PKG / "missile-defense.ico"
    variants[256].save(ico, sizes=[(s, s) for s in SIZES])
    print(f"wrote {ico.relative_to(ROOT)}")

    for s in SIZES:
        out = PKG / "icons" / "hicolor" / f"{s}x{s}" / "apps" / "missile-defense.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        variants[s].save(out)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
