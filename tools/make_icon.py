# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Generate the application icons from code (no external art assets).

Two emblems, drawn at high resolution and downscaled to the sizes each platform
needs. Run once to (re)generate the committed assets:

    python -m tools.make_icon           # from the repo root

**The game**: a fireball inside a targeting reticle over three cities.

**The trainer**: the same night sky, palette and rounded square, with the arcade
cabinet replaced by an engineering control centre — a screen showing the score
curve the console actually draws, the fireball reduced to the marker riding its
leading edge, and the three cities become the three keys of a control desk. A
sibling rather than a duplicate, because the two are separate products (see the
trainer's own desktop entry) and someone reaching for one in a taskbar full of
windows has to be able to tell which is which at 16 pixels.

Outputs (committed; the build consumes these, so Pillow is dev-only):
    packaging/missile-defense.ico                         (Windows: 16..256)
    packaging/icons/hicolor/<size>x<size>/apps/missile-defense.png  (Linux)
    packaging/missile-defense.icns                        (macOS bundle)
    packaging/icons/hicolor/<size>x<size>/apps/missile-defense-trainer.png
    python/missile_defense/ui/icon.png                    (inside the wheel)

**Two formats for the trainer, not four**, and the missing two are the point:
there is no `.ico` and no `.icns` because on Windows and macOS the console is a
wheel — pip's launcher is the only executable it has and its icon is not ours to
embed, and there is no bundle to put an `.icns` in (CMakeLists.txt refuses
MD_INSTALL_PYTHON_PACKAGE on both). The window icon comes from the PNG in the
wheel on every platform instead, which is one code path rather than three.
"""

from __future__ import annotations

import io
import math
import struct
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
#: Where the trainer's own copy lives, because it is the one icon that has to
#: travel *inside* the Python package: the console is a wheel on Windows and
#: macOS, and nothing beside it is on any search path it could look along.
UI_ICON = ROOT / "python" / "missile_defense" / "ui" / "icon.png"
SIZES = [16, 32, 48, 64, 128, 256]
R = 1024  # master render resolution (downscaled per target size)

#: ICNS members: (four-character type, pixel size). Modern macOS reads PNG
#: payloads directly, so the container is just a header plus one PNG per entry —
#: which means it can be written anywhere, with no `iconutil` (macOS-only) and no
#: Pillow ICNS *save* support (which shells out to `iconutil` on macOS and is
#: absent elsewhere). Sizes repeat because the @2x types carry the same pixels at
#: a different nominal scale, which is what Finder and the Dock expect.
ICNS_MEMBERS: list[tuple[str, int]] = [
    ("icp4", 16),
    ("icp5", 32),
    ("ic07", 128),
    ("ic08", 256),
    ("ic09", 512),
    ("ic10", 1024),  # 512@2x
    ("ic11", 32),  # 16@2x
    ("ic12", 64),  # 32@2x
    ("ic13", 256),  # 128@2x
    ("ic14", 512),  # 256@2x
]

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


def _icns(master: Image.Image) -> bytes:
    """Pack ``master`` into an ICNS container: header, then one PNG per member.

    Each member is ``<4-byte type><big-endian length incl. these 8 bytes><PNG>``;
    the file header is the same shape with the type ``icns`` and the total length.
    """
    body = b""
    for code, size in ICNS_MEMBERS:
        buf = io.BytesIO()
        master.resize((size, size), Image.Resampling.LANCZOS).save(buf, "PNG")
        png = buf.getvalue()
        body += code.encode("ascii") + struct.pack(">I", len(png) + 8) + png
    return b"icns" + struct.pack(">I", len(body) + 8) + body


def _backdrop() -> Image.Image:
    """The night sky both emblems stand on: navy gradient plus starfield.

    Shared so the two icons are visibly the same product. The alpha on the stars
    is not blending — nothing is drawn under them yet, and :func:`main` replaces
    the alpha channel wholesale with the rounded-square mask — it only takes the
    colour down towards the sky it sits in.
    """
    img = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Background: vertical navy gradient, rounded-square.
    for row in range(R):
        d.line([(0, row), (R, row)], fill=(*_lerp(NAVY_TOP, NAVY_BOT, row / (R - 1)), 255))
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
    return img


def _fireball(img: Image.Image, cx: float, cy: float, body: float) -> None:
    """A glowing warhead at ``(cx, cy)``: halo, starburst spikes, hot core.

    Composited rather than drawn straight on, because both layers are
    translucent and `ImageDraw` on an RGBA image *replaces* pixels instead of
    blending them — the halo would punch a hole in the sky rather than glow
    through it.
    """
    halo = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse(
        [cx - body * 1.15, cy - body * 1.15, cx + body * 1.15, cy + body * 1.15],
        fill=(*ORANGE, 200),
    )
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(R * 0.045)))

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


def render_master() -> Image.Image:
    img = _backdrop()
    d = ImageDraw.Draw(img)

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

    _fireball(img, cx, cy, body=0.24 * R)

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


def render_trainer_master() -> Image.Image:
    """The console's emblem: the same sky, an engineering control centre in it.

    Every shape has a counterpart in the game's icon, which is what makes the two
    read as one product: the reticle becomes the bezel of a monitor, the fireball
    shrinks to the marker riding the leading edge of a score curve, and the three
    cities become the three keys of the control desk under it — same positions,
    same teal, one row lower.

    Nothing here is decoration for its own sake. The curve is the shape the
    console's own score chart draws, so the icon says what the application does.
    """
    img = _backdrop()
    d = ImageDraw.Draw(img)
    lw = max(6, R // 42)  # the game's reticle weight, so the two match at 16px

    # ---- The screen -----------------------------------------------------------
    left, right = 0.11 * R, 0.89 * R
    top, bottom = 0.13 * R, 0.61 * R
    d.rounded_rectangle([left, top, right, bottom], radius=R // 20, fill=(6, 10, 22, 255))

    # Grid, plotted inside a margin so the curve never touches the bezel. On its
    # own layer: these are faint on purpose, and drawing translucent lines
    # directly onto an RGBA image would cut holes rather than dim what is under
    # them.
    pad = 0.055 * R
    px0, py0, px1, py1 = left + pad, top + pad, right - pad, bottom - pad
    grid = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid)
    for i in range(1, 4):
        y = py0 + (py1 - py0) * i / 4
        gd.line([(px0, y), (px1, y)], fill=(*TEAL, 50), width=max(2, R // 300))
    for i in range(1, 5):
        x = px0 + (px1 - px0) * i / 5
        gd.line([(x, py0), (x, py1)], fill=(*TEAL, 50), width=max(2, R // 300))
    img.alpha_composite(grid)

    # ---- The score curve ------------------------------------------------------
    # A learning curve: fast early gains flattening towards an asymptote, with a
    # small wobble so it reads as a measurement rather than a logo swoosh. Fixed
    # numbers rather than a random generator — this file must produce the same
    # bytes every time it is run, or a regenerated icon is a diff nobody can review.
    wobble = (0.0, -0.03, 0.02, -0.015, 0.01, -0.02, 0.008, -0.01, 0.0)
    points: list[tuple[float, float]] = []
    for i, jitter in enumerate(wobble):
        t = i / (len(wobble) - 1)
        value = min(1.0, max(0.0, (1.0 - math.exp(-2.7 * t)) / (1.0 - math.exp(-2.7)) + jitter))
        # Stops short of the right edge on purpose: the marker at the end has a
        # halo half its own width again, and a run that is still going should not
        # look like it has hit the side of the screen.
        points.append((px0 + (px1 - px0) * 0.9 * t, py1 - (py1 - py0) * 0.92 * value))

    trace = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    ImageDraw.Draw(trace).line(points, fill=(*CYAN, 255), width=max(6, R // 45), joint="curve")
    # The phosphor glow around it, from the same shape blurred underneath.
    img.alpha_composite(trace.filter(ImageFilter.GaussianBlur(R * 0.02)))
    img.alpha_composite(trace)

    # The fireball, where the run has got to. Small — it is a data point here,
    # not the subject — but the same ramp and spikes as the game's.
    _fireball(img, points[-1][0], points[-1][1], body=0.08 * R)

    # ---- The control desk -----------------------------------------------------
    # The game's cities, at the game's x positions, one row lower: three keys on
    # a desk with a lamp at each end.
    desk = 0.85 * R
    d.line([(0.12 * R, desk), (0.88 * R, desk)], fill=(70, 96, 140, 255), width=max(2, R // 200))
    for i, bx in enumerate((0.30, 0.50, 0.70)):
        bw, bh = 0.075 * R, (0.05 + 0.015 * (i % 2)) * R
        x = bx * R
        d.rounded_rectangle(
            [x - bw / 2, desk - bh, x + bw / 2, desk], radius=R // 90, fill=(*TEAL, 255)
        )
    for lx, colour in ((0.17, CYAN), (0.83, ORANGE)):
        x, r = lx * R, 0.028 * R
        d.ellipse([x - r, desk - 0.038 * R - r, x + r, desk - 0.038 * R + r], fill=(*colour, 255))

    # ---- The bezel ------------------------------------------------------------
    # Drawn last, so it sits on top of the trace and its glow the way a real
    # frame does, and drawn with a gap and an inward tick at the middle of each
    # side — the game's reticle grammar, squared off. That is the whole family
    # resemblance at 16 pixels, where the curve is three stairsteps and the desk
    # is a smudge: both icons are a cyan frame with four gaps and something
    # burning inside.
    frame = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    fd = ImageDraw.Draw(frame)
    fd.rounded_rectangle([left, top, right, bottom], radius=R // 20, outline=(*CYAN, 255), width=lw)
    gap = 0.10 * R
    mx, my = (left + right) / 2, (top + bottom) / 2
    for x0, y0, x1, y1 in (
        (mx - gap, top - lw, mx + gap, top + lw),
        (mx - gap, bottom - lw, mx + gap, bottom + lw),
        (left - lw, my - gap, left + lw, my + gap),
        (right - lw, my - gap, right + lw, my + gap),
    ):
        fd.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))  # replaces, not blends
    tick = 2.2 * lw
    fd.line([(mx, top), (mx, top + tick)], fill=(*CYAN, 255), width=lw)
    fd.line([(mx, bottom), (mx, bottom - tick)], fill=(*CYAN, 255), width=lw)
    fd.line([(left, my), (left + tick, my)], fill=(*CYAN, 255), width=lw)
    fd.line([(right, my), (right - tick, my)], fill=(*CYAN, 255), width=lw)
    img.alpha_composite(frame)
    return img


def _emblem(render: Callable[[], Image.Image]) -> tuple[Image.Image, dict[int, Image.Image]]:
    """A rendered master and its downscales, cut to the rounded square."""
    master = render()
    master.putalpha(_rounded_mask(R, R * 0.18))
    # Image.LANCZOS is a legacy alias; the enum is the typed spelling.
    return master, {s: master.resize((s, s), Image.Resampling.LANCZOS) for s in SIZES}


def _write_hicolor(variants: dict[int, Image.Image], name: str) -> None:
    for s in SIZES:
        out = PKG / "icons" / "hicolor" / f"{s}x{s}" / "apps" / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        variants[s].save(out)
        print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    PKG.mkdir(parents=True, exist_ok=True)

    master, variants = _emblem(render_master)
    ico = PKG / "missile-defense.ico"
    variants[256].save(ico, sizes=[(s, s) for s in SIZES])
    print(f"wrote {ico.relative_to(ROOT)}")
    _write_hicolor(variants, "missile-defense")
    icns = PKG / "missile-defense.icns"
    icns.write_bytes(_icns(master))
    print(f"wrote {icns.relative_to(ROOT)}")

    trainer_master, trainer_variants = _emblem(render_trainer_master)
    _write_hicolor(trainer_variants, "missile-defense-trainer")

    # The copy that travels in the wheel. One size rather than an `.ico`: Qt
    # scales a 256 down cleanly for a title bar, and a PNG needs no image plugin
    # beyond the one Qt always has — on a machine where the console is the only
    # thing installed, a missing plugin would mean no icon and no way to tell why.
    UI_ICON.parent.mkdir(parents=True, exist_ok=True)
    trainer_master.resize((256, 256), Image.Resampling.LANCZOS).save(UI_ICON)
    print(f"wrote {UI_ICON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
