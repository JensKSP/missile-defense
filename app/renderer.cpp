// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "renderer.hpp"

#include "game_window.hpp"
#include "md/replay/match.hpp"
#include "md/rng.hpp"
#include "md/version.hpp"
#include "projection.hpp"
#include "terrain.hpp"

#include <QVulkanDeviceFunctions>
#include <QVulkanInstance>

// Embedded SPIR-V (generated at build time by glslangValidator --vn).
#include "quad_frag_spv.h"
#include "quad_vert_spv.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace md {

namespace {

// Push constants: world -> clip transform, clip.xy = worldPos * a + b.
struct PushConstants {
    float a[2];
    float b[2];
};

// Two worlds fit in one frame: the split screen draws both sides' entities into
// this single buffer and issues one draw per half, so the budget is per frame
// and not per viewport. The landscape is the bulk of it — a heightfield drawn as
// columns, twice over in a match — which is why the ceiling is this far above
// what the entities alone ever need.
constexpr std::size_t max_instances = 8192;

InstanceData rect(float cx, float cy, float hx, float hy, float r, float g, float b,
                  float a = 1.0f) {
    return InstanceData{cx, cy, hx, hy, 0.0f, r, g, b, a, 0.0f};
}

InstanceData circle(float cx, float cy, float radius, float r, float g, float b, float a = 1.0f) {
    return InstanceData{cx, cy, radius, radius, 0.0f, r, g, b, a, 1.0f};
}

InstanceData glow(float cx, float cy, float radius, float r, float g, float b, float a) {
    return InstanceData{cx, cy, radius, radius, 0.0f, r, g, b, a, 2.0f};
}

// An oriented line segment from `from` to `to`, drawn as a thin rotated rect.
InstanceData line(Vec2 from, Vec2 to, float thick, float r, float g, float b, float a) {
    const float dx = to.x - from.x;
    const float dy = to.y - from.y;
    const float len = std::sqrt((dx * dx) + (dy * dy));
    const float angle = std::atan2(dy, dx);
    return InstanceData{
        (from.x + to.x) * 0.5f, (from.y + to.y) * 0.5f, len * 0.5f, thick, angle, r, g, b, a, 0.0f};
}

// A dangerous-looking fireball: an opaque body + white-hot core + outer glow,
// shifting yellow -> orange -> deep red over its life (`t` in [0, 1]).
void add_fireball(std::vector<InstanceData>& inst, float cx, float cy, float radius, float t) {
    const float fade = t < 0.6f ? 1.0f : (1.0f - ((t - 0.6f) / 0.4f)); // opaque, fades late
    const float body_g = 0.85f - (0.72f * t);                          // yellow -> red
    const float body_b = 0.18f - (0.16f * t);
    const float hot = 1.0f - t; // white-hot when young
    inst.push_back(glow(cx, cy, radius * 1.3f, 1.0f, body_g * 0.5f, body_b * 0.4f, 0.85f * fade));
    inst.push_back(circle(cx, cy, radius * 0.78f, 1.0f, body_g, body_b, 0.98f * fade));
    inst.push_back(
        circle(cx, cy, radius * 0.4f, 1.0f, 0.78f + (0.22f * hot), 0.32f + (0.58f * hot), fade));
}

// A little skyline filling [cx-half_w, cx+half_w]: `towers` vertical buildings of
// stable, pseudo-random heights, standing on the ground at `ground_y`. The
// tallest reaches `top_y` exactly so the silhouette keeps its full height —
// which is what lets the landscape rise and fall under a town without moving the
// skyline into the HUD above it. `lit` adds warm window rows.
void add_building(std::vector<InstanceData>& inst, float cx, float half_w, float ground_y,
                  float top_y, int towers, float r, float g, float b, bool lit) {
    // Founded half a unit *into* the ground rather than balanced on it: the lit
    // rim along the surface has a real thickness, and a tower resting exactly on
    // `ground_y` shows a bright line running under its own foot.
    const float foot = ground_y - 0.5f;
    const float full = std::max(top_y - foot, 0.5f);
    const float slot = (half_w * 2.0f) / static_cast<float>(towers);
    const auto h_frac = [cx](int i) {
        const float s =
            (std::sin((cx * 0.7f) + (static_cast<float>(i) * 2.3999632f)) * 0.5f) + 0.5f;
        return 0.5f + (0.5f * s); // 0.5 .. 1.0 of the full height
    };
    int tallest = 0;
    for (int i = 1; i < towers; ++i) {
        if (h_frac(i) > h_frac(tallest)) {
            tallest = i;
        }
    }
    for (int i = 0; i < towers; ++i) {
        const float frac = (i == tallest) ? 1.0f : h_frac(i);
        const float th = full * frac; // this tower's height
        const float tcx = (cx - half_w) + (slot * (static_cast<float>(i) + 0.5f));
        const float hx = slot * 0.40f; // leaves a thin gap between towers
        const float shade = 0.80f + (0.20f * h_frac(i));
        inst.push_back(
            rect(tcx, foot + (th * 0.5f), hx, th * 0.5f, r * shade, g * shade, b * shade));
        if (lit) {
            const int rows = static_cast<int>((th - 0.7f) / 1.8f);
            for (int rw = 0; rw < rows; ++rw) {
                const float wy = ground_y + 1.1f + (static_cast<float>(rw) * 1.8f);
                inst.push_back(rect(tcx, wy, hx * 0.55f, 0.26f, 1.0f, 0.9f, 0.5f, 0.85f));
            }
        }
    }
}

/// How one layer of ground is filled and how its surface edge catches the light.
struct GroundStyle {
    float step; // column width, world units
    std::array<float, 3> fill;
    std::array<float, 3> rim;
    float rim_thickness; // minimum; a steep column widens it to cover the join
    float grain;         // per-column darkening of the fill, so it is not a slab
};

/// One layer of landscape: a heightfield filled as columns, capped by a lit rim
/// laid along its true surface.
///
/// Columns because the pipeline draws exactly one primitive — an oriented box —
/// and a filled curve has to come from somewhere. The rim is a box rotated onto
/// the segment between two successive surface points, so the silhouette the eye
/// follows is the curve itself and not the staircase holding it up.
template <typename Height>
void add_ground_layer(std::vector<InstanceData>& inst, float world_w, const GroundStyle& style,
                      Height height) {
    const int columns = std::max(1, static_cast<int>(world_w / style.step));
    const float step = world_w / static_cast<float>(columns);
    float h0 = height(0.0f);
    for (int i = 0; i < columns; ++i) {
        const float x0 = step * static_cast<float>(i);
        const float x1 = x0 + step;
        const float h1 = height(x1);
        const float mid = (h0 + h1) * 0.5f; // the column meets the rim halfway up the segment
        const float shade = 1.0f - (style.grain * ((std::sin(x0 * 1.37f) * 0.5f) + 0.5f));
        inst.push_back(rect((x0 + x1) * 0.5f, mid * 0.5f, step * 0.5f, mid * 0.5f,
                            style.fill[0] * shade, style.fill[1] * shade, style.fill[2] * shade));
        // Thickened on a slope by exactly enough to bury the step it spans.
        const float thick = std::max(style.rim_thickness, std::abs(h1 - h0) * 0.6f);
        inst.push_back(line(Vec2{x0, h0}, Vec2{x1, h1}, thick, style.rim[0], style.rim[1],
                            style.rim[2], 1.0f));
        h0 = h1;
    }
}

/// The two layers of ground, back to front: a distant ridge barely off the sky,
/// then the near ground the game is played on.
///
/// Built once and replayed from a buffer every frame — the heightfield never
/// moves, and a match would otherwise pay for it twice per frame.
std::vector<InstanceData> build_ground(const Terrain& terrain, float world_w) {
    std::vector<InstanceData> ground;
    add_ground_layer(ground, world_w,
                     // Distance washes the warmth out of the far ridge and pulls
                     // it towards the sky, which is what puts it *behind* rather
                     // than merely above the ground in front of it.
                     GroundStyle{.step = 4.0f,
                                 .fill = {0.115f, 0.105f, 0.150f},
                                 .rim = {0.170f, 0.152f, 0.192f},
                                 .rim_thickness = 0.35f,
                                 .grain = 0.08f},
                     [](float x) { return Terrain::ridge(x); });
    add_ground_layer(ground, world_w,
                     // Earth, lit by the same cold moon as everything else: a
                     // warm dark body under a tan rim. Brown also buys the one
                     // thing a blue ground could not — it is nobody else's
                     // colour on this field, so the horizon never reads as sky.
                     GroundStyle{.step = 1.0f,
                                 .fill = {0.175f, 0.140f, 0.115f},
                                 .rim = {0.460f, 0.360f, 0.250f},
                                 .rim_thickness = 0.50f,
                                 .grain = 0.12f},
                     [&terrain](float x) { return terrain.height(x); });
    return ground;
}

// Draw an oriented rocket body + two swept-back tail fins from `pos` back by
// `len`, pointing along origin -> pos (falls back to straight down). Returns the
// unit travel direction so the caller can place warhead nose(s) at the tip.
Vec2 add_rocket_body(std::vector<InstanceData>& inst, Vec2 origin, Vec2 pos, float len, float width,
                     float r, float g, float b) {
    Vec2 d{pos.x - origin.x, pos.y - origin.y};
    const float dl = std::sqrt((d.x * d.x) + (d.y * d.y));
    d = (dl > 1.0e-4f) ? Vec2{d.x / dl, d.y / dl} : Vec2{0.0f, -1.0f};
    const Vec2 tail{pos.x - (d.x * len), pos.y - (d.y * len)};
    const Vec2 perp{-d.y, d.x};
    const float fin = width * 2.0f;
    const Vec2 fin_fwd{tail.x + (d.x * width * 2.0f), tail.y + (d.y * width * 2.0f)};
    inst.push_back(line(Vec2{tail.x + (perp.x * fin), tail.y + (perp.y * fin)}, fin_fwd,
                        width * 0.6f, r * 0.8f, g * 0.8f, b * 0.8f, 1.0f));
    inst.push_back(line(Vec2{tail.x - (perp.x * fin), tail.y - (perp.y * fin)}, fin_fwd,
                        width * 0.6f, r * 0.8f, g * 0.8f, b * 0.8f, 1.0f));
    inst.push_back(line(tail, pos, width, r, g, b, 1.0f)); // body
    return d;
}

// A plain ICBM: a rocket body with a single white-hot warhead nose.
void add_missile(std::vector<InstanceData>& inst, Vec2 origin, Vec2 pos, float len, float width,
                 float r, float g, float b) {
    add_rocket_body(inst, origin, pos, len, width, r, g, b);
    inst.push_back(circle(pos.x, pos.y, width * 1.15f, 1.0f, 0.95f, 0.85f)); // warhead nose
}

// A MIRV: a heavier rocket carrying a cluster of warheads (it splits into
// several), shown as three warhead tips fanned across the nose.
void add_mirv(std::vector<InstanceData>& inst, Vec2 origin, Vec2 pos, float len, float width,
              float r, float g, float b) {
    const Vec2 d = add_rocket_body(inst, origin, pos, len, width, r, g, b);
    const Vec2 perp{-d.y, d.x};
    const Vec2 shoulder{pos.x - (d.x * width * 0.7f), pos.y - (d.y * width * 0.7f)};
    const float off = width * 1.35f;
    inst.push_back(circle(pos.x, pos.y, width * 0.9f, 1.0f, 0.9f, 0.85f)); // lead warhead
    inst.push_back(circle(shoulder.x + (perp.x * off), shoulder.y + (perp.y * off), width * 0.72f,
                          1.0f, 0.85f, 0.95f));
    inst.push_back(circle(shoulder.x - (perp.x * off), shoulder.y - (perp.y * off), width * 0.72f,
                          1.0f, 0.85f, 0.95f));
}

// A warhead / re-entry vehicle: a compact, finless, pointed projectile — a
// short tapered body with a rounded base and a white-hot nose tip + aura. Used
// for the children a MIRV splits into (kept purple to show their lineage).
void add_warhead(std::vector<InstanceData>& inst, Vec2 origin, Vec2 pos, float len, float width,
                 float r, float g, float b) {
    Vec2 d{pos.x - origin.x, pos.y - origin.y};
    const float dl = std::sqrt((d.x * d.x) + (d.y * d.y));
    d = (dl > 1.0e-4f) ? Vec2{d.x / dl, d.y / dl} : Vec2{0.0f, -1.0f};
    const Vec2 tail{pos.x - (d.x * len), pos.y - (d.y * len)};
    inst.push_back(glow(pos.x, pos.y, width * 2.6f, r, g, b, 0.5f));                // aura
    inst.push_back(line(tail, pos, width, r, g, b, 1.0f));                          // body
    inst.push_back(circle(tail.x, tail.y, width * 1.05f, r * 0.85f, g * 0.85f, b)); // base
    inst.push_back(circle(pos.x, pos.y, width * 0.72f, 1.0f, 0.95f, 0.9f));         // hot nose
}

// A smart bomb: a maneuvering decoy, drawn as a spinning diamond pod with a
// bright counter-rotating core and an aura — visually agile, not ballistic.
void add_smartbomb(std::vector<InstanceData>& inst, Vec2 pos, float radius, float spin, float r,
                   float g, float b) {
    inst.push_back(glow(pos.x, pos.y, radius * 2.4f, r, g, b, 0.5f));
    InstanceData body = rect(pos.x, pos.y, radius, radius, r, g, b);
    body.angle = spin; // rotate the square into a spinning diamond
    inst.push_back(body);
    InstanceData core = rect(pos.x, pos.y, radius * 0.5f, radius * 0.5f, 1.0f, 1.0f, 0.9f);
    core.angle = -spin * 1.4f;
    inst.push_back(core);
}

// A 3x5 pixel font for digits, drawn as small quads (no font textures). Each
// row's low 3 bits are the lit pixels; most-significant bit = leftmost column.
constexpr std::array<std::array<std::uint8_t, 5>, 10> digit_font = {{
    {{7, 5, 5, 5, 7}},
    {{2, 6, 2, 2, 7}},
    {{7, 1, 7, 4, 7}},
    {{7, 1, 7, 1, 7}},
    {{5, 5, 7, 1, 1}},
    {{7, 4, 7, 1, 7}},
    {{7, 4, 7, 5, 7}},
    {{7, 1, 2, 2, 2}},
    {{7, 5, 7, 5, 7}},
    {{7, 5, 7, 1, 7}},
}};

void draw_glyph(std::vector<InstanceData>& inst, const std::array<std::uint8_t, 5>& rows,
                float left_x, float top_y, float px, float r, float g, float b) {
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            if ((rows[static_cast<std::size_t>(row)] & (1u << (2 - col))) != 0u) {
                inst.push_back(rect(left_x + ((static_cast<float>(col) + 0.5f) * px),
                                    top_y - ((static_cast<float>(row) + 0.5f) * px), px * 0.42f,
                                    px * 0.42f, r, g, b));
            }
        }
    }
}

// Draw an unsigned integer as pixel-font digits. right_align anchors to x's right.
void draw_number(std::vector<InstanceData>& inst, std::uint32_t value, float x, float top_y,
                 float px, float r, float g, float b, bool right_align) {
    std::array<int, 12> digits{};
    int count = 0;
    if (value == 0) {
        digits[static_cast<std::size_t>(count++)] = 0;
    } else {
        std::uint32_t v = value;
        while (v > 0 && count < 12) {
            digits[static_cast<std::size_t>(count++)] = static_cast<int>(v % 10);
            v /= 10;
        }
    }
    const float advance = px * 4.0f; // 3 columns + 1 gap
    const float start_x = right_align ? (x - (static_cast<float>(count) * advance)) : x;
    for (int i = 0; i < count; ++i) {
        const auto d = static_cast<std::size_t>(digits[static_cast<std::size_t>(count - 1 - i)]);
        draw_glyph(inst, digit_font[d], start_x + (static_cast<float>(i) * advance), top_y, px, r,
                   g, b);
    }
}

// 3x5 uppercase letters A-Z (same encoding as digit_font). Cramped but readable.
constexpr std::array<std::array<std::uint8_t, 5>, 26> letter_font = {{
    {{2, 5, 7, 5, 5}}, {{6, 5, 6, 5, 6}}, {{3, 4, 4, 4, 3}}, {{6, 5, 5, 5, 6}}, {{7, 4, 6, 4, 7}},
    {{7, 4, 6, 4, 4}}, {{3, 4, 5, 5, 3}}, {{5, 5, 7, 5, 5}}, {{7, 2, 2, 2, 7}}, {{1, 1, 1, 5, 2}},
    {{5, 6, 4, 6, 5}}, {{4, 4, 4, 4, 7}}, {{5, 7, 7, 5, 5}}, {{5, 7, 7, 7, 5}}, {{7, 5, 5, 5, 7}},
    {{6, 5, 6, 4, 4}}, {{7, 5, 5, 7, 3}}, {{6, 5, 6, 5, 5}}, {{3, 4, 2, 1, 6}}, {{7, 2, 2, 2, 2}},
    {{5, 5, 5, 5, 7}}, {{5, 5, 5, 5, 2}}, {{5, 5, 7, 7, 5}}, {{5, 5, 2, 5, 5}}, {{5, 5, 2, 2, 2}},
    {{7, 1, 2, 4, 7}},
}};

std::array<std::uint8_t, 5> glyph_for(char c) {
    if (c >= '0' && c <= '9') {
        return digit_font[static_cast<std::size_t>(c - '0')];
    }
    if (c >= 'A' && c <= 'Z') {
        return letter_font[static_cast<std::size_t>(c - 'A')];
    }
    if (c == '.') {
        return {{0, 0, 0, 0, 2}}; // period: a single bottom-centre pixel (for "0.1.0")
    }
    return {{0, 0, 0, 0, 0}};
}

// Draw a string with the pixel font. `centered` treats x as the horizontal centre.
void draw_text(std::vector<InstanceData>& inst, std::string_view text, float x, float top_y,
               float px, float r, float g, float b, bool centered) {
    const float advance = px * 4.0f;
    float cursor = centered ? (x - (static_cast<float>(text.size()) * advance * 0.5f)) : x;
    for (const char c : text) {
        if (c != ' ') {
            draw_glyph(inst, glyph_for(c), cursor, top_y, px, r, g, b);
        }
        cursor += advance;
    }
}

/// The screen palette, named once.
///
/// These are not new colours: they are the ones the game already speaks in — the
/// heading every screen wears, the amber a selected item turns, the blue the
/// byline recedes to, the grey a footer sits in. Naming them is what lets a
/// screen be laid out by *role* instead of by three fresh floats, and it is why
/// HELP and ABOUT can now be structured without inventing a second vocabulary.
namespace ink {
constexpr std::array<float, 3> heading{0.85f, 0.92f, 1.00f}; // a screen title
constexpr std::array<float, 3> accent{0.95f, 0.75f, 0.25f};  // the thing in hand
constexpr std::array<float, 3> body{0.78f, 0.83f, 0.90f};    // ordinary copy
constexpr std::array<float, 3> muted{0.55f, 0.60f, 0.70f};   // a step behind body
constexpr std::array<float, 3> recede{0.34f, 0.45f, 0.70f};  // small print
constexpr std::array<float, 3> footer{0.60f, 0.65f, 0.70f};  // PRESS ENTER
constexpr std::array<float, 3> city{0.35f, 0.70f, 0.98f};    // the cities' own blue
constexpr std::array<float, 3> gold{1.00f, 0.85f, 0.35f};    // the best score there is
} // namespace ink

/// `draw_text` in one of the palette's colours.
void draw_text(std::vector<InstanceData>& inst, std::string_view text, float x, float top_y,
               float px, const std::array<float, 3>& colour, bool centered) {
    draw_text(inst, text, x, top_y, px, colour[0], colour[1], colour[2], centered);
}

/// One row of a key table: what you press, and what it does.
struct Binding {
    std::string_view key;
    std::string_view action;
};

/// A two-column table of `KEY  ACTION` rows, centred as a block.
///
/// The key takes the colour a selected menu item has and the action the colour
/// of ordinary copy — the player has already learned that pair everywhere else,
/// so the table needs no legend. Right-aligning the keys against a shared split
/// is what makes it read as a table rather than as four sentences: the actions
/// start in one place, and the eye can run down either column on its own.
void draw_bindings(std::vector<InstanceData>& inst, std::span<const Binding> rows, float centre,
                   float top_y, float line_gap, float px) {
    const float advance = px * 4.0f;
    std::size_t key_chars = 0;
    std::size_t action_chars = 0;
    for (const Binding& row : rows) {
        key_chars = std::max(key_chars, row.key.size());
        action_chars = std::max(action_chars, row.action.size());
    }
    constexpr std::size_t gap_chars = 3;
    const float width = static_cast<float>(key_chars + gap_chars + action_chars) * advance;
    const float split = (centre - (width * 0.5f)) + (static_cast<float>(key_chars) * advance);
    float y = top_y;
    for (const Binding& row : rows) {
        draw_text(inst, row.key, split - (static_cast<float>(row.key.size()) * advance), y, px,
                  ink::accent, false);
        draw_text(inst, row.action, split + (static_cast<float>(gap_chars) * advance), y, px,
                  ink::body, false);
        y -= line_gap;
    }
}

VkShaderModule make_shader(QVulkanDeviceFunctions* dev, VkDevice device, const uint32_t* code,
                           std::size_t bytes) {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = bytes;
    info.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    dev->vkCreateShaderModule(device, &info, nullptr, &module);
    return module;
}

void alloc_buffer(QVulkanWindow* win, QVulkanDeviceFunctions* dev, VkDeviceSize size,
                  VkBuffer* buffer, VkDeviceMemory* memory) {
    VkDevice device = win->device();

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    dev->vkCreateBuffer(device, &bi, nullptr, buffer);

    VkMemoryRequirements req{};
    dev->vkGetBufferMemoryRequirements(device, *buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = win->hostVisibleMemoryIndex();
    dev->vkAllocateMemory(device, &ai, nullptr, memory);
    dev->vkBindBufferMemory(device, *buffer, *memory, 0);
}

/// A viewport covering a rectangle of the swapchain image, in pixels.
VkViewport rect_viewport(float x, float width, float height) {
    VkViewport viewport{};
    viewport.x = x;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    return viewport;
}

/// The whole window — every frame that is not a match.
VkViewport whole(QSize size) {
    return rect_viewport(0.0f, static_cast<float>(size.width()), static_cast<float>(size.height()));
}

/// How many digits `draw_number` will draw — the pixel font is fixed-width, so
/// this is all it takes to centre a number the way `draw_text` centres a word.
int digit_count(std::uint32_t value) {
    int digits = 1;
    for (std::uint32_t v = value / 10; v > 0; v /= 10) {
        ++digits;
    }
    return digits;
}

/// How wide `draw_stat` will be, so two of them can be laid out side by side.
float stat_width(std::string_view label, std::uint32_t value, float px) {
    const float advance = px * 4.0f;
    return static_cast<float>(label.size() + 1 + static_cast<std::size_t>(digit_count(value))) *
           advance; // one blank between caption and number
}

/// A caption and its number, centred together: `WAVE 7`.
///
/// One call rather than two placements, because the two halves of a match have
/// to line up exactly — eyeballing an offset per label is how a scoreboard ends
/// up a pixel out of step with the one beside it.
void draw_stat(std::vector<InstanceData>& inst, std::string_view label, std::uint32_t value,
               float centre, float top_y, float px, float r, float g, float b) {
    const float half = stat_width(label, value, px) * 0.5f;
    draw_text(inst, label, centre - half, top_y, px, r, g, b, false);
    draw_number(inst, value, centre + half, top_y, px, r, g, b, true);
}

/// Fold to upper case for the pixel font, which has no lower case — otherwise a
/// model name like "Amber Anvil" renders as two capitals and nine blanks.
std::string shout(std::string_view text) {
    std::string out{text};
    std::ranges::transform(out, out.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return out;
}

/// The field a game is played on: the landscape, the cities, the launch bases.
///
/// Drawn in every state — the menus sit over a live skyline — and once per side
/// in a match, which is why it is a function rather than a stretch of
/// `startNextFrame`. The ground arrives prebuilt: it is the same every frame.
void build_backdrop(const Sim& sim, const Terrain& terrain, std::span<const InstanceData> ground,
                    std::vector<InstanceData>& inst) {
    inst.insert(inst.end(), ground.begin(), ground.end());
    for (const auto& city : sim.cities()) {
        const float ground_y = terrain.height(city.pos.x);
        if (city.alive) {
            add_building(inst, city.pos.x, 7.0f, ground_y, 10.0f, 5, 0.25f, 0.62f, 0.95f,
                         true); // skyscrapers
        } else {
            inst.push_back(
                rect(city.pos.x, ground_y + 1.0f, 6.0f, 1.2f, 0.25f, 0.22f, 0.21f)); // rubble
        }
    }
    for (const auto& base : sim.bases()) {
        const float ground_y = terrain.height(base.pos.x);
        if (!base.alive) {
            inst.push_back(
                rect(base.pos.x, ground_y + 1.0f, 6.0f, 1.2f, 0.25f, 0.22f, 0.21f)); // rubble
            continue;
        }
        const bool empty = base.ammo == 0;
        add_building(inst, base.pos.x, 6.0f, ground_y, 12.0f, 3, empty ? 0.42f : 0.85f,
                     empty ? 0.38f : 0.58f, empty ? 0.32f : 0.24f, !empty); // launch towers
    }
}

/// Everything currently in the air: threats, interceptors, blasts, explosions.
///
/// `tsec` is wall-clock animation time, not simulation time — it drives the
/// smart bomb's spin and pulse, which are decoration. Both sides of a match get
/// the same value, so the two screens animate together.
void build_entities(const Sim& sim, std::vector<InstanceData>& inst, float tsec) {
    for (const auto& threat : sim.threats()) {
        if (threat.type == ThreatType::Mirv) { // splitter — purple, multi-warhead
            inst.push_back(line(threat.origin, threat.pos, 0.4f, 0.6f, 0.3f, 0.85f, 0.5f));
            inst.push_back(glow(threat.pos.x, threat.pos.y, 4.5f, 0.8f, 0.4f, 1.0f, 0.6f));
            add_mirv(inst, threat.origin, threat.pos, 5.5f, 1.0f, 0.8f, 0.45f, 1.0f);
        } else if (threat.type == ThreatType::SmartBomb) { // dodger — green, spinning pod
            inst.push_back(line(threat.origin, threat.pos, 0.4f, 0.3f, 0.8f, 0.4f, 0.4f));
            const float pulse = 1.7f + (0.25f * std::sin((tsec * 6.0f) + threat.pos.x));
            add_smartbomb(inst, threat.pos, pulse, tsec * 3.0f, 0.35f, 0.95f, 0.5f);
        } else if (threat.type == ThreatType::Warhead) { // MIRV child — purple warhead
            inst.push_back(line(threat.origin, threat.pos, 0.3f, 0.6f, 0.35f, 0.85f, 0.4f));
            add_warhead(inst, threat.origin, threat.pos, 2.8f, 1.0f, 0.82f, 0.45f, 1.0f);
        } else { // ICBM — red rocket
            inst.push_back(line(threat.origin, threat.pos, 0.35f, 0.85f, 0.25f, 0.20f, 0.45f));
            inst.push_back(glow(threat.pos.x, threat.pos.y, 4.0f, 0.95f, 0.35f, 0.30f, 0.55f));
            add_missile(inst, threat.origin, threat.pos, 5.0f, 0.8f, 0.95f, 0.4f, 0.35f);
        }
    }
    for (const auto& it : sim.interceptors()) {
        inst.push_back(line(it.origin, it.pos, 0.3f, 0.6f, 0.85f, 1.0f, 0.5f));
        inst.push_back(circle(it.pos.x, it.pos.y, 0.9f, 0.9f, 0.97f, 1.0f));
    }
    for (const auto& blast : sim.blasts()) {
        add_fireball(inst, blast.center.x, blast.center.y, blast.radius,
                     blast.age / sim.config().blast_lifetime);
    }
    for (const auto& explosion : sim.explosions()) {
        add_fireball(inst, explosion.center.x, explosion.center.y, explosion.radius,
                     explosion.age / sim.config().explosion_lifetime);
    }
}

} // namespace

// Not noexcept: build_stars() fills a std::vector, so construction can throw on
// allocation failure. Only the simulation hot path promises never to throw.
Renderer::Renderer(GameWindow* window) : window_{window} {
    build_stars();
    build_landscape();
}

// Raise the landscape once, from where the installations actually stand. Both
// halves of a match play on the same layout, so one heightfield serves both.
void Renderer::build_landscape() {
    const Sim& sim = window_->sim();
    std::vector<float> city_x;
    city_x.reserve(sim.cities().size());
    for (const auto& city : sim.cities()) {
        city_x.push_back(city.pos.x);
    }
    std::vector<float> base_x;
    base_x.reserve(sim.bases().size());
    for (const auto& base : sim.bases()) {
        base_x.push_back(base.pos.x);
    }
    terrain_ = Terrain{sim.config().world_width, city_x, base_x};
    ground_ = build_ground(terrain_, sim.config().world_width);
}

// Scatter a fixed set of stars across the upper sky, each with its own dim base
// brightness and slow twinkle. Seeded so the field is stable run to run.
void Renderer::build_stars() {
    const float w = window_->sim().config().world_width;
    const float h = window_->sim().config().world_height;
    Pcg32 rng{1337};
    stars_.reserve(80);
    for (int i = 0; i < 80; ++i) {
        stars_.push_back(Star{.x = rng.uniform(0.0f, w),
                              .y = rng.uniform(h * 0.16f, h * 0.98f), // above the skyline
                              .base = rng.uniform(0.10f, 0.40f),      // dim, not distracting
                              .phase = rng.uniform(0.0f, 6.2831853f),
                              .speed = rng.uniform(0.4f, 1.8f), // slow twinkle
                              .size = rng.uniform(0.35f, 0.8f)});
    }
}

void Renderer::initResources() {
    dev_ = window_->vulkanInstance()->deviceFunctions(window_->device());
    createBuffers();
    createPipeline();
}

void Renderer::createBuffers() {
    // Unit quad (two triangles), corners in [-0.5, 0.5].
    static const std::array<float, 12> quad = {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f,  0.5f,
                                               -0.5f, -0.5f, 0.5f, 0.5f,  -0.5f, 0.5f};
    alloc_buffer(window_, dev_, sizeof(quad), &quad_buf_, &quad_mem_);
    void* mapped = nullptr;
    dev_->vkMapMemory(window_->device(), quad_mem_, 0, sizeof(quad), 0, &mapped);
    std::memcpy(mapped, quad.data(), sizeof(quad));
    dev_->vkUnmapMemory(window_->device(), quad_mem_);

    // One persistently-mapped instance buffer per frame in flight.
    const int frames = window_->concurrentFrameCount();
    const VkDeviceSize inst_bytes = max_instances * sizeof(InstanceData);
    instance_bufs_.resize(static_cast<std::size_t>(frames));
    instance_mems_.resize(static_cast<std::size_t>(frames));
    instance_mapped_.resize(static_cast<std::size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        const auto f = static_cast<std::size_t>(i);
        alloc_buffer(window_, dev_, inst_bytes, &instance_bufs_[f], &instance_mems_[f]);
        dev_->vkMapMemory(window_->device(), instance_mems_[f], 0, inst_bytes, 0,
                          &instance_mapped_[f]);
    }
}

void Renderer::createPipeline() {
    VkDevice device = window_->device();

    VkShaderModule vert = make_shader(dev_, device, quad_vert_spv, sizeof(quad_vert_spv));
    VkShaderModule frag = make_shader(dev_, device, quad_frag_spv, sizeof(quad_frag_spv));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset = 0;
    push.size = static_cast<std::uint32_t>(sizeof(PushConstants));

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    dev_->vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_);

    // Built as two separate structs rather than zero-initialising an array of
    // them: each stage names its shader-stage bit explicitly, which is clearer and
    // keeps the enum out of a default-initialised aggregate.
    VkPipelineShaderStageCreateInfo vert_stage{};
    vert_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage.module = vert;
    vert_stage.pName = "main";
    VkPipelineShaderStageCreateInfo frag_stage{};
    frag_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage.module = frag;
    frag_stage.pName = "main";
    const std::array<VkPipelineShaderStageCreateInfo, 2> stages{vert_stage, frag_stage};

    std::array<VkVertexInputBindingDescription, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].stride = static_cast<std::uint32_t>(sizeof(float) * 2);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = static_cast<std::uint32_t>(sizeof(InstanceData));
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 6> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    attrs[1] = {1, 1, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, cx))};
    attrs[2] = {2, 1, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, hx))};
    attrs[3] = {3, 1, VK_FORMAT_R32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, angle))};
    attrs[4] = {4, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, r))};
    attrs[5] = {5, 1, VK_FORMAT_R32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(InstanceData, shape))};

    VkPipelineVertexInputStateCreateInfo vin{};
    vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vin.vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size());
    vin.pVertexBindingDescriptions = bindings.data();
    vin.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
    vin.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = window_->sampleCountFlagBits();

    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dss.depthTestEnable = VK_FALSE;
    dss.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;

    std::array<VkDynamicState, 2> dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<std::uint32_t>(dynamics.size());
    dyn.pDynamicStates = dynamics.data();

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vin;
    info.pInputAssemblyState = &ia;
    info.pViewportState = &vp;
    info.pRasterizationState = &rs;
    info.pMultisampleState = &ms;
    info.pDepthStencilState = &dss;
    info.pColorBlendState = &cb;
    info.pDynamicState = &dyn;
    info.layout = pipeline_layout_;
    info.renderPass = window_->defaultRenderPass();
    info.subpass = 0;
    dev_->vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_);

    dev_->vkDestroyShaderModule(device, vert, nullptr);
    dev_->vkDestroyShaderModule(device, frag, nullptr);
}

void Renderer::startNextFrame() {
    window_->advance();

    const QSize sz = window_->swapChainImageSize();

    // A match is its own screen: two recordings, two viewports, one clock. It
    // shares the instance buffer with the single-sim path below and nothing else,
    // so it returns rather than threading a "which sim?" question through 400
    // lines of menu and HUD drawing.
    if (const replay::MatchPlayer* match = window_->match(); match != nullptr) {
        draw_match(*match, sz);
        return;
    }

    const Sim& sim = window_->sim();
    const float world_w = sim.config().world_width;
    const float world_h = sim.config().world_height;

    const auto state = window_->state();
    const bool playing = state == GameWindow::State::Playing;
    const bool game_over = state == GameWindow::State::GameOver;
    const bool paused_menu = state == GameWindow::State::Menu && window_->in_progress();
    // Draw the game (and freeze it) while playing, on game over, and behind the
    // pause menu (Menu state with a game in progress).
    const bool show_game = playing || game_over || paused_menu;
    const float cx = world_w * 0.5f;

    std::vector<InstanceData> inst;
    inst.reserve(max_instances);

    // Twinkling starfield, behind everything (drawn in every state).
    const float tsec =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - start_).count();
    for (const auto& star : stars_) {
        const float tw = 0.55f + (0.45f * std::sin(star.phase + (star.speed * tsec)));
        inst.push_back(glow(star.x, star.y, star.size, 0.85f, 0.9f, 1.0f, star.base * tw));
    }

    // Field backdrop (drawn in every state).
    build_backdrop(sim, terrain_, ground_, inst);

    if (show_game) {
        build_entities(sim, inst, tsec);
        if (!game_over) { // HUD: score / wave / ammo — hidden on the game-over screen
            const float digit_px = world_h * 0.013f;
            const float hud_top = world_h * 0.97f;
            draw_number(inst, static_cast<std::uint32_t>(sim.score() < 0 ? 0 : sim.score()),
                        world_w * 0.02f, hud_top, digit_px, 1.0f, 1.0f, 1.0f, false);
            draw_number(inst, sim.wave(), world_w * 0.98f, hud_top, digit_px, 0.95f, 0.75f, 0.30f,
                        true);
            for (const auto& base : sim.bases()) {
                if (!base.alive) {
                    continue;
                }
                const float spacing = 1.6f;
                const float base_x0 =
                    base.pos.x - ((static_cast<float>(base.ammo) - 1.0f) * spacing * 0.5f);
                for (std::uint32_t k = 0; k < base.ammo; ++k) {
                    inst.push_back(rect(base_x0 + (static_cast<float>(k) * spacing), 14.0f, 0.55f,
                                        0.55f, 0.40f, 0.90f, 0.55f));
                }
            }
        }
        if (playing) {
            // The crosshair is sim state (speed-capped), so draw where it actually
            // is — that is where a shot detonates, not where the mouse points.
            const Vec2 aim = sim.crosshair(); // four arms + a centre dot
            constexpr float arm = 3.0f;       // arm length
            constexpr float gap = 1.3f;       // centre gap
            constexpr float th = 0.3f;        // line thickness
            const float off = gap + (arm * 0.5f);
            inst.push_back(rect(aim.x - off, aim.y, arm * 0.5f, th, 1.0f, 1.0f, 1.0f)); // left
            inst.push_back(rect(aim.x + off, aim.y, arm * 0.5f, th, 1.0f, 1.0f, 1.0f)); // right
            inst.push_back(rect(aim.x, aim.y - off, th, arm * 0.5f, 1.0f, 1.0f, 1.0f)); // down
            inst.push_back(rect(aim.x, aim.y + off, th, arm * 0.5f, 1.0f, 1.0f, 1.0f)); // up
            inst.push_back(circle(aim.x, aim.y, 0.35f, 1.0f, 1.0f, 1.0f)); // centre dot
        }
        // The spectator key hints, shared by replay and watch mode. They had the
        // menu footer's problem — a dim green sitting inside the skyline, darker
        // than the buildings it crossed — but not its constraint: there is no menu
        // above them during play, so they simply move up until they clear the
        // 12-unit launch towers (a glyph hangs 5*px below this top_y). Left cool
        // rather than the menu's red, which would read as an alert mid-game.
        const float hint_y = world_h * 0.115f;
        constexpr float hint_r = 0.62f;
        constexpr float hint_g = 0.68f;
        constexpr float hint_b = 0.76f;

        // Replay: say what is being watched, and how far through it is.
        if (playing && window_->replaying()) {
            // The pixel font has no lower case, so fold the label — otherwise a
            // trainer-written tag like "update-1200" renders as half blanks.
            std::string label{window_->replay_label()};
            std::ranges::transform(label, label.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            draw_text(inst, label.empty() ? std::string{"REPLAY"} : label, cx, world_h * 0.955f,
                      world_h * 0.012f, 0.55f, 0.80f, 0.95f, true);
            // A thin progress bar — a recording has a known length, unlike a live
            // game. Below 0.895, where the label's glyphs stop hanging.
            constexpr float bar_w = 0.18f;
            constexpr float bar_y = 0.878f;
            const float filled = bar_w * window_->replay_progress();
            inst.push_back(rect(cx, world_h * bar_y, world_w * bar_w * 0.5f, world_h * 0.002f,
                                0.22f, 0.26f, 0.34f));
            if (filled > 0.0f) {
                inst.push_back(rect(cx - (world_w * (bar_w - filled) * 0.5f), world_h * bar_y,
                                    world_w * filled * 0.5f, world_h * 0.002f, 0.55f, 0.80f,
                                    0.95f));
            }
            draw_text(inst, "T TAKE OVER   ARROWS SEEK   R RESTART", cx, hint_y, world_h * 0.008f,
                      hint_r, hint_g, hint_b, true);
        }
        // Watch mode: say plainly *which* agent is at the controls, and how to
        // take over. The name and not "AI PLAYING": watching two agents and
        // being unable to tell which one is on screen makes the whole feature
        // nearly useless, and a path would not help — `policy-best.pt` says
        // nothing about which run produced it. `driver_name()` comes from the
        // model's own `.mdp` (docs/ROADMAP.md, M8; docs/API.md §7).
        if (playing && window_->ai_driving()) {
            // Upper-cased here and not at the source: the report wants the
            // model's name as its `.mdp` spells it, and this font has no lower
            // case at all — `glyph_for` returns a blank for one, so "Parity"
            // would draw as a P and five spaces.
            std::string banner{window_->driver_name()};
            std::ranges::transform(banner, banner.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            banner = banner.empty() ? std::string{"AI PLAYING"} : banner + " PLAYING";
            draw_text(inst, banner, cx, world_h * 0.955f, world_h * 0.012f, 0.45f, 0.95f, 0.65f,
                      true);
            draw_text(inst, "T TAKE OVER    BRACKETS SPEED", cx, hint_y, world_h * 0.008f, hint_r,
                      hint_g, hint_b, true);
        }
        // Shared by both spectator modes. Always shown, including 1X — otherwise
        // pressing the speed keys at normal speed looks like nothing happened.
        // Parked in the top-right corner under the wave counter: centred it
        // collided with the banner above, whose glyphs hang down to 0.895.
        if (playing && (window_->ai_driving() || window_->replaying())) {
            const int speed = window_->speed();
            const bool fast = speed > 1;
            const std::string label = "SPEED " + std::to_string(speed) + "X";
            const float speed_px = world_h * 0.008f;
            const float speed_w = static_cast<float>(label.size()) * speed_px * 4.0f;
            draw_text(inst, label, (world_w * 0.98f) - speed_w, world_h * 0.885f, speed_px,
                      fast ? 0.95f : 0.45f, fast ? 0.80f : 0.55f, fast ? 0.35f : 0.50f, false);
        }
    }

    // Dim the field behind every screen that is mostly text.
    //
    // The front menu is the exception and is left alone: its lines all sit above
    // the tallest thing on the ground (a battery's towers reach 12, and the lower
    // of its two hints hangs to 13.6), so it gets to keep the live skyline that
    // makes it the game's front page. Every other screen here runs its copy down
    // to the foot of the frame — ABOUT's last line lands at 0.1 world units, in
    // the dirt — and a 3x5 glyph crossing a lit hillside or a row of windows is
    // not readable at any colour. Scrimming is what the game-over screen already
    // did; these screens have the same problem and now get the same answer.
    const bool text_screen =
        state == GameWindow::State::Help || state == GameWindow::State::About ||
        state == GameWindow::State::Options || state == GameWindow::State::Highscores ||
        state == GameWindow::State::Replays || state == GameWindow::State::Watch ||
        state == GameWindow::State::EnterScore;
    if (game_over || paused_menu || text_screen) {
        // A text screen ghosts the field almost out: ABOUT and HIGHSCORES both
        // run their copy right down into the ground and neither has a spare line
        // to give, so the backdrop is what has to yield. The other two keep more
        // of the field, because on those the field is the point — the pause menu
        // has a game the player wants to keep an eye on, and the game-over screen
        // is there to show what is left of it.
        float dim = 0.88f;
        if (game_over) {
            dim = 0.74f;
        } else if (paused_menu) {
            dim = 0.55f;
        }
        inst.push_back(
            rect(cx, world_h * 0.5f, world_w * 0.5f, world_h * 0.5f, 0.02f, 0.02f, 0.05f, dim));
    }

    if (state == GameWindow::State::Menu) {
        draw_text(inst, "MISSILE DEFENSE", cx, world_h * 0.90f, world_h * 0.022f, ink::heading,
                  true);
        // Deeper and bluer than the title above it, so the byline recedes instead
        // of competing — near enough the city blue (0.25, 0.62, 0.95) to belong to
        // the same palette, dark enough to sit a step behind the name.
        draw_text(inst, "BY JENS KOEHLER", cx, world_h * 0.78f, world_h * 0.010f, ink::recede,
                  true);
        const int count = window_->menu_count();
        for (int i = 0; i < count; ++i) {
            const bool sel = window_->menu_index() == i;
            const float y = window_->menu_item_top_y(i);
            draw_text(inst, window_->menu_label(i), cx, y, window_->menu_text_px(),
                      sel ? 0.95f : 0.45f, sel ? 0.75f : 0.45f, sel ? 0.25f : 0.50f, true);
        }
        // Both hint lines clear the skyline entirely rather than being coloured to
        // survive crossing it. A glyph hangs 5*px *below* the top_y given here, so
        // the usable band is from the launch towers' 12 units up to the menu
        // band's 0.16h — 14.5 units for two lines and the gap between them, which
        // only fits if they are set smaller. They are, and the deep red then works
        // as a colour rather than as a contrast problem: it reads against the dark
        // sky and stays quieter than the menu items it belongs to.
        draw_text(inst, "ARROWS ENTER OR MOUSE", cx, world_h * 0.151f, world_h * 0.0070f, 0.62f,
                  0.17f, 0.16f, true);
        draw_text(inst, "F FULLSCREEN   M MUSIC   A AUDIO", cx, world_h * 0.105f, world_h * 0.0060f,
                  0.62f, 0.17f, 0.16f, true);
    } else if (game_over) {
        draw_text(inst, "GAME OVER", cx, world_h * 0.70f, world_h * 0.042f, 0.95f, 0.30f, 0.25f,
                  true);
        draw_text(inst, "SCORE", cx, world_h * 0.46f, world_h * 0.015f, ink::body, true);
        draw_text(inst, std::to_string(sim.score() < 0 ? 0 : sim.score()), cx, world_h * 0.36f,
                  world_h * 0.028f, 1.0f, 1.0f, 1.0f, true);
        if (window_->ai_assisted()) {
            // Say why it was not offered, rather than silently skipping the entry.
            // Below 0.22, where the score's glyphs stop hanging — at 0.25 the two
            // lines printed through each other.
            draw_text(inst, "AI RUN   NOT A HIGHSCORE", cx, world_h * 0.19f, world_h * 0.011f,
                      0.45f, 0.95f, 0.65f, true);
        }
        draw_text(inst, "PRESS ENTER", cx, world_h * 0.12f, world_h * 0.013f, ink::footer, true);
    } else if (state == GameWindow::State::Replays) {
        // One screen, two contents. The heading and the empty-state copy are the
        // only difference: scrolling, hover, selection and paging are identical,
        // and a second implementation of them would drift from this one.
        const bool models = window_->browsing() == GameWindow::Browse::Models;
        draw_text(inst, models ? "MODELS" : "REPLAYS", cx, world_h * 0.90f, world_h * 0.026f,
                  ink::heading, true);
        const int count = window_->replay_count();
        if (count == 0) {
            // Never a bare empty panel: say what is missing and what puts it
            // there, or a person is left deciding whether the feature is broken.
            draw_text(inst, models ? "NO MODELS INSTALLED" : "NO RECORDINGS IN RUNS", cx,
                      world_h * 0.55f, world_h * 0.012f, 0.6f, 0.65f, 0.7f, true);
            draw_text(inst, models ? "PROMOTE ONE IN THE CONSOLE" : "TRAINING WRITES THEM THERE",
                      cx, world_h * 0.46f, world_h * 0.009f, 0.4f, 0.45f, 0.5f, true);
        } else {
            // A window of at most `replay_rows_visible` rows, scrolled by the
            // window rather than computed here: the mouse hit test reads the same
            // layout, and two copies of it would drift apart the moment either
            // moved.
            const int selected = window_->menu_index();
            const int first = window_->replay_scroll();
            const int last = std::min(count, first + GameWindow::replay_rows_visible);
            for (int i = first; i < last; ++i) {
                const bool sel = selected == i;
                draw_text(inst, window_->replay_name(i), cx, window_->replay_row_top_y(i),
                          window_->replay_row_px(), sel ? 0.95f : 0.45f, sel ? 0.75f : 0.45f,
                          sel ? 0.25f : 0.50f, true);
            }
            if (count > GameWindow::replay_rows_visible) {
                draw_text(inst, std::to_string(selected + 1) + " OF " + std::to_string(count), cx,
                          world_h * 0.14f, world_h * 0.009f, 0.4f, 0.45f, 0.5f, true);
            }
        }
        draw_text(inst, "ARROWS ENTER   ESC BACK", cx, world_h * 0.07f, world_h * 0.009f, 0.4f,
                  0.45f, 0.5f, true);
    } else if (state == GameWindow::State::Help) {
        draw_text(inst, "HELP", cx, world_h * 0.88f, world_h * 0.026f, ink::heading, true);
        // The objective first and on its own, in the cities' own blue: it is the
        // one line here that is not a key, and it is what all the keys are for.
        // Five equally-weighted sentences made the player read all five to find
        // out what the game wanted.
        draw_text(inst, "DEFEND YOUR CITIES", cx, world_h * 0.71f, world_h * 0.016f, ink::city,
                  true);
        static constexpr std::array<Binding, 4> keys{
            {{"MOUSE", "AIM"}, {"CLICK", "FIRE"}, {"ESC", "PAUSE MENU"}, {"ENTER", "SELECT"}}};
        draw_bindings(inst, keys, cx, world_h * 0.55f, world_h * 0.10f, world_h * 0.014f);
        draw_text(inst, "PRESS ENTER", cx, world_h * 0.13f, world_h * 0.011f, ink::footer, true);
    } else if (state == GameWindow::State::About) {
        draw_text(inst, "ABOUT", cx, world_h * 0.94f, world_h * 0.024f, ink::heading, true);
        const std::string version_line = "VERSION " + std::string(version());
        // Uppercase-only legal notices + credits (the pixel font has no lower case
        // or punctuation beyond the period added for the version string). Lines are
        // kept short so none overflows the width, and spaced so none overlaps.
        //
        // Three weights, not one. As a flat block of nine identical lines this
        // read as a legal notice with the game's name accidentally at the top:
        // everything shouted equally, so nothing did. Now the name carries the
        // heading's colour, what this build *is* sits in ordinary copy, and the
        // trademark notice drops to the byline's recede blue — still legible,
        // visibly the small print, and grouped by colour rather than by a gap.
        const std::array<std::pair<std::string_view, const std::array<float, 3>&>, 9> lines{{
            {"MISSILE DEFENSE", ink::heading},
            {version_line, ink::body},
            {"COPYRIGHT 2026 JENS KOEHLER", ink::body},
            {"MIT LICENSE", ink::muted},
            {"DEVELOPED WITH CLAUDE CODE", ink::muted},
            {"USES QT MINIAUDIO VULKAN", ink::muted},
            {"MISSILE COMMAND IS AN", ink::recede},
            {"ATARI TRADEMARK", ink::recede},
            {"INDEPENDENT NON COMMERCIAL HOMAGE", ink::recede},
        }};
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const float y = world_h * (0.80f - (static_cast<float>(i) * 0.075f));
            draw_text(inst, lines[i].first, cx, y, world_h * 0.011f, lines[i].second, true);
        }
        // Tightened by half a percent a line so this one can come up off the
        // floor: at 0.05 its bottom row of pixels fell *below* y = 0 and was
        // clipped by the frame. It now clears the ground as well.
        draw_text(inst, "PRESS ENTER", cx, world_h * 0.13f, world_h * 0.010f, ink::footer, true);
    } else if (state == GameWindow::State::Options) {
        draw_text(inst, "OPTIONS", cx, world_h * 0.88f, world_h * 0.026f, ink::heading, true);
        const int count = GameWindow::options_count();
        for (int i = 0; i < count; ++i) {
            const bool sel = window_->menu_index() == i;
            const float y = window_->menu_item_top_y(i);
            draw_text(inst, window_->options_label(i), cx, y, window_->menu_text_px(),
                      sel ? 0.95f : 0.45f, sel ? 0.75f : 0.45f, sel ? 0.25f : 0.50f, true);
        }
        draw_text(inst, "ARROWS ENTER OR MOUSE", cx, world_h * 0.09f, world_h * 0.010f, 0.4f, 0.45f,
                  0.5f, true);
    } else if (state == GameWindow::State::Watch) {
        // The same centred list as OPTIONS, and deliberately so: this is the
        // third screen with that shape, and a chooser that looked different
        // from the other two would read as a different kind of thing.
        draw_text(inst, "WATCH AI", cx, world_h * 0.88f, world_h * 0.026f, ink::heading, true);
        for (int i = 0; i < window_->watch_count(); ++i) {
            const bool sel = window_->menu_index() == i;
            const float y = window_->menu_item_top_y(i);
            draw_text(inst, window_->watch_label(i), cx, y, window_->menu_text_px(),
                      sel ? 0.95f : 0.45f, sel ? 0.75f : 0.45f, sel ? 0.25f : 0.50f, true);
        }
        draw_text(inst, "WHO PLAYS. T TAKES OVER MID GAME", cx, world_h * 0.09f, world_h * 0.010f,
                  0.4f, 0.45f, 0.5f, true);
    } else if (state == GameWindow::State::Highscores) {
        draw_text(inst, "HIGHSCORES", cx, world_h * 0.90f, world_h * 0.026f, ink::heading, true);
        const auto& table = window_->highscores();
        if (table.entries().empty()) {
            draw_text(inst, "NO SCORES YET", cx, world_h * 0.5f, world_h * 0.015f, ink::footer,
                      true);
        } else {
            int row = 0;
            for (const auto& entry : table.entries()) {
                const std::string ini(entry.initials.begin(), entry.initials.end());
                const std::string line =
                    std::to_string(row + 1) + "  " + ini + "  " + std::to_string(entry.score);
                // A full table is ten rows between the heading and the footer, and
                // a glyph hangs 4.92*px below the top it is given. At 0.012 type
                // on a 0.062 step that leaves under half a unit of air, which
                // printed the table as one vertical smear. Size and step are set
                // together here for that reason: the gap is the constraint, not
                // either number on its own.
                const float y = world_h * (0.76f - (static_cast<float>(row) * 0.064f));
                // A podium rather than a winner and nine also-rans: gold, then
                // the two still worth chasing in ordinary copy, then the tail a
                // step back. Ten identically-bright rows made the table a list
                // to search instead of a standing to read.
                const std::array<float, 3>* colour = &ink::muted;
                if (row == 0) {
                    colour = &ink::gold;
                } else if (row < 3) {
                    colour = &ink::body;
                }
                draw_text(inst, line, cx, y, world_h * 0.0105f, *colour, true);
                ++row;
            }
        }
        // As on ABOUT: a full ten-row table used to push this line off the bottom
        // of the frame. The rows give up a little spacing so it has somewhere to
        // go — and a footer that is half cut off reads as a rendering fault.
        draw_text(inst, "PRESS ENTER", cx, world_h * 0.11f, world_h * 0.011f, ink::footer, true);
    } else if (state == GameWindow::State::EnterScore) {
        draw_text(inst, "NEW HIGH SCORE", cx, world_h * 0.82f, world_h * 0.024f, ink::gold, true);
        draw_text(inst, std::to_string(window_->final_score()), cx, world_h * 0.66f,
                  world_h * 0.022f, 1.0f, 1.0f, 1.0f, true);
        draw_text(inst, "ENTER YOUR INITIALS", cx, world_h * 0.52f, world_h * 0.012f, ink::body,
                  true);
        const auto initials = window_->entry_initials();
        const int slot = window_->entry_slot();
        const float gpx = world_h * 0.05f; // big initial glyphs
        const float spacing = gpx * 4.0f;
        const float top_y = world_h * 0.42f;
        for (int i = 0; i < 3; ++i) {
            const float center_x = cx + ((static_cast<float>(i) - 1.0f) * spacing);
            const bool sel = (i == slot);
            draw_glyph(inst, glyph_for(initials[static_cast<std::size_t>(i)]),
                       center_x - (1.5f * gpx), top_y, gpx, sel ? 1.0f : 0.85f, sel ? 0.8f : 0.9f,
                       sel ? 0.3f : 1.0f);
            if (sel) { // caret underline under the active slot
                inst.push_back(
                    rect(center_x, top_y - (5.5f * gpx), 1.6f * gpx, 0.25f, 1.0f, 0.8f, 0.3f));
            }
        }
        draw_text(inst, "ARROWS TYPE ENTER", cx, world_h * 0.10f, world_h * 0.011f, 0.6f, 0.65f,
                  0.7f, true);
    }

    // World -> clip transform (matches the mouse->world mapping in GameWindow).
    submit(inst, {Pass{Projection::make(world_w, world_h, static_cast<float>(sz.width()),
                                        static_cast<float>(sz.height())),
                       whole(sz), 0, static_cast<std::uint32_t>(inst.size())}});
}

void Renderer::draw_match(const replay::MatchPlayer& match, QSize size) {
    const Sim& left = match.left().player.sim();
    const Sim& right = match.right().player.sim();
    const float world_w = left.config().world_width;
    const float world_h = left.config().world_height;
    const float tsec =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - start_).count();
    const auto w = static_cast<float>(size.width());
    const auto h = static_cast<float>(size.height());
    const float half = w * 0.5f;

    std::vector<InstanceData> inst;
    inst.reserve(max_instances);
    std::vector<Pass> passes;

    // Four passes, drawn in this order: sky, left world, right world, chrome.
    // Depth testing is off, so pass order *is* the layering — the two worlds sit
    // on the shared sky, and the scoreboard sits on both.
    const Projection full = Projection::make(world_w, world_h, w, h);
    const Projection side = Projection::make(world_w, world_h, half, h);

    // The starfield spans the window rather than each half, so the divider cuts
    // through one continuous sky instead of two visibly repeated ones.
    const auto sky_first = static_cast<std::uint32_t>(inst.size());
    for (const auto& star : stars_) {
        const float tw = 0.55f + (0.45f * std::sin(star.phase + (star.speed * tsec)));
        inst.push_back(glow(star.x, star.y, star.size, 0.85f, 0.9f, 1.0f, star.base * tw));
    }
    passes.push_back(
        Pass{full, whole(size), sky_first, static_cast<std::uint32_t>(inst.size()) - sky_first});

    for (int which = 0; which < 2; ++which) {
        const Sim& sim = which == 0 ? left : right;
        const auto first = static_cast<std::uint32_t>(inst.size());
        build_backdrop(sim, terrain_, ground_, inst);
        build_entities(sim, inst, tsec);
        passes.push_back(Pass{side, rect_viewport(which == 0 ? 0.0f : half, half, h), first,
                              static_cast<std::uint32_t>(inst.size()) - first});
    }

    const auto chrome_first = static_cast<std::uint32_t>(inst.size());

    // The divider. Deliberately plain and dim: it separates, it does not decorate.
    inst.push_back(
        rect(world_w * 0.5f, world_h * 0.5f, 0.12f, world_h * 0.5f, 0.30f, 0.34f, 0.42f, 0.9f));

    // One scoreboard per side, above its own half. Both are drawn under the
    // window-wide projection so the two read at the same size — text scaled per
    // viewport would make the halves look like different screens.
    // Stacked by glyph height, not by eye: `draw_text` hangs 5*px *below* its
    // `top_y`, so rows placed on a pretty arithmetic series overlap.
    const float name_px = world_h * 0.0080f;
    const float score_px = world_h * 0.0140f;
    const float small_px = world_h * 0.0060f;
    for (int which = 0; which < 2; ++which) {
        const auto& entry = which == 0 ? match.left() : match.right();
        const Sim& sim = which == 0 ? left : right;
        const Sim& other = which == 0 ? right : left;
        const float centre = world_w * (which == 0 ? 0.25f : 0.75f);

        draw_text(inst, shout(entry.name), centre, world_h * 0.995f, name_px, 0.72f, 0.82f, 0.94f,
                  true);

        // The leader is tinted, not badged: a label saying WINNING would have to
        // be right, and mid-episode it is only ever "ahead so far".
        const bool ahead = sim.score() > other.score();
        const auto score = static_cast<std::uint32_t>(sim.score() < 0 ? 0 : sim.score());
        const auto width = static_cast<float>(digit_count(score)) * score_px * 4.0f;
        draw_number(inst, score, centre + (width * 0.5f), world_h * 0.948f, score_px,
                    ahead ? 0.55f : 0.72f, ahead ? 1.00f : 0.76f, ahead ? 0.65f : 0.82f, true);

        // What the tournament measured, when a manifest said — so a viewer can
        // tell one episode's luck from the average it came from. Beside the wave
        // rather than under it: a fourth row would reach into the playfield.
        const auto mean = static_cast<std::uint32_t>(
            entry.mean_score.value_or(0.0) < 0.0 ? 0.0 : entry.mean_score.value_or(0.0));
        const float wave_w = stat_width("WAVE", sim.wave(), small_px);
        const float mean_w =
            entry.mean_score.has_value() ? stat_width("MEAN", mean, small_px) : 0.0f;
        const float gap = entry.mean_score.has_value() ? small_px * 12.0f : 0.0f;
        const float row = wave_w + gap + mean_w;
        const float row_y = world_h * 0.868f;
        draw_stat(inst, "WAVE", sim.wave(), centre - (row * 0.5f) + (wave_w * 0.5f), row_y,
                  small_px, 0.48f, 0.54f, 0.62f);
        if (entry.mean_score.has_value()) {
            draw_stat(inst, "MEAN", mean, centre + (row * 0.5f) - (mean_w * 0.5f), row_y, small_px,
                      0.42f, 0.47f, 0.55f);
        }
    }

    // The shared transport, at the foot of the window and spanning both halves —
    // one clock, stated once, because there is only one.
    //
    // Lifted clear of the ground rather than laid on it: a match has no scrim to
    // hide behind (both halves are live), so the only thing keeping this line
    // readable is height. The caption's glyphs hang 4.92*px below the top given
    // here — 13.6 world units — which clears the tallest thing the field carries,
    // a battery's 12-unit towers. The bar then sits above the caption.
    const float bar_y = world_h * 0.125f;
    const float bar_w = world_w * 0.36f;
    inst.push_back(rect(world_w * 0.5f, bar_y, bar_w, 0.22f, 0.22f, 0.26f, 0.34f));
    const float done = std::clamp(match.progress(), 0.0f, 1.0f);
    if (done > 0.0f) {
        inst.push_back(rect((world_w * 0.5f) - bar_w + (bar_w * done), bar_y, bar_w * done, 0.22f,
                            0.45f, 0.75f, 0.95f));
    }
    draw_text(inst,
              match.finished() ? "MATCH COMPLETE   R RESTART   ESC BACK"
                               : "SPACE PAUSE   ARROWS SEEK   R RESTART   ESC BACK",
              world_w * 0.5f, world_h * 0.105f, world_h * 0.0060f, 0.46f, 0.53f, 0.62f, true);

    passes.push_back(Pass{full, whole(size), chrome_first,
                          static_cast<std::uint32_t>(inst.size()) - chrome_first});

    submit(inst, passes);
}

void Renderer::submit(std::vector<InstanceData>& inst, const std::vector<Pass>& passes) {
    // One hard budget for the whole frame, however many viewports share it.
    // Dropping the tail is the right failure: instances are drawn back to
    // front, so what is lost is the topmost decoration and not the field.
    if (inst.size() > max_instances) {
        inst.resize(max_instances);
    }
    const auto total = static_cast<std::uint32_t>(inst.size());

    const auto frame = static_cast<std::size_t>(window_->currentFrame());
    std::memcpy(instance_mapped_[frame], inst.data(), inst.size() * sizeof(InstanceData));

    const QSize sz = window_->swapChainImageSize();
    std::array<VkClearValue, 3> clear{};
    clear[0].color = {{0.05f, 0.06f, 0.12f, 1.0f}}; // night sky
    clear[1].depthStencil = {1.0f, 0};
    clear[2].color = clear[0].color;
    const bool msaa = window_->sampleCountFlagBits() > VK_SAMPLE_COUNT_1_BIT;

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = window_->defaultRenderPass();
    begin.framebuffer = window_->currentFramebuffer();
    begin.renderArea.extent.width = static_cast<std::uint32_t>(sz.width());
    begin.renderArea.extent.height = static_cast<std::uint32_t>(sz.height());
    begin.clearValueCount = msaa ? 3u : 2u;
    begin.pClearValues = clear.data();

    const VkCommandBuffer cb = window_->currentCommandBuffer();
    dev_->vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
    dev_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    std::array<VkBuffer, 2> vbufs{quad_buf_, instance_bufs_[frame]};
    std::array<VkDeviceSize, 2> offsets{0, 0};
    dev_->vkCmdBindVertexBuffers(cb, 0, static_cast<std::uint32_t>(vbufs.size()), vbufs.data(),
                                 offsets.data());

    for (const Pass& pass : passes) {
        // Re-clamped against the budget above, not trusted: a caller computes
        // its slice while building, and the truncation happens afterwards.
        if (pass.first >= total) {
            continue;
        }
        const std::uint32_t count = std::min(pass.count, total - pass.first);
        if (count == 0) {
            continue;
        }

        dev_->vkCmdSetViewport(cb, 0, 1, &pass.viewport);
        // Scissored to the viewport, so a half-screen pass cannot bleed into
        // its neighbour through the letterbox bars its projection leaves.
        VkRect2D scissor{};
        scissor.offset.x = static_cast<std::int32_t>(pass.viewport.x);
        scissor.offset.y = static_cast<std::int32_t>(pass.viewport.y);
        scissor.extent.width = static_cast<std::uint32_t>(pass.viewport.width);
        scissor.extent.height = static_cast<std::uint32_t>(pass.viewport.height);
        dev_->vkCmdSetScissor(cb, 0, 1, &scissor);

        PushConstants pc{};
        pc.a[0] = pass.proj.ax;
        pc.a[1] = pass.proj.ay;
        pc.b[0] = pass.proj.bx;
        pc.b[1] = pass.proj.by;
        dev_->vkCmdPushConstants(cb, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                                 static_cast<std::uint32_t>(sizeof(PushConstants)), &pc);
        dev_->vkCmdDraw(cb, 6, count, 0, pass.first);
    }

    dev_->vkCmdEndRenderPass(cb);
    window_->frameReady();

    // Wait for the frame we just handed to Qt before starting the next one.
    //
    // This is a workaround for a defect in `QVulkanWindow`, not a property this
    // renderer wants. Qt hardcodes `frameLag = 2` and so allocates two sets of
    // per-frame semaphores, but it asks the driver for a swapchain of whatever
    // `VkSurfaceCapabilitiesKHR::minImageCount` requires — three on every driver
    // tested here. It then reuses an acquire semaphore two frames later without
    // having waited for the submit that waits on that same semaphore to
    // complete, which is `VUID-vkAcquireNextImageKHR-semaphore-01779`: the
    // semaphore still has an uncompleted wait pending. That is undefined
    // behaviour that happens to work on the drivers we run on.
    //
    // Nothing in this tree can reorder Qt's acquire against its own fence wait,
    // and the swapchain image count is not application-controllable. Serialising
    // here is the one lever an application has: it guarantees the submit has
    // retired before Qt can reuse that semaphore. The evidence that this is
    // Qt's and not ours is in `python/tests/e2e/test_vulkan_validation.py`,
    // which reproduces the VUID from a bare `QVulkanWindow` with no project code
    // in the process at all, and fails if that stops being true on a driver that
    // can exhibit it. Not every driver can: the hazard needs a swapchain deeper
    // than Qt's two frame-resource sets, and one that asks for only two images
    // never trips it. That is a property of the driver, not a fix.
    //
    // The cost is the CPU/GPU overlap of one frame. Measured over 600 frames at
    // 1280x720 it was inside run-to-run noise on both an RTX 5090 and lavapipe,
    // because this renderer is instanced quads and is bound by neither. Revisit
    // if the renderer ever becomes GPU-bound.
    dev_->vkQueueWaitIdle(window_->graphicsQueue());

    window_->requestUpdate();
}

void Renderer::releaseResources() {
    VkDevice device = window_->device();
    for (std::size_t i = 0; i < instance_bufs_.size(); ++i) {
        if (instance_mems_[i] != VK_NULL_HANDLE) {
            dev_->vkUnmapMemory(device, instance_mems_[i]);
            dev_->vkFreeMemory(device, instance_mems_[i], nullptr);
        }
        if (instance_bufs_[i] != VK_NULL_HANDLE) {
            dev_->vkDestroyBuffer(device, instance_bufs_[i], nullptr);
        }
    }
    instance_bufs_.clear();
    instance_mems_.clear();
    instance_mapped_.clear();

    if (pipeline_ != VK_NULL_HANDLE) {
        dev_->vkDestroyPipeline(device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        dev_->vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (quad_buf_ != VK_NULL_HANDLE) {
        dev_->vkDestroyBuffer(device, quad_buf_, nullptr);
        quad_buf_ = VK_NULL_HANDLE;
    }
    if (quad_mem_ != VK_NULL_HANDLE) {
        dev_->vkFreeMemory(device, quad_mem_, nullptr);
        quad_mem_ = VK_NULL_HANDLE;
    }
}

} // namespace md
