<!-- SPDX-License-Identifier: MIT -->
# Third-party licenses

Missile Defense (© 2026 Jens Köhler) is released under the [MIT License](LICENSE).
It builds on the third-party components below. This file is the required
attribution/notice; the full license texts it refers to are in [`licenses/`](licenses/)
and are bundled with every binary distribution (Windows installer, Debian package).

| Component | Version | License | Shipped in the game? | Obligation |
|---|---|---|---|---|
| [Qt 6](https://www.qt.io/) (Core, Gui, Network) | 6.11 | **LGPL-3.0-only** | Yes — dynamically linked DLLs/SOs | License text + this notice + relinkability + source offer |
| [miniaudio](https://miniaud.io/) | 0.11.22 | Public domain (Unlicense) / MIT-0 | Yes — compiled in | None (no attribution required) |
| [Vulkan](https://www.vulkan.org/) loader + headers | 1.4 | Apache-2.0 | Loader is the OS/GPU driver's `vulkan-1.dll`; headers are build-time only | None (loader not distributed by us) |
| [Catch2](https://github.com/catchorg/Catch2) | 3.7.1 | BSL-1.0 | No — unit-test binaries only | None |
| [glslang](https://github.com/KhronosGroup/glslang) | — | BSD-3-Clause / Apache-2.0 | No — build tool; shaders are baked to SPIR-V | None |
| C/C++ runtime (libc++, libunwind, libwinpthread, ICU, HarfBuzz, FreeType, glib, PCRE2, zlib, …) | — | Apache-2.0-with-LLVM-exception + permissive (MinGW-w64) | Windows bundle only | Include the respective notices |

The trademark **Missile Command** belongs to Atari. This game is an independent,
non-commercial fan homage and is not affiliated with or endorsed by Atari.

## Qt — LGPL-3.0 notice (required)

This program uses the Qt framework, © The Qt Company Ltd and contributors,
licensed under the **GNU Lesser General Public License, version 3**
([`licenses/qt6/LGPL-3.0-only.txt`](licenses/qt6/LGPL-3.0-only.txt)).

To honour the LGPL, Qt is **dynamically linked**: the Qt libraries ship as
separate `Qt6*.dll` / `libQt6*.so` files, so you may replace them with your own
build of the same Qt version. No changes were made to Qt itself.

**Written offer for source code.** The complete corresponding source code for the
exact Qt version used is available from The Qt Company at
<https://download.qt.io/official_releases/qt/> (choose 6.11) and via the
distribution's Qt source packages (e.g. `apt-get source qt6-base`, or the MSYS2
`mingw-w64-clang-x86_64-qt6-base` source). The author will also, on request,
provide the corresponding source for the Qt version bundled with a given release.

## Vulkan

Vulkan® is a registered trademark of the Khronos Group. The Vulkan **loader**
(`vulkan-1.dll` on Windows, `libvulkan.so` on Linux) is provided by the operating
system / GPU driver and is not redistributed here. The Vulkan-Headers used at
build time are licensed under Apache-2.0.

## miniaudio / Catch2 / glslang

miniaudio is dedicated to the public domain (or, at your option, MIT-0) and
requires no attribution; it is credited here as a courtesy. Catch2 (BSL-1.0) and
glslang (BSD-3-Clause / Apache-2.0) are used only for testing and for the build,
respectively, and are not part of the distributed game.
