<!-- SPDX-License-Identifier: MIT -->
# Third-party licenses

Missile Defense (© 2026 Jens Köhler) is released under the [MIT License](LICENSE).
It builds on the third-party components below. This file is the required
attribution/notice; the full license texts it refers to are in [`licenses/`](licenses/)
and are bundled with every binary distribution (Windows installer, Debian package).

**Three products ship, and they carry different obligations.** The game
(`missile-defense`) is self-contained C++; the bindings (`python3-md`) add a
compiled Python extension; the trainer (`missile-defense-trainer`) is
pure Python that *depends on* PySide6 and, optionally, PyTorch. The distinction
that matters here is **bundling versus depending**: an obligation attaches to
what a release actually redistributes, and the trainer redistributes none of its
Python dependencies — they come from the distribution's own packages, from the
user's `pip`, or from the managed runtime the trainer installs into the user's
home directory on request.

## The game and the bindings — what a release redistributes

| Component | Version | License | Shipped in a release? | Obligation |
|---|---|---|---|---|
| [Qt 6](https://www.qt.io/) (Core, Gui, Network) | 6.11 | **LGPL-3.0-only** | Yes — dynamically linked DLLs/SOs | License text + this notice + relinkability + source offer |
| [miniaudio](https://miniaud.io/) | 0.11.22 | Public domain (Unlicense) / MIT-0 | Yes — compiled in | None (no attribution required) |
| [nanobind](https://github.com/wjakob/nanobind) | 2.13 | BSD-3-Clause | Yes — its runtime is linked into `_md_native` | Reproduce the copyright notice: [`licenses/nanobind/BSD-3-Clause.txt`](licenses/nanobind/BSD-3-Clause.txt) |
| [Vulkan](https://www.vulkan.org/) loader + headers | 1.4 | Apache-2.0 | Loader is the OS/GPU driver's `vulkan-1.dll`; headers are build-time only | None (loader not distributed by us) |
| [Catch2](https://github.com/catchorg/Catch2) | 3.7.1 | BSL-1.0 | No — unit-test binaries only | None |
| [glslang](https://github.com/KhronosGroup/glslang) | — | BSD-3-Clause / Apache-2.0 | No — build tool; shaders are baked to SPIR-V | None |
| C/C++ runtime (libc++, libunwind, libwinpthread, ICU, HarfBuzz, FreeType, glib, PCRE2, zlib, …) | — | Apache-2.0-with-LLVM-exception + permissive (MinGW-w64) | Windows bundle only | Include the respective notices |

## The trainer — depended on, never redistributed

None of these is inside any artifact a release publishes. The Debian trainer
package *depends* on the distribution's copies; the Windows and macOS trainer
payloads are our own Python plus `_md_native` and run on an interpreter the user
already has; and the managed runtime (**Set up training…**) `pip install`s into
`~/.local/share/MissileDefense/runtime/`, which is the user acquiring the
package, not us shipping it. They are credited because a user is entitled to
know what the trainer will pull onto their machine before it does.

| Component | Version | License | How it arrives | Obligation |
|---|---|---|---|---|
| [PySide6](https://doc.qt.io/qtforpython/) (QtWidgets, QtCharts, QtGui) | 6.11 | **LGPL-3.0-only** | Debian: `python3-pyside6.*`; elsewhere the user's `pip` | Notice only — not redistributed; see below |
| [PyTorch](https://pytorch.org/) | ≥ 2.0 | BSD-3-Clause | `Suggests:` on Debian; otherwise the managed runtime or the user's `pip` | Notice only |
| [NumPy](https://numpy.org/) | ≥ 1.24 | BSD-3-Clause | `python3-numpy`, a hard dependency of the bindings | Notice only |
| [psutil](https://github.com/giampaolo/psutil) | ≥ 5 | BSD-3-Clause | Optional — the CPU/RAM meters | Notice only |
| [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) (`pynvml`) | — | BSD-3-Clause | Optional — the NVIDIA GPU probe | Notice only |
| [amdsmi](https://github.com/ROCm/amdsmi) / [pyrsmi](https://github.com/ROCm/pyrsmi) | — | MIT | Optional — the AMD GPU probe | Notice only |

**Why PySide6 being LGPL does not make this project LGPL.** The trainer imports
PySide6 at run time from a copy the user's system provides; it is dynamically
linked, unmodified, and replaceable. That is the same arrangement the game has
with Qt itself, and the reason the trainer is a *separate package* that the
MIT-licensed game does not depend on — a rule
[`python/tests/test_packaging.py`](python/tests/test_packaging.py) asserts against
`debian/control` rather than leaving to memory. Should a future release ever
bundle PySide6 (a frozen trainer, say), the full Qt obligation below applies to
it unchanged.

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

## nanobind — BSD-3-Clause notice (required)

`_md_native`, the Python extension in `python3-md` and in the Windows/macOS
trainer payload, is built with **nanobind** (© 2022 Wenzel Jakob), whose runtime
is statically linked into it. BSD-3-Clause clause 2 requires that a *binary*
redistribution reproduce the copyright notice, so unlike the credits below this
one is an obligation and not a courtesy:
[`licenses/nanobind/BSD-3-Clause.txt`](licenses/nanobind/BSD-3-Clause.txt),
bundled with every artifact that carries the extension.

## miniaudio / Catch2 / glslang

miniaudio is dedicated to the public domain (or, at your option, MIT-0) and
requires no attribution; it is credited here as a courtesy. Catch2 (BSL-1.0) and
glslang (BSD-3-Clause / Apache-2.0) are used only for testing and for the build,
respectively, and are not part of the distributed game.
