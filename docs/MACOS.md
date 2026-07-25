# macOS

The game builds on macOS with the same Clang + CMake + Ninja toolchain as Linux.
Vulkan is the one real difference: macOS has no Vulkan driver, so the renderer
runs on **MoltenVK**, which translates Vulkan to Metal. Nothing in `app/` had to
change for that — the app creates its instance and device through `QVulkanWindow`
(see [app/main.cpp](../app/main.cpp)), so Qt owns the portability plumbing that
MoltenVK needs.

Everything that is not platform-specific — how to play, what the scripted AI is,
the project layout — is in the [main README](../README.md). This page is only the
macOS delta.

> **Status: built and tested in CI, never run by a human.** This port was written
> without access to a Mac. The `macos` job in
> [.github/workflows/ci.yml](../.github/workflows/ci.yml) compiles the renderer
> and runs all 104 C++ tests — Debug with sanitizers, and Release — on Apple
> silicon on every push, and they pass: the *simulation* is genuinely verified on
> arm64 and libc++. What no runner can check is the part that needs a screen.
> Nobody has yet watched a frame of this. If you have a Mac and it misbehaves,
> that is a bug worth reporting, not your setup.

## Quick start

```bash
# 1 — dependencies (Homebrew; qtbase, not qt — see below)
brew install cmake ninja qtbase molten-vk vulkan-headers vulkan-loader glslang

# 2 — tell CMake where Homebrew put them
export CMAKE_PREFIX_PATH="$(brew --prefix qtbase):$(brew --prefix)"

# 3 — build
git clone https://github.com/JensKSP/missile-defense.git
cd missile-defense
cmake --preset release && cmake --build --preset release

# 4 — play
open build/release/app/md_app.app
```

`open` launches the bundle; to pass arguments, run the executable inside it:

```bash
./build/release/app/md_app.app/Contents/MacOS/md_app --watch
```

From here the [main README](../README.md#quick-start) applies unchanged.

## What the packages are

The set maps onto the Debian dependencies in the
[main README](../README.md#requirements):

| Homebrew formula | Debian equivalent | Why |
|---|---|---|
| `qtbase` | `qt6-base-dev` | Window, event loop, `QVulkanWindow` |
| `molten-vk` | `mesa-vulkan-drivers` | The driver — Vulkan on top of Metal |
| `vulkan-loader` | `libvulkan-dev` | `libvulkan.dylib`; MoltenVK ships no loader |
| `vulkan-headers` | `libvulkan-dev` | Headers, so `find_package(Vulkan)` succeeds |
| `glslang` | `glslang-tools` | `glslangValidator`, compiles the shaders to SPIR-V |
| `vulkan-tools` *(optional)* | `vulkan-tools` | `vulkaninfo`, to check the driver is visible |

**`qtbase`, not `qt`.** The app needs `Qt6::Gui` only; the `qt` umbrella formula
pulls in WebEngine and three dozen other modules that cost a long download and
buy nothing here. Homebrew's `qtbase` build-depends on `molten-vk` and
`vulkan-headers`, which is why it is configured *with* Vulkan support — a Qt
built without it would fail at `QVulkanInstance::create()` no matter what else
is installed.

## Build notes

- **Deployment target.** [CMakeLists.txt](../CMakeLists.txt) defaults
  `CMAKE_OSX_DEPLOYMENT_TARGET` to **14.0**. That is not arbitrary: Apple's
  libc++ puts `std::print` / `std::println` behind an availability attribute, and
  the CLI tools use them, so an older target fails to compile `<print>`. Override
  it on the command line if your libc++ allows lower.
- **Apple Clang is fine.** The tree needs C++23, which Xcode 16's Clang provides.
  Homebrew's `llvm` would match the pinned clang-21 more exactly, but it brings
  its own libc++ — mixing that with a Homebrew Qt built against Apple's is an ABI
  hazard. Use Apple Clang for anything that links Qt.
- **Single architecture.** Builds are native-arch, because Homebrew's Qt is not
  universal. A universal binary would mean building Qt twice yourself.
- **The bundle is built in the dev tree too**, not just at install time, so what
  you play is what ships. A bare Mach-O executable never becomes a foreground GUI
  process on macOS: no Dock icon, and it cannot reliably take keyboard focus,
  which for a game is fatal. The directory is `md_app.app` on every platform's
  build path; the name a user sees comes from `CFBundleName`.

## Packaging

`poe dmg` builds a drag-to-Applications disk image into `build/release/`. The work
is in [app/deploy_macos.cmake.in](../app/deploy_macos.cmake.in), which runs at
install time in three steps whose order is forced — each would undo the next:

1. **`macdeployqt`** copies the Qt frameworks and the cocoa platform plugin into
   the bundle and rewrites the executable's load commands. The direct equivalent
   of [tools/windeploy.sh](../tools/windeploy.sh) on Windows.
2. **MoltenVK is copied in by hand**, and its ICD manifest rewritten to point at
   the copy by a path relative to the bundle. `macdeployqt` cannot do this: it
   follows link-time dependencies, and `md_app` links the Vulkan *loader*, never
   the driver — the loader `dlopen`s that from a manifest it locates by path at
   runtime. [app/main.cpp](../app/main.cpp) aims `VK_DRIVER_FILES` at the bundled
   manifest before the first Vulkan call, and does nothing when there is none, so
   a build tree still uses whatever Homebrew installed.
3. **Everything is signed**, nested code first and the bundle last. Not optional
   even for a local build: the linker ad-hoc signs every arm64 binary it emits,
   step 1 invalidated that by rewriting load commands, and an arm64 binary whose
   signature does not match is killed on launch.

The result depends on nothing but macOS itself. It is the same promise the
Windows installer and the .deb make.

### Signing it for other people

The identity defaults to `-`, an ad-hoc signature: enough to run the bundle on the
machine that built it, never enough to hand to someone else, because it carries no
identity for Gatekeeper to check. Whoever downloads that disk image has to clear
quarantine by hand:

```bash
xattr -dr com.apple.quarantine "/Applications/md_app.app"
```

With a Developer ID Application certificate in the keychain, point the build at
it. The hardened runtime and a secure timestamp are switched on with it rather
than separately, because notarisation refuses a submission lacking either:

```bash
cmake --preset release \
  -DMD_MACOS_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
poe dmg
xcrun notarytool submit build/release/missile-defense-*.dmg \
  --apple-id you@example.com --team-id TEAMID --password "$APP_PASSWORD" --wait
xcrun stapler staple build/release/missile-defense-*.dmg
```

The certificate is the part that cannot be worked around: it needs a paid Apple
Developer account. The *machine* can be — [`rcodesign`][rcodesign] does Mach-O
signing and notary submission from Linux or Windows, if running a release through
a CI runner is not what you want.

[rcodesign]: https://github.com/indygreg/apple-platform-tools

## If the game starts and immediately dies

`Failed to create Vulkan instance` means the loader found no driver. Check what
it can see:

```bash
vulkaninfo --summary        # expect a device named "Apple M<n>" via MoltenVK
```

If that is empty, point the loader at MoltenVK's manifest explicitly — Homebrew's
prefix is not one of the loader's default search directories on Apple Silicon:

```bash
export VK_DRIVER_FILES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
```

## Why there is no Docker path

There is no macOS container, and there cannot be one: containers share the host
kernel, and Apple ships no macOS kernel you can host. Docker Desktop on a Mac
runs a Linux VM, and Apple's own Containerization framework also runs Linux
guests. The alternatives are worse rather than merely inconvenient:

- **Cross-compiling** (OSXCross) needs a macOS SDK extracted from Xcode, whose
  licence limits use to Apple hardware, and it cannot *run* anything — so the
  test suite, the whole point of the gate, is gone.
- **macOS in a VM on non-Apple hardware** violates the macOS licence, and more
  decisively has no Metal GPU — which for a renderer that reaches Metal through
  MoltenVK is a dead end regardless of the licensing.

Hence the CI job: hosted Apple hardware is the only honest way to build and test
this without owning a Mac. It is free for public repositories; private ones bill
macOS minutes at ten times the Linux rate.
