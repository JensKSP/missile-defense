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
> and runs the full simulation test suite on every push, so the *simulation* is
> genuinely verified on arm64. Nobody has yet watched a frame of it. If you have
> a Mac and it misbehaves, that is a bug worth reporting, not your setup.

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

`poe dmg` builds a drag-to-Applications disk image into `build/release/`. It runs
`macdeployqt` to copy the Qt frameworks into the bundle and rewrite the load
paths — the direct equivalent of [tools/windeploy.sh](../tools/windeploy.sh) on
Windows — then re-signs the bundle ad-hoc, because rewriting load commands
invalidates the signature the linker applies to every arm64 binary, and an arm64
binary with a broken signature is killed on launch.

Two honest limits on that DMG:

- **It is not notarised, and not signed with a Developer ID.** That needs an
  Apple Developer account (99 USD/year). Gatekeeper will refuse to open it on
  another machine until the user clears quarantine
  (`xattr -dr com.apple.quarantine "/Applications/md_app.app"`). Signing and
  notarising can in fact be done without a Mac — [`rcodesign`][rcodesign] does
  Mach-O signing and notary submission from Linux or Windows — but it still needs
  the paid certificate.
- **MoltenVK is not inside the bundle.** `macdeployqt` bundles what the binary
  *links*; the Metal driver is `dlopen`ed by the Vulkan loader from an ICD
  manifest outside the bundle. So the DMG currently expects
  `brew install molten-vk` on the target machine. Making it self-contained means
  copying `libMoltenVK.dylib` into `Contents/Frameworks` with its own ICD JSON in
  `Contents/Resources/vulkan/icd.d` and pointing `VK_DRIVER_FILES` at it on
  launch. Worth doing before anything is published; not done yet.

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
