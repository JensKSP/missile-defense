# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
cmake_minimum_required(VERSION 3.25)

# ---- Compiler selection ------------------------------------------------------
# An ordinary build uses the compiler the operating system already has: GCC on
# Linux, Apple Clang on macOS, MSVC on Windows. Nothing is chosen here for it.
# That is what makes `cmake --preset release` work on a machine nobody prepared
# — the quick start, the wheel pip builds on a stranger's computer, and the .deb
# a distribution builds are all that same path, and each of them wants the
# toolchain of the system it is on rather than one this file went looking for.
#
# Development is the exception, and only on Linux, where the OS default is not
# Clang: clang-tidy, the sanitizer builds and LLVM source-based coverage are the
# instruments this project is measured with, so the presets that mean "I am
# working on this" (debug, profile, coverage) ask for Clang by setting
# MD_DEV_TOOLCHAIN. macOS is Clang already and Windows is MSVC's, so neither
# needs the nudge. An explicit -DCMAKE_CXX_COMPILER or a $CXX in the environment
# beats all of it. Must run before project(), which is when the compiler is
# detected — and MD_DEV_TOOLCHAIN is readable here because a preset's cache
# variables arrive on the command line, before this file is first read.
if(MD_DEV_TOOLCHAIN AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX}
   AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  # The versioned name first is the quality gate's pin; the bare one is whatever
  # the distribution calls its default Clang, which is all a dev build needs.
  find_program(MD_CLANGXX NAMES clang++-21 clang++)
  if(MD_CLANGXX)
    set(CMAKE_CXX_COMPILER "${MD_CLANGXX}" CACHE FILEPATH "C++ compiler")
  else()
    message(STATUS
      "md: no clang++ found — this development build uses the system default. "
      "clang-tidy and `poe coverage` need Clang; install it to run them.")
  endif()
endif()

# ---- macOS deployment target -------------------------------------------------
# Also before project(), and `APPLE` is not set yet — hence the host test. 14.0
# is not a guess: Apple's libc++ gates std::print / std::println behind an
# availability attribute, and the CLI tools (agent/eval_main.cpp, bench/) use
# them, so an older target fails to compile <print>. Raise or lower it with
# -DCMAKE_OSX_DEPLOYMENT_TARGET=<version>. Left unset off macOS.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum supported macOS version")
endif()

project(missile_defense
  VERSION 0.1.0
  DESCRIPTION "Retro arcade missile-defense game + RL environment"
  LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
# C++23 is the newest complete standard the floor compilers below implement, so
# it is the one the tree is written in. (Both offer a partial C++26 through
# -std=c++2c; a preview mode is not something a minimum requirement can name.)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Minimum compiler --------------------------------------------------------
# Every supported compiler is welcome; each has a floor, and the floor is the
# same idea in three dialects — the newest complete C++ standard this tree is
# written in is C++23, and the library half of it (the <print> that agent/ and
# bench/ use) arrives with the standard library, not the front end.
#
#   GCC 14      the default on Debian 13 and Ubuntu 26.04. libstdc++ 13 has no
#               <print>, which is what makes 14 rather than 13 the number.
#   Clang 19    the newest in Debian 13's `main`, so a development machine needs
#               no backport and no third-party apt repository to get one.
#   AppleClang  numbered on Xcode's own scale, so it is not compared against
#               either of the above; docs/MACOS.md carries its requirement.
#   MSVC        docs/WINDOWS.md carries its requirement.
#
# Refused here, with a sentence, rather than eight screens into a template error.
# NOT MSVC excludes the MSVC-frontend clang-cl, which versions itself on LLVM's
# scale but is Windows' business.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR
    "GCC ${CMAKE_CXX_COMPILER_VERSION} is older than this project's minimum, "
    "GCC 14 — earlier libstdc++ has no C++23 <print>, which the command-line "
    "tools use. Ubuntu 24.04 ships g++-14 alongside its default g++-13: install "
    "it and configure with CXX=g++-14.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
  message(FATAL_ERROR
    "Clang ${CMAKE_CXX_COMPILER_VERSION} is older than this project's minimum, "
    "Clang 19 — the `clang` package on Debian 13. Install a newer one and point "
    "CXX / -DCMAKE_CXX_COMPILER at it.")
endif()

# Always emit compile_commands.json (clangd / clang-tidy consume it).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options -----------------------------------------------------------------
option(MD_SANITIZE "Enable AddressSanitizer + UBSan" OFF)
option(MD_WERROR "Treat warnings as errors" ON)
# Both default ON, and both are meant to be turned off by builds whose job is to
# produce an artifact rather than to judge the code. A release or packaging build
# runs on whatever compiler its machine ships, and a newer one inventing a new
# diagnostic — or being stricter about ISO conformance in a third-party header —
# must not be able to stop a package being built. The quality gate, on a pinned
# clang-21, is where zero-warnings is enforced and where it means something.
option(MD_PEDANTIC "Enable -Wpedantic (strict ISO conformance diagnostics)" ON)
option(MD_BUILD_APP "Build the Qt 6 + Vulkan app (needs Qt6 + Vulkan)" ON)
option(MD_COVERAGE "Instrument for LLVM source-based coverage (clang)" OFF)
# The one thing that genuinely cannot be had from another compiler: this is LLVM
# source-based instrumentation, read back by llvm-profdata/llvm-cov
# (tools/coverage.py). GCC produces gcov data instead, which those tools cannot
# read — and the failure without this check is an empty report rather than an
# error, which is to say a coverage gate that passes by measuring nothing.
if(MD_COVERAGE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "MD_COVERAGE needs Clang (LLVM source-based coverage); this build uses "
    "${CMAKE_CXX_COMPILER_ID}. The `coverage` preset asks for Clang on Linux — "
    "install one, or point CXX / -DCMAKE_CXX_COMPILER at it.")
endif()
option(MD_VULKAN_VALIDATION "Request Vulkan validation layers (dev only)" OFF)
option(MD_FETCH_MINIAUDIO "Fetch miniaudio when no system copy (libminiaudio-dev) is found" ON)
option(MD_NATIVE "Optimise for this machine's CPU (-march=native); not portable" OFF)
option(MD_LTO "Link-time optimisation (ThinLTO) — cross-TU inlining" OFF)
# ON, so that `cmake --build` alone produces the whole product rather than the
# C++ two-thirds of it — the extension is part of what this repository builds,
# and a build system that needs a task runner standing behind it to finish the
# job is two build systems. Safe as a default because bindings/CMakeLists.txt
# looks for Python and nanobind before it builds anything and reports a skip
# rather than failing: a checkout with no Python toolchain configures and builds
# exactly as it did when this was OFF. Turn it off explicitly to be certain (the
# `nopython` Debian profile and the install-tree test both do).
option(MD_BUILD_BINDINGS "Build the nanobind Python extension (needs Python + nanobind)" ON)
# The two headless command-line tools: the scripted-agent evaluator (poe eval)
# and the throughput benchmark (poe bench). On for a dev or packaging build; the
# wheel turns it OFF (pyproject.toml). They are the only translation units in the
# tree that use <print>/std::println, which needs libstdc++ 14 — and a wheel is
# compiled on whatever the installer's default compiler is, which on Ubuntu 24.04
# is g++-13, with no <print>. The wheel ships only _md_native, so it has no
# reason to build them and every reason not to.
option(MD_BUILD_TOOLS "Build the headless CLI tools (md_agent_eval, md_bench)" ON)
set(MD_PGO "off" CACHE STRING "Profile-guided optimisation: off | generate | use")
set_property(CACHE MD_PGO PROPERTY STRINGS off generate use)
# macOS only. "-" is an ad-hoc signature: enough to launch the bundle on the
# machine that built it (arm64 refuses to run unsigned code at all), never enough
# to hand to anyone else, because it carries no identity Gatekeeper can check.
# Set it to a "Developer ID Application: ..." certificate to produce something
# notarisable — see docs/MACOS.md.
set(MD_MACOS_CODESIGN_IDENTITY "-" CACHE STRING
    "codesign identity for the macOS bundle; \"-\" means ad-hoc")

# ---- Warnings ----------------------------------------------------------------
# Opt-in via `target_link_libraries(<t> PRIVATE md::warnings)`. Keeps strict
# flags on our own code without imposing them on third-party deps (e.g. Catch2).
add_library(md_warnings INTERFACE)
add_library(md::warnings ALIAS md_warnings)
# MSVC's driver rejects the GCC/Clang spellings outright, so the strict set is
# expressed twice. clang-cl takes the MSVC driver flags, hence MSVC-not-Clang
# rather than a plain compiler-id test.
if(MSVC)
  target_compile_options(md_warnings INTERFACE
    /W4          # /Wall on MSVC includes noise from its own system headers
    /permissive- # standard conformance
    /w14242      # conversion, possible loss of data (-Wconversion)
    /w14254      # larger bit-field conversion
    /w14263      # member function does not override (-Woverloaded-virtual)
    /w14265      # non-virtual destructor (-Wnon-virtual-dtor)
    /w14287      # unsigned/negative constant mismatch
    /w14296      # expression is always true/false
    /w14311      # pointer truncation
    /w14545 /w14546 /w14547 /w14549 /w14555 # malformed expressions
    /w14619      # unknown #pragma warning
    /w14640      # non-thread-safe static init
    /w14826      # sign-extending conversion
    /w14928)     # illegal copy-initialization
else()
  target_compile_options(md_warnings INTERFACE
    -Wall
    -Wextra
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wnon-virtual-dtor
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wmisleading-indentation)
  # GCC only, and not a preference: -Wnull-dereference fires *inside libstdc++*
  # on an optimised GCC build, reporting `_M_in_cur` / `_M_in_end` in
  # <streambuf> as potentially null after it inlines
  # `std::string{istreambuf_iterator, {}}` — the ordinary read-a-file-into-a-
  # string idiom (agent/src/policy.cpp). There is nothing in our code to fix: the
  # pointers it doubts are basic_streambuf's own, and the usual "warnings do not
  # come from system headers" rule does not save us, because the diagnostic is
  # raised by the optimiser after inlining rather than by the parser.
  #
  # It stays on for Clang, which does not have the false positive, so the check
  # is still enforced somewhere real — and dropping it here is what lets the
  # default build on Debian 13 (GCC 14) compile at all under -Werror.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(md_warnings INTERFACE -Wno-null-dereference)
  endif()
endif()
# Separate from the list above so a packaging build can drop it alone. MSVC has
# no -Wpedantic; /permissive- above is the nearest thing and is conformance mode
# rather than a diagnostic, so it stays either way.
if(MD_PEDANTIC AND NOT MSVC)
  target_compile_options(md_warnings INTERFACE -Wpedantic)
endif()

if(MD_WERROR)
  target_compile_options(md_warnings INTERFACE $<IF:$<BOOL:${MSVC}>,/WX,-Werror>)
endif()

# ---- Codegen tuning ----------------------------------------------------------
# All off by default: the shipped artifact must run on any x86-64 and stay exactly
# reproducible. Each of these is opt-in for throughput work, and each is verified
# against the golden determinism test before use — see docs/PERFORMANCE.md.
if(MD_NATIVE)
  add_compile_options(-march=native)
endif()

if(MD_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT md_ipo_ok OUTPUT md_ipo_why)
  if(md_ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "MD_LTO requested but unsupported here: ${md_ipo_why}")
  endif()
endif()

# Profile-guided optimisation, two passes:
#   1. cmake --preset release -DMD_PGO=generate && build && run the workload
#   2. llvm-profdata merge -output=md.profdata build/release/*.profraw
#   3. cmake --preset release -DMD_PGO=use -DMD_PGO_FILE=<abs path to md.profdata>
set(MD_PGO_FILE "" CACHE FILEPATH "Merged .profdata for MD_PGO=use")
if(MD_PGO STREQUAL "generate")
  add_compile_options(-fprofile-instr-generate)
  add_link_options(-fprofile-instr-generate)
elseif(MD_PGO STREQUAL "use")
  if(NOT MD_PGO_FILE)
    message(FATAL_ERROR "MD_PGO=use needs -DMD_PGO_FILE=<merged .profdata>")
  endif()
  add_compile_options("-fprofile-instr-use=${MD_PGO_FILE}")
  add_link_options("-fprofile-instr-use=${MD_PGO_FILE}")
endif()

# ---- Determinism -------------------------------------------------------------
# Disable floating-point contraction (FMA) so results are bit-identical across
# Debug/Release and optimization levels — required for seed+action replays.
if(NOT MSVC)
  add_compile_options(-ffp-contract=off)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(/clang:-ffp-contract=off) # clang-cl: same flag, MSVC driver
else()
  # cl.exe has no exact equivalent. /fp:precise forbids most reassociation, but
  # the golden checksum test is the authority — run it before trusting a replay
  # or a recording produced by an MSVC build.
  add_compile_options(/fp:precise)
  message(WARNING
    "MSVC: no exact -ffp-contract=off equivalent. Determinism parity with the "
    "Clang builds is unverified here — run the golden checksum test (ctest -L e2e).")
endif()

# ---- Sanitizers --------------------------------------------------------------
# Global on purpose: instrument our code *and* the deps it links under test.
if(MD_SANITIZE AND WIN32)
  # ASan/UBSan are unreliable under MinGW (no LeakSanitizer runtime), so the
  # sanitized debug preset degrades to a plain Debug build here rather than fail.
  message(WARNING "MD_SANITIZE: not enabling ASan/UBSan on Windows (MinGW).")
elseif(MD_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- Coverage ----------------------------------------------------------------
# Clang source-based coverage. Used by scripts/coverage.sh (core-only build) to
# gate line coverage of the pure simulation. Global so tests + library are both
# instrumented and merge into one profile.
if(MD_COVERAGE)
  add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
  add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif()

# ---- Testing -----------------------------------------------------------------
include(CTest)  # defines BUILD_TESTING (ON by default) and calls enable_testing()

# Test framework: Catch2 v3, acquired once here so every test directory (core/,
# agent/) shares one copy and one CMAKE_MODULE_PATH. Prefer a system Catch2
# (Debian package `catch2`) so distro/offline builds fetch nothing; otherwise
# fetch it. If it is neither installed nor fetchable, skip tests rather than fail.
if(BUILD_TESTING)
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    if(Catch2_DIR)
      list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}") # Catch.cmake ships beside the config
    endif()
  elseif(FETCHCONTENT_FULLY_DISCONNECTED)
    message(WARNING "Catch2 not found and fetching is disabled — skipping tests. "
                    "Install the 'catch2' package to build and run them.")
  else()
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.1
      GIT_SHALLOW    TRUE
      SYSTEM)
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  endif()

  if(TARGET Catch2::Catch2WithMain)
    include(Catch)
    # Clang 22+ (newer than the pinned Debian clang-21) flags Catch2's __COUNTER__
    # use under -Wpedantic as a C2y extension. It comes from Catch2 (SYSTEM,
    # third-party), so silence it — but the -Wno must sit AFTER md::warnings'
    # -Wpedantic/-Werror on the command line, so carry it on its own interface
    # target that each test links *after* md::warnings. Scoped to the tests; the
    # rest of the tree keeps the pedantic check.
    add_library(md_test_warnings INTERFACE)
    target_compile_options(md_test_warnings INTERFACE
      "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,22>>:-Wno-c2y-extensions>")
  endif()
endif()

# nlohmann/json — for the `.mdp` manifest and the match manifest.
#
# A library rather than a hand-rolled parser because these files can arrive from
# outside: a policy someone downloaded is untrusted input, and a parser for
# untrusted input is the one place in this program where a bug is a security bug
# rather than a wrong number. Header-only and MIT, so it costs nothing to vet.
#
# Acquired here rather than in `agent/` because two libraries need it —
# `md::agent` for `.mdp` and `md::replay` for a paired-match manifest — and a
# find_package buried in one subdirectory that another silently depends on is
# a build that works by accident of subdirectory ordering.
#
# Prefer the system copy (Debian `nlohmann-json3-dev`, MSYS2
# `mingw-w64-clang-x86_64-nlohmann-json`, Homebrew `nlohmann-json`), otherwise
# fetch — the same shape as Catch2 and miniaudio, so a distribution build never
# reaches the network and an offline developer build says why it cannot.
find_package(nlohmann_json 3.10 QUIET)
if(nlohmann_json_FOUND)
  message(STATUS "nlohmann_json: using the system package")
elseif(NOT FETCHCONTENT_FULLY_DISCONNECTED)
  message(STATUS "nlohmann_json: not found — fetching")
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
    SYSTEM)
  FetchContent_MakeAvailable(nlohmann_json)
else()
  message(FATAL_ERROR
    "nlohmann/json not found. Install nlohmann-json3-dev, or enable fetching "
    "with network access.")
endif()

# ---- Where the native extension is installed ---------------------------------
# Decided *here*, before `add_subdirectory(bindings)`, because that is where the
# `install(TARGETS _md_native)` rule is registered and a destination chosen
# afterwards is chosen too late. It used to be overridden further down, next to
# the .py sources — so on macOS the sources landed inside the trainer bundle and
# the extension landed beside the prefix, giving a bundle that imports `missile_defense` and
# then cannot import `missile_defense._md_native`. A trainer without the binding is a trainer
# that cannot train, which is the one thing it is for.
#
# A cache entry, so `-DMD_PYTHON_INSTALL_DIR=...` still wins — debian/rules
# passes the distribution's own dist-packages path and must keep doing so.
if(APPLE OR WIN32)
  # Nothing: the trainer is not installed by this build on either platform. It
  # ships as a wheel beside the game and the game pip-installs it on request
  # (app/install.hpp), because only whoever owns the interpreter can put a Python
  # package into it — and on these two that is the user, not us.
  set(md_python_install_default "")
elseif(DEFINED SKBUILD)
  # scikit-build-core's install tree *is* the wheel root, so a relative `missile_defense` is
  # the package directory the wheel unpacks into site-packages.
  set(md_python_install_default "missile_defense")
else()
  # Nothing can guess this on Linux. A distribution owns the interpreter and
  # knows where its dist-packages are; debian/rules asks it and passes the
  # answer. Empty means nobody has said, and the Python install rules are then
  # not registered at all — because the alternative, which is what used to
  # happen, is a relative `missile_defense` resolved against the prefix: `/usr/local/missile_defense/`, an
  # extension and a stub sitting where no interpreter will ever look. It went
  # unnoticed while the bindings were off by default and there was no target to
  # install; the moment they built by default, `cmake --install` started
  # littering.
  set(md_python_install_default "")
endif()
set(MD_PYTHON_INSTALL_DIR "${md_python_install_default}" CACHE STRING
    "Where `cmake --install` puts the missile_defense package and _md_native")

# ---- Components ---------------------------------------------------------------
add_subdirectory(core)
add_subdirectory(agent)   # scripted reference AI (M4) + the evaluation protocol
add_subdirectory(replay)  # recorded runs: file format + playback (M3)
add_subdirectory(bench)   # throughput benchmark (poe bench)
add_subdirectory(bindings) # md::rl (always) + the optional nanobind module (M5)

if(MD_BUILD_APP)
  add_subdirectory(app)
endif()

# ---- The Python half as a distribution package -------------------------------
# Off by default, and that is the important part: a wheel gets its .py sources
# and its `missile-defense-trainer` / `missile-defense-train` commands from
# scikit-build-core, so CMake
# installing them too would put two copies of everything in it. Only a
# distribution package — which is not built by pip and has no entry-point
# machinery — needs this, and debian/rules is what turns it on.
#
# It is what makes the trainer *reachable from an install* rather than only from
# a checkout, which is the whole point of the split: the game is one product and
# the trainer is another, and neither may drag the other in.
option(MD_INSTALL_PYTHON_PACKAGE
       "Install the md Python sources, launcher scripts and trainer desktop entry" OFF)

# Where the trainer lands differs on every platform, and the reason is the same
# each time: **nothing may assume the interpreter knows where `missile_defense` is.**
#
#   Linux    a distribution owns the interpreter, so the package goes where that
#            interpreter already looks (debian/rules asks it, and passes the
#            answer as MD_PYTHON_INSTALL_DIR) and the launchers are plain
#            wrappers in bin/.
#   Windows  the *user* owns the interpreter and may not have installed one yet,
#            so the payload ships beside the game and the .cmd launcher puts its
#            own directory on PYTHONPATH.
#   macOS    nothing inside an .app is on any search path at all, so the trainer
#            becomes its own bundle carrying the package in Contents/Resources —
#            which is also what makes it a separate icon a person can drag, i.e.
#            a separate product rather than a hidden feature of the game.
if(MD_INSTALL_PYTHON_PACKAGE)
  include(GNUInstallDirs)

  # Asking to install the Python package without saying where is a mistake in
  # the job that configured this, not something to paper over with a guess: a
  # relative default resolved against the prefix is how `/usr/local/md/` came
  # about. Fatal for the same reason the binding check below is — a packaging
  # build that silently puts the trainer somewhere unimportable produces an
  # installer nobody notices is broken until it is in someone's hands.
  if(NOT MD_PYTHON_INSTALL_DIR)
    message(FATAL_ERROR
      "MD_INSTALL_PYTHON_PACKAGE is ON but MD_PYTHON_INSTALL_DIR is empty, so "
      "there is nowhere to put the missile_defense package. Pass the target interpreter's "
      "site-packages directory, the way debian/rules does.")
  endif()

  # A trainer without the native binding starts, browses and replays — and
  # cannot train, which is the one thing it is for. On Debian that cannot happen
  # (the trainer package carries the binding), but an installer payload is just files
  # and would ship the hole silently. bindings/ skips itself when Python or
  # nanobind is missing, so this is exactly the case a packaging job hits.
  #
  # FATAL, not a warning. It was a warning, and it warned exactly as designed —
  # into a CI log nobody reads on a green step — while the Windows job shipped
  # an installer whose trainer had no `_md_native.pyd`. Only a packaging
  # assertion three steps later noticed. Asking to install the Python package
  # and not being able to build it is not a situation with a sensible partial
  # answer, so it stops the configure instead.
  # Only when bindings were *asked for*. Installing the sources without the
  # extension is a real configuration — Debian splits them into two binary
  # packages, and `tests/e2e/test_packages.py` stages the sources half on its
  # own — so the fault is not "no extension", it is "an extension was requested
  # and silently not produced". That is the Windows case: nanobind was missing,
  # bindings/ skipped itself, and the installer shipped a trainer without one.
  if(MD_BUILD_BINDINGS AND NOT TARGET _md_native)
    message(FATAL_ERROR
      "MD_INSTALL_PYTHON_PACKAGE is ON but the native binding was not built, so "
      "the installed trainer could not start a training run. Enable "
      "MD_BUILD_BINDINGS and install Python development files + nanobind for "
      "*this* interpreter (${Python_EXECUTABLE}), or turn "
      "MD_INSTALL_PYTHON_PACKAGE off.")
  endif()

  # And it has to be a binding the *user's* interpreter can load. A packaged
  # trainer is exec'd by whatever `python3` the machine has — the distribution's
  # on Debian, the user's own on Windows and macOS — which is never the one that
  # built the module. A version-tagged extension therefore works only where the
  # minor version happens to match, and fails as `ModuleNotFoundError:
  # missile_defense._md_native` on every other machine: an installed trainer that browses and
  # replays and cannot train, with nothing to say why.
  #
  # A developer build may be version-tagged; a build that is being packaged may
  # not, and this is the point where the two are distinguishable.
  if(MD_BUILD_BINDINGS AND NOT MD_NATIVE_IS_ABI3)
    message(FATAL_ERROR
      "MD_INSTALL_PYTHON_PACKAGE is ON but _md_native was built for one "
      "interpreter rather than against the stable ABI, so the installed trainer "
      "would only train on a machine running Python ${Python_VERSION} exactly. "
      "The stable ABI needs CPython >= 3.12 and its SABI development files "
      "(Debian: python3-dev; look for Development.SABIModule in the configure "
      "output). Configure with a newer interpreter, or turn "
      "MD_INSTALL_PYTHON_PACKAGE off.")
  endif()

  # `python3` unqualified, so the package follows the distribution's Python
  # rather than pinning the absolute path of whichever one configured the build.
  set(MD_LAUNCHER_PYTHON "python3" CACHE STRING
      "The interpreter the installed missile-defense-* launchers exec")

  # `MD_PYTHON_INSTALL_DIR` is already decided, above the components — see the
  # note there. Only the launcher directory is left to choose, and there is now
  # only one platform to choose it for.
  #
  # **This whole block is the distribution's path and nobody else's.** Windows and
  # macOS used to come through here too: the installer wrote the `md` package
  # beside the game, a `.cmd` or an `.app` wrapper next to it, and the game
  # searched for the payload. All of that is gone. There, the trainer is a wheel
  # the game installs with pip on request (app/install.hpp), because the thing
  # that made the payload work — an interpreter — was never ours to ship. Only a
  # distribution owns the interpreter, so only a distribution can install a
  # Python package into it.
  if(WIN32 OR APPLE)
    message(FATAL_ERROR
      "MD_INSTALL_PYTHON_PACKAGE is for distribution packaging, and this is not "
      "a distribution. On Windows and macOS the trainer ships as a wheel beside "
      "the game (MD_TRAINER_WHEEL) and is installed by the game on request.")
  endif()
  include(GNUInstallDirs)
  set(md_trainer_bindir "${CMAKE_INSTALL_BINDIR}")

  # The sources. _md_native and its stub are excluded because bindings/ installs
  # those — it is the one that knows whether the extension was built at all.
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/missile_defense/"
    DESTINATION "${MD_PYTHON_INSTALL_DIR}"
    COMPONENT python
    FILES_MATCHING
      PATTERN "*.py"
      PATTERN "*.pyi"
      PATTERN "_md_native.pyi" EXCLUDE
      PATTERN "__pycache__" EXCLUDE)

  foreach(md_launcher missile-defense-trainer missile-defense-train)
    if(md_launcher STREQUAL "missile-defense-trainer")
      set(MD_LAUNCHER_MODULE "missile_defense.ui")
    else()
      set(MD_LAUNCHER_MODULE "missile_defense.training.cli")
    endif()
    configure_file(packaging/launcher.in "${CMAKE_CURRENT_BINARY_DIR}/${md_launcher}" @ONLY)
    install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/${md_launcher}"
      DESTINATION "${md_trainer_bindir}"
      COMPONENT python)
  endforeach()

  # Its own menu entry, in Development/Science rather than Games — someone
  # looking for the arcade cabinet must not find two of them.
  install(FILES "${CMAKE_SOURCE_DIR}/packaging/missile-defense-trainer.desktop"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications"
    COMPONENT python)
endif()

# ---- Packaging (CPack) -------------------------------------------------------
# Only meaningful for the full app build (that is what gets installed).
#   Windows: cmake --build --preset release && cpack --preset release  (-> NSIS + ZIP)
#   Debian:  poe deb            (-> .deb)
# Packaging must run from the MSYS2 CLANG64 shell on Windows (windeployqt/ldd).
if(MD_BUILD_APP AND PROJECT_IS_TOP_LEVEL)
  set(CPACK_PACKAGE_NAME "missile-defense")
  set(CPACK_PACKAGE_VENDOR "Jens Köhler")
  set(CPACK_PACKAGE_CONTACT "Jens Köhler <jens@koehler-speyer.de>")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Retro arcade missile-defense game (Qt 6 + Vulkan)")
  set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/JensKSP/missile-defense")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

  # What the package filenames say. Separate from PROJECT_VERSION because
  # project() only accepts a plain numeric triple, and a nightly needs to stamp
  # something like 0.1.0-dev128-g83a3fe03 — which is the whole point of a
  # nightly, since a file called missile-defense-0.1.0.dmg that is not 0.1.0 is
  # worse than useless once it is sitting in someone's Downloads folder.
  set(MD_PACKAGE_VERSION "${PROJECT_VERSION}" CACHE STRING
      "Version stamped into package filenames; defaults to the project version")
  set(CPACK_PACKAGE_VERSION "${MD_PACKAGE_VERSION}")

  if(WIN32)
    # Windows installer (NSIS) + portable archive (ZIP). The install tree already
    # carries the bundled Qt/runtime DLLs and licence texts (see app/CMakeLists).
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Missile Defense")
    set(CPACK_PACKAGE_EXECUTABLES "missile-defense;Missile Defense") # Start-menu shortcut
    set(CPACK_NSIS_PACKAGE_NAME "Missile Defense")
    set(CPACK_NSIS_DISPLAY_NAME "Missile Defense")
    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "missile-defense.exe") # offer to launch after install
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_URL_INFO_ABOUT "${CPACK_PACKAGE_HOMEPAGE_URL}")
    set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/packaging/missile-defense.ico")        # installer
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/packaging/missile-defense.ico")     # uninstaller
    set(CPACK_NSIS_INSTALLED_ICON_NAME "missile-defense.exe") # icon in Add/Remove Programs
  elseif(APPLE)
    # A drag-to-Applications disk image holding the .app, with everything it
    # needs inside it: Qt via macdeployqt, and the Vulkan loader with it because
    # the game links it; MoltenVK by hand, because the loader dlopens the driver
    # and macdeployqt therefore never sees it (app/deploy_macos.cmake.in).
    # Single-architecture: it is built for whichever arch the host is, because
    # Homebrew's Qt is not universal. The DMG is unsigned and unnotarised, which
    # is a deliberate limit of building without an Apple Developer ID, not an
    # oversight — docs/MACOS.md says what that means for anyone who downloads it.
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_PACKAGING_INSTALL_PREFIX "/")
    set(CPACK_DMG_VOLUME_NAME "Missile Defense ${MD_PACKAGE_VERSION}")
    set(CPACK_DMG_FORMAT "UDZO") # compressed read-only, the usual shape
    set(CPACK_PACKAGE_FILE_NAME
        "missile-defense-${MD_PACKAGE_VERSION}-macos-${CMAKE_SYSTEM_PROCESSOR}")
  else()
    # Debian package. Auto-derive shared-lib deps (Qt 6, Vulkan loader, libc, ...)
    # from the linked binary via dpkg-shlibdeps (needs dpkg-dev installed).
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
    set(CPACK_GENERATOR "DEB")
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT") # missile-defense_<version>_<arch>.deb
    set(CPACK_DEBIAN_PACKAGE_SECTION "games")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  endif()

  # ---- One component, because there is one thing to install ---------------
  # There used to be two: `game`, and a `python` one the NSIS installer offered
  # unticked so someone who came for a game could decline an interpreter. That
  # choice is gone, and so is the checkbox — what the installer ships is the
  # game, plus the trainer's wheel beside it. The choice moved into the game,
  # where TRAIN AI asks before it downloads anything, and where it can be made by
  # someone who has already decided they want to train an agent rather than by
  # someone reading a component list on their way to playing.
  #
  # Two things that cost us went with it: an unticked component that (still) had
  # no Start-menu entry, so anyone who did tick it had nothing to click; and a
  # `.cmd` launcher Smart App Control blocks outright on a stock Windows 11.
  set(CPACK_COMPONENT_GAME_DISPLAY_NAME "Missile Defense")
  set(CPACK_COMPONENT_GAME_DESCRIPTION
      "The game. Self-contained: no Python, no training tools.")
  set(CPACK_COMPONENT_GAME_REQUIRED ON)
  set(CPACK_COMPONENTS_ALL game)

  if(APPLE)
    # A disk image is not an installer with checkboxes — it is a window with
    # icons in it, and the choice is made by dragging one or both. Splitting it
    # into two DMGs would be the wrong shape for the platform, so the components
    # exist for the layout's sake and the image itself stays monolithic.
    set(CPACK_MONOLITHIC_INSTALL ON)
  elseif(WIN32)
    # The ZIP is the portable copy of the whole thing; only the installer asks.
    set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
    set(CPACK_NSIS_COMPONENT_INSTALL ON)
  else()
    # `poe deb` builds one .deb from CPack; the *real* three-binary split is
    # debian/control's, built by dpkg-buildpackage. Two different mechanisms for
    # two different jobs, and the CPack one must not start emitting three files.
    set(CPACK_DEB_COMPONENT_INSTALL OFF)
  endif()

  include(CPack)
endif()
