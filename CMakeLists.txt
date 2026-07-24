# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
cmake_minimum_required(VERSION 3.25)

# ---- Compiler selection ------------------------------------------------------
# The project targets Clang (LLVM source-based coverage + clang-tidy). Windows
# (MSYS2 CLANG64) and macOS already default to Clang; only Linux defaults to GCC,
# so nudge it there. An explicit -DCMAKE_CXX_COMPILER or a $CXX in the environment
# always wins. Must run before project() so it takes effect at compiler detection.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX}
   AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  find_program(MD_CLANGXX NAMES clang++-21 clang++)
  if(MD_CLANGXX)
    set(CMAKE_CXX_COMPILER "${MD_CLANGXX}" CACHE FILEPATH "C++ compiler")
  endif()
endif()

project(missile_defense
  VERSION 0.1.0
  DESCRIPTION "Faithful Atari Missile Command clone — game + RL environment"
  LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json (clangd / clang-tidy consume it).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options -----------------------------------------------------------------
option(MD_SANITIZE "Enable AddressSanitizer + UBSan" OFF)
option(MD_WERROR "Treat warnings as errors" ON)
option(MD_BUILD_APP "Build the Qt 6 + Vulkan app (needs Qt6 + Vulkan)" ON)
option(MD_COVERAGE "Instrument for LLVM source-based coverage (clang)" OFF)
option(MD_VULKAN_VALIDATION "Request Vulkan validation layers (dev only)" OFF)
option(MD_FETCH_MINIAUDIO "Fetch miniaudio when no system copy (libminiaudio-dev) is found" ON)

# ---- Warnings ----------------------------------------------------------------
# Opt-in via `target_link_libraries(<t> PRIVATE md::warnings)`. Keeps strict
# flags on our own code without imposing them on third-party deps (e.g. Catch2).
add_library(md_warnings INTERFACE)
add_library(md::warnings ALIAS md_warnings)
target_compile_options(md_warnings INTERFACE
  -Wall
  -Wextra
  -Wpedantic
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
if(MD_WERROR)
  target_compile_options(md_warnings INTERFACE -Werror)
endif()

# ---- Determinism -------------------------------------------------------------
# Disable floating-point contraction (FMA) so results are bit-identical across
# Debug/Release and optimization levels — required for seed+action replays.
add_compile_options(-ffp-contract=off)

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

# ---- Components ---------------------------------------------------------------
add_subdirectory(core)
# bindings/ (nanobind)  -> added in Step 2

if(MD_BUILD_APP)
  add_subdirectory(app)
endif()

# ---- Packaging (CPack -> .deb) -----------------------------------------------
# Only meaningful for the full app build (that is what gets installed). Build a
# package with:  poe deb   (or: cmake --build --preset release && cd build/release && cpack -G DEB)
if(MD_BUILD_APP AND PROJECT_IS_TOP_LEVEL)
  set(CPACK_PACKAGE_NAME "missile-defense")
  set(CPACK_PACKAGE_VENDOR "Jens Köhler")
  set(CPACK_PACKAGE_CONTACT "Jens Köhler <jens@koehler-speyer.de>")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Faithful Atari Missile Command clone (Qt 6 + Vulkan)")
  set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/JensKSP/missile-defense")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
  set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

  set(CPACK_GENERATOR "DEB")
  set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT") # missile-defense_<version>_<arch>.deb
  set(CPACK_DEBIAN_PACKAGE_SECTION "games")
  set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
  # Auto-derive the shared-library dependencies (Qt 6, Vulkan loader, libc, ...)
  # from the linked binary via dpkg-shlibdeps (needs dpkg-dev installed).
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

  include(CPack)
endif()
