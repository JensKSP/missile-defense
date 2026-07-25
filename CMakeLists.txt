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
option(MD_PROFILE "Compile the per-phase timers in Sim::step (md/profile.hpp)" OFF)
option(MD_NATIVE "Optimise for this machine's CPU (-march=native); not portable" OFF)
option(MD_LTO "Link-time optimisation (ThinLTO) — cross-TU inlining" OFF)
set(MD_PGO "off" CACHE STRING "Profile-guided optimisation: off | generate | use")
set_property(CACHE MD_PGO PROPERTY STRINGS off generate use)

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

# ---- Components ---------------------------------------------------------------
add_subdirectory(core)
add_subdirectory(agent)   # scripted reference AI (M4) + the evaluation protocol
add_subdirectory(bench)   # throughput benchmark (poe bench)
# bindings/ (nanobind)  -> added in Step 2

if(MD_BUILD_APP)
  add_subdirectory(app)
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
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Faithful Atari Missile Command clone (Qt 6 + Vulkan)")
  set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/JensKSP/missile-defense")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

  if(WIN32)
    # Windows installer (NSIS) + portable archive (ZIP). The install tree already
    # carries the bundled Qt/runtime DLLs and licence texts (see app/CMakeLists).
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Missile Defense")
    set(CPACK_PACKAGE_EXECUTABLES "md_app;Missile Defense") # Start-menu shortcut
    set(CPACK_NSIS_PACKAGE_NAME "Missile Defense")
    set(CPACK_NSIS_DISPLAY_NAME "Missile Defense")
    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "md_app.exe")         # offer to launch after install
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_URL_INFO_ABOUT "${CPACK_PACKAGE_HOMEPAGE_URL}")
    set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/packaging/missile-defense.ico")        # installer
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/packaging/missile-defense.ico")     # uninstaller
    set(CPACK_NSIS_INSTALLED_ICON_NAME "md_app.exe") # icon in Add/Remove Programs
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

  include(CPack)
endif()
