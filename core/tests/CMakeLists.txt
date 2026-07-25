# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# ---- Test framework: Catch2 v3 ------------------------------------------------
# Prefer a system Catch2 (Debian/Ubuntu package `catch2`) so distro/offline
# builds do not fetch anything; otherwise fetch it into the build tree. If it is
# neither installed nor fetchable (offline with no package), skip the tests
# rather than failing the build.
find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
  if(Catch2_DIR)
    list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}") # Catch.cmake ships beside the config
  endif()
elseif(FETCHCONTENT_FULLY_DISCONNECTED)
  message(WARNING "Catch2 not found and fetching is disabled — skipping tests. "
                  "Install the 'catch2' package to build and run them.")
  return()
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
include(Catch)

# Clang 22+ (newer than the pinned Debian clang-21) flags Catch2's __COUNTER__
# use under -Wpedantic as a C2y extension. It comes from Catch2 (SYSTEM, third-
# party), so silence it — but the -Wno must sit AFTER md::warnings' -Wpedantic/
# -Werror on the command line, so carry it on its own interface target that each
# test links *after* md::warnings. Scoped to the tests; the rest of the tree keeps
# the pedantic check. No-op on compilers that don't emit the warning.
add_library(md_test_warnings INTERFACE)
target_compile_options(md_test_warnings INTERFACE
  "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,22>>:-Wno-c2y-extensions>")

# ---- Unit tests: fast, isolated (CTest label: unit) --------------------------
add_executable(md_core_unit_tests
  unit/test_vec2.cpp
  unit/test_rng.cpp
  unit/test_entities.cpp
  unit/test_action.cpp
  unit/test_sim.cpp
  unit/test_observation.cpp
  unit/test_intercept.cpp)
target_link_libraries(md_core_unit_tests
  PRIVATE md::core md::warnings md_test_warnings Catch2::Catch2WithMain)
catch_discover_tests(md_core_unit_tests PROPERTIES LABELS unit)

# ---- End-to-end tests: full-episode / integration (CTest label: e2e) ---------
add_executable(md_core_e2e_tests
  e2e/test_smoke_e2e.cpp)
target_link_libraries(md_core_e2e_tests
  PRIVATE md::core md::warnings md_test_warnings Catch2::Catch2WithMain)
catch_discover_tests(md_core_e2e_tests PROPERTIES LABELS e2e)
