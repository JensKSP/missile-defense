# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# ---- Test framework: Catch2 v3 (fetched + cached in the build tree) ----------
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  GIT_SHALLOW    TRUE
  SYSTEM)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ---- Unit tests: fast, isolated (CTest label: unit) --------------------------
add_executable(md_core_unit_tests
  unit/test_vec2.cpp
  unit/test_rng.cpp
  unit/test_entities.cpp
  unit/test_action.cpp
  unit/test_sim.cpp)
target_link_libraries(md_core_unit_tests
  PRIVATE md::core md::warnings Catch2::Catch2WithMain)
catch_discover_tests(md_core_unit_tests PROPERTIES LABELS unit)

# ---- End-to-end tests: full-episode / integration (CTest label: e2e) ---------
add_executable(md_core_e2e_tests
  e2e/test_smoke_e2e.cpp)
target_link_libraries(md_core_e2e_tests
  PRIVATE md::core md::warnings Catch2::Catch2WithMain)
catch_discover_tests(md_core_e2e_tests PROPERTIES LABELS e2e)
