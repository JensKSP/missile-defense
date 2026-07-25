# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Catch2 and the shared md_test_warnings target are set up once by the top-level
# CMakeLists; if it could not be found or fetched, there is nothing to build here.
if(NOT TARGET Catch2::Catch2WithMain)
  return()
endif()

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
