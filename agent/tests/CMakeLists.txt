# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Catch2 and md_test_warnings come from the top-level CMakeLists.
if(NOT TARGET Catch2::Catch2WithMain)
  return()
endif()

add_executable(md_agent_tests
  unit/test_heuristic.cpp
  unit/test_policy.cpp)
# The fixture directory as a compile definition rather than a relative path:
# ctest makes no promise about the working directory, and a test that only finds
# its data when run from one place is a test that is skipped in CI.
target_compile_definitions(md_agent_tests
  PRIVATE MD_POLICY_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_link_libraries(md_agent_tests
  PRIVATE md::agent md::warnings md_test_warnings
          nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
catch_discover_tests(md_agent_tests PROPERTIES LABELS unit)
