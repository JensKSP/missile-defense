# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Catch2 and md_test_warnings come from the top-level CMakeLists.
if(NOT TARGET Catch2::Catch2WithMain)
  return()
endif()

add_executable(md_agent_tests
  unit/test_heuristic.cpp)
target_link_libraries(md_agent_tests
  PRIVATE md::agent md::warnings md_test_warnings Catch2::Catch2WithMain)
catch_discover_tests(md_agent_tests PROPERTIES LABELS unit)
