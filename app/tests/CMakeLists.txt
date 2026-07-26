# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
if(NOT TARGET Catch2::Catch2WithMain)
  return()
endif()

add_executable(md_app_unit_tests
  unit/test_console.cpp
  unit/test_human_input.cpp
  ../console.cpp)
target_include_directories(md_app_unit_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(md_app_unit_tests
  PRIVATE md::core md::warnings md_test_warnings Catch2::Catch2WithMain)
catch_discover_tests(md_app_unit_tests PROPERTIES LABELS unit)
