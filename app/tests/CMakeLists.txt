# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: OpenAI Codex
if(NOT TARGET Catch2::Catch2WithMain)
  return()
endif()

add_executable(md_app_unit_tests
  unit/test_trainer.cpp
  unit/test_install.cpp
  unit/test_instance.cpp
  unit/test_human_input.cpp
  unit/test_terrain.cpp
  unit/test_voices.cpp
  ../trainer.cpp
  ../install.cpp
  ../instance.cpp)
target_include_directories(md_app_unit_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
# Threads: instance.cpp serves its channel on a std::thread of its own.
target_link_libraries(md_app_unit_tests
  PRIVATE md::core md::warnings md_test_warnings Catch2::Catch2WithMain Threads::Threads)
catch_discover_tests(md_app_unit_tests PROPERTIES LABELS unit)
