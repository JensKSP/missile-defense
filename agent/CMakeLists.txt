# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# md_agent — the scripted reference AI (M4) and the evaluation protocol.
#
# A sibling of md::core, not part of it: the simulation must never depend on an
# agent. Depends only on the headless core, so it builds with MD_BUILD_APP=OFF and
# needs neither Qt nor Vulkan.
add_library(md_agent)
add_library(md::agent ALIAS md_agent)

target_sources(md_agent
  PRIVATE
    src/heuristic.cpp
    src/eval.cpp)

target_include_directories(md_agent PUBLIC include)
target_compile_features(md_agent PUBLIC cxx_std_23)
target_link_libraries(md_agent PUBLIC md::core PRIVATE md::warnings)

# The benchmark runner: prints the metrics the learned agent must beat.
add_executable(md_agent_eval eval_main.cpp)
target_link_libraries(md_agent_eval PRIVATE md::agent md::warnings)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
