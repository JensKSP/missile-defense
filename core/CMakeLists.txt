# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# md_core — the pure, headless game simulation library.
# No Qt, no rendering, no Python. This is the single source of truth for game rules.
add_library(md_core)
add_library(md::core ALIAS md_core)

target_sources(md_core
  PRIVATE
    src/version.cpp
    src/sim.cpp
    src/observation.cpp
    src/intercept.cpp)

target_include_directories(md_core PUBLIC include)
target_compile_features(md_core PUBLIC cxx_std_23)
target_link_libraries(md_core PRIVATE md::warnings)

# Single-source the semantic version from the top-level project(VERSION ...),
# so version.cpp and the CMake project version can never drift apart.
target_compile_definitions(md_core PRIVATE MD_VERSION="${PROJECT_VERSION}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
