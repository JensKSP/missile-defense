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
    src/intercept.cpp
    src/profile.cpp
    src/vec_sim.cpp)

target_include_directories(md_core PUBLIC include)
target_compile_features(md_core PUBLIC cxx_std_23)

# VecSim steps a batch across a worker pool; everything else stays single-threaded.
find_package(Threads REQUIRED)
target_link_libraries(md_core PUBLIC Threads::Threads PRIVATE md::warnings)

# Per-phase timers in Sim::step. PUBLIC so anything reading md::prof counters sees
# the same switch, and OFF by default — the shipping hot path carries no timers.
if(MD_PROFILE)
  target_compile_definitions(md_core PUBLIC MD_PROFILE)
endif()

# Single-source the semantic version from the top-level project(VERSION ...),
# so version.cpp and the CMake project version can never drift apart.
target_compile_definitions(md_core PRIVATE MD_VERSION="${PROJECT_VERSION}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
