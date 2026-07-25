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
    src/vec_sim.cpp)

target_include_directories(md_core PUBLIC include)
target_compile_features(md_core PUBLIC cxx_std_23)

# VecSim steps a batch across a worker pool; everything else stays single-threaded.
find_package(Threads REQUIRED)
target_link_libraries(md_core PUBLIC Threads::Threads PRIVATE md::warnings)

# Note there is deliberately no in-code profiling hook here. `Sim::step` is a pure
# function of (state, action) on a fixed timestep — it reads no clock, and adding
# one distorted it 46x when tried (docs/PERFORMANCE.md). Profile it from outside
# with a sampler; the `profile` preset builds for exactly that.

# Single-source the semantic version from the top-level project(VERSION ...),
# so version.cpp and the CMake project version can never drift apart.
target_compile_definitions(md_core PRIVATE MD_VERSION="${PROJECT_VERSION}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
