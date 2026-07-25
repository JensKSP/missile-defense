# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Two things live here:
#   md_rl       — the RL environment (pure C++, no Python). Always built.
#   _md_native  — the nanobind extension wrapping it. Optional.
#
# md_rl is deliberately unconditional: the training loop sits on it, so it carries
# the same test gate as the core. Gating it behind MD_BUILD_BINDINGS would have
# meant the batch environment never compiled — let alone ran its tests — in a
# normal build.
add_library(md_rl)
add_library(md::rl ALIAS md_rl)

target_sources(md_rl PRIVATE vec_env.cpp)
target_include_directories(md_rl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(md_rl PUBLIC cxx_std_23)
# Threads::Threads arrives transitively (md_core links it PUBLIC).
target_link_libraries(md_rl PUBLIC md::core PRIVATE md::warnings)

# nanobind_add_module builds a shared object, so everything it links must be PIC.
if(CMAKE_POSITION_INDEPENDENT_CODE OR MD_BUILD_BINDINGS)
  set_target_properties(md_rl PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# ---- The Python extension (optional) -----------------------------------------
# Only built when MD_BUILD_BINDINGS is ON *and* Python + nanobind are actually
# present, so a plain C++ checkout never needs a Python toolchain.
if(NOT MD_BUILD_BINDINGS)
  return()
endif()

find_package(Python 3.11 COMPONENTS Interpreter Development.Module QUIET)
if(NOT Python_FOUND)
  message(STATUS "md bindings: Python development files not found — skipping")
  return()
endif()

# nanobind ships its CMake config inside the installed package.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE nanobind_probe)
if(NOT nanobind_probe EQUAL 0)
  message(STATUS "md bindings: nanobind not installed (pip install nanobind) — skipping")
  return()
endif()

find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_ROOT}" NO_DEFAULT_PATH)

nanobind_add_module(_md_native STABLE_ABI NB_STATIC
  module.cpp)

target_link_libraries(_md_native PRIVATE md::rl)

# The strict project warnings are deliberately *not* applied here: nanobind's
# generated glue trips several of them, and they are not our code to fix.
set_target_properties(_md_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/md")
