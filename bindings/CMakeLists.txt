# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# _md_native — the nanobind extension exposing the simulation to Python.
#
# Optional: only built when MD_BUILD_BINDINGS is ON *and* Python + nanobind are
# actually present, so a plain C++ checkout never needs a Python toolchain.
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
  module.cpp
  vec_env.cpp)

target_link_libraries(_md_native PRIVATE md::core)
target_include_directories(_md_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# The strict project warnings are deliberately *not* applied here: nanobind's
# generated glue trips several of them, and they are not our code to fix.
set_target_properties(_md_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/md")
