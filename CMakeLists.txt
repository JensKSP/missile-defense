cmake_minimum_required(VERSION 3.25)

project(missile_defense
  VERSION 0.1.0
  DESCRIPTION "Faithful Atari Missile Command clone — game + RL environment"
  LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json (clangd / clang-tidy consume it).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Options -----------------------------------------------------------------
option(MD_SANITIZE "Enable AddressSanitizer + UBSan" OFF)
option(MD_WERROR "Treat warnings as errors" ON)

# ---- Warnings ----------------------------------------------------------------
# Opt-in via `target_link_libraries(<t> PRIVATE md::warnings)`. Keeps strict
# flags on our own code without imposing them on third-party deps (e.g. Catch2).
add_library(md_warnings INTERFACE)
add_library(md::warnings ALIAS md_warnings)
target_compile_options(md_warnings INTERFACE
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wold-style-cast
  -Wcast-align
  -Wunused
  -Woverloaded-virtual
  -Wnon-virtual-dtor
  -Wnull-dereference
  -Wdouble-promotion
  -Wformat=2
  -Wimplicit-fallthrough
  -Wmisleading-indentation)
if(MD_WERROR)
  target_compile_options(md_warnings INTERFACE -Werror)
endif()

# ---- Sanitizers --------------------------------------------------------------
# Global on purpose: instrument our code *and* the deps it links under test.
if(MD_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- Testing -----------------------------------------------------------------
include(CTest)  # defines BUILD_TESTING (ON by default) and calls enable_testing()

# ---- Components ---------------------------------------------------------------
add_subdirectory(core)
# bindings/ (nanobind)  -> added in Step 2
# app/      (Qt+Vulkan) -> added when the core sim is frozen
