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

# ---- Global warning flags (project code only; see per-target settings) -------
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)

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
