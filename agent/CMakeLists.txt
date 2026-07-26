# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# md_agent — the scripted reference AI (M4) and the evaluation protocol.
#
# A sibling of md::core, not part of it: the simulation must never depend on an
# agent. Depends only on the headless core, so it builds with MD_BUILD_APP=OFF and
# needs neither Qt nor Vulkan.
# nlohmann/json — for the `.mdp` manifest, and nothing else in this tree.
#
# A library rather than a hand-rolled parser because these files can arrive from
# outside: a policy someone downloaded is untrusted input, and a parser for
# untrusted input is the one place in this program where a bug is a security bug
# rather than a wrong number. Header-only and MIT, so it costs nothing to vet.
#
# Prefer the system copy (Debian `nlohmann-json3-dev`, MSYS2
# `mingw-w64-clang-x86_64-nlohmann-json`, Homebrew `nlohmann-json`), otherwise
# fetch — the same shape as Catch2 and miniaudio, so a distribution build never
# reaches the network and an offline developer build says why it cannot.
find_package(nlohmann_json 3.10 QUIET)
if(nlohmann_json_FOUND)
  message(STATUS "nlohmann_json: using the system package")
elseif(NOT FETCHCONTENT_FULLY_DISCONNECTED)
  message(STATUS "nlohmann_json: not found — fetching")
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
    SYSTEM)
  FetchContent_MakeAvailable(nlohmann_json)
else()
  message(FATAL_ERROR
    "nlohmann/json not found. Install nlohmann-json3-dev, or enable fetching "
    "with network access.")
endif()

add_library(md_agent)
add_library(md::agent ALIAS md_agent)

target_sources(md_agent
  PRIVATE
    src/heuristic.cpp
    src/eval.cpp
    src/policy.cpp)

target_include_directories(md_agent PUBLIC include)
target_compile_features(md_agent PUBLIC cxx_std_23)
# PRIVATE: the manifest parser is an implementation detail of policy.cpp, so
# nothing that links md::agent — the game included — inherits the dependency.
target_link_libraries(md_agent
  PUBLIC md::core
  PRIVATE md::warnings nlohmann_json::nlohmann_json)

# md::rl links this and is itself linked into the Python extension, a shared object.
if(CMAKE_POSITION_INDEPENDENT_CODE OR MD_BUILD_BINDINGS)
  set_target_properties(md_agent PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# The benchmark runner: prints the metrics the learned agent must beat. It uses
# std::println (libstdc++ 14), so a wheel built on an older compiler leaves it
# out — MD_BUILD_TOOLS is OFF there; the library above is all the wheel needs.
if(MD_BUILD_TOOLS)
  add_executable(md_agent_eval eval_main.cpp)
  target_link_libraries(md_agent_eval PRIVATE md::agent md::warnings)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
  if(MD_BUILD_TOOLS)
    add_test(NAME md_agent_eval_rejects_malformed_seed_offset
      COMMAND md_agent_eval --seed-offset not-a-number)
    set_tests_properties(md_agent_eval_rejects_malformed_seed_offset PROPERTIES
      LABELS unit
      WILL_FAIL TRUE)
  endif()
endif()
