# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# md_replay — recorded runs: the file format, and replaying one into a Sim.
#
# A sibling of md::core, like md::agent: the simulation itself knows nothing about
# files or episodes-as-artifacts. Shared by the RL environment (which writes
# recordings during training) and the app (which plays them back), which is why it
# is its own library rather than living in either.
add_library(md_replay)
add_library(md::replay ALIAS md_replay)

target_sources(md_replay PRIVATE src/recording.cpp src/match.cpp)

target_include_directories(md_replay PUBLIC include)
target_compile_features(md_replay PUBLIC cxx_std_23)
# nlohmann/json is PRIVATE: parsing a match manifest is an implementation
# detail of match.cpp, so nothing that links md::replay inherits it.
target_link_libraries(md_replay
  PUBLIC md::core
  PRIVATE md::warnings nlohmann_json::nlohmann_json)

# md_rl links this and is itself linked into the Python extension, a shared object.
if(CMAKE_POSITION_INDEPENDENT_CODE OR MD_BUILD_BINDINGS)
  set_target_properties(md_replay PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
