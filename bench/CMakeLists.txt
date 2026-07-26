# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# md_bench — throughput benchmark: how many simulations can this machine run?
# Headless (core + agent only), so it builds with MD_BUILD_APP=OFF.
# A CLI tool, not part of the wheel — MD_BUILD_TOOLS is OFF for that build.
if(MD_BUILD_TOOLS)
  add_executable(md_bench main.cpp)
  target_link_libraries(md_bench PRIVATE md::core md::agent md::warnings)
endif()
