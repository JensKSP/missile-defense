# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# app/ — the human client (Qt 6 + Vulkan). Depends on the pure md::core sim.
find_package(Qt6 QUIET COMPONENTS Gui)
find_package(Vulkan QUIET)

if(NOT Qt6_FOUND OR NOT Vulkan_FOUND)
  message(STATUS "md_app: Qt6 or Vulkan not found — skipping the app target "
                 "(build core-only with -DMD_BUILD_APP=OFF to silence this)")
  return()
endif()

find_program(GLSLANG glslangValidator REQUIRED)

# Compile a GLSL shader to SPIR-V embedded as a C header (const uint32_t array),
# so shaders ship inside the binary with no runtime file loading.
set(shader_hdr_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${shader_hdr_dir}")
set(shader_headers "")
function(md_embed_shader src var)
  set(hdr "${shader_hdr_dir}/${var}.h")
  add_custom_command(
    OUTPUT "${hdr}"
    COMMAND ${GLSLANG} -V --vn ${var} -o "${hdr}" "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
    COMMENT "Compiling shader ${src} -> ${var}.h"
    VERBATIM)
  set(shader_headers "${shader_headers};${hdr}" PARENT_SCOPE)
endfunction()

md_embed_shader(shaders/quad.vert quad_vert_spv)
md_embed_shader(shaders/quad.frag quad_frag_spv)

# miniaudio (single-header). Prefer the system copy (Debian/Ubuntu package
# `libminiaudio-dev`), otherwise fetch it — so distro/offline builds use the
# system header and never an embedded copy. No shared lib exists for miniaudio;
# we compile its implementation ourselves in miniaudio_impl.cpp.
find_path(MINIAUDIO_INCLUDE_DIR miniaudio.h)
if(MINIAUDIO_INCLUDE_DIR)
  message(STATUS "miniaudio: using system header (${MINIAUDIO_INCLUDE_DIR})")
elseif(MD_FETCH_MINIAUDIO AND NOT FETCHCONTENT_FULLY_DISCONNECTED)
  message(STATUS "miniaudio: system header not found — fetching")
  include(FetchContent)
  FetchContent_Declare(miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        0.11.22
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake-skip) # download only; do not add miniaudio's own CMake
  FetchContent_MakeAvailable(miniaudio)
  set(MINIAUDIO_INCLUDE_DIR "${miniaudio_SOURCE_DIR}")
else()
  message(FATAL_ERROR
    "miniaudio not found. Install libminiaudio-dev, or enable MD_FETCH_MINIAUDIO "
    "with network access.")
endif()

find_package(Threads REQUIRED)
add_library(md_miniaudio STATIC miniaudio_impl.cpp)
target_include_directories(md_miniaudio SYSTEM PUBLIC "${MINIAUDIO_INCLUDE_DIR}")
target_compile_options(md_miniaudio PRIVATE -w)
target_link_libraries(md_miniaudio PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)

add_executable(md_app
  main.cpp
  game_window.cpp
  renderer.cpp
  audio.cpp
  highscores.cpp
  ${shader_headers})

target_include_directories(md_app PRIVATE "${shader_hdr_dir}")

# Qt/Vulkan headers are imported targets → their includes are SYSTEM, so our
# strict -Werror only holds our own code to account.
target_link_libraries(md_app
  PRIVATE md::core md::warnings Qt6::Gui Vulkan::Vulkan md_miniaudio)

set_target_properties(md_app PROPERTIES AUTOMOC ON)

# Vulkan validation layers are opt-in (dev builds); never requested in a
# release/package build, so shipped binaries don't depend on the layer package.
if(MD_VULKAN_VALIDATION)
  target_compile_definitions(md_app PRIVATE MD_VULKAN_VALIDATION)
endif()

# ---- Install (for `cpack -G DEB`) --------------------------------------------
# The binary is built as `md_app` (scripts reference it) but installs to the
# Debian games dir as `missile-defense`, with a desktop entry.
include(GNUInstallDirs)
install(PROGRAMS "$<TARGET_FILE:md_app>"
  DESTINATION games
  RENAME missile-defense)
install(FILES "${CMAKE_SOURCE_DIR}/packaging/missile-defense.desktop"
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications")
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/missile-defense")
