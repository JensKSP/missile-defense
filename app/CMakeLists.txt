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

# Vendored miniaudio (single header) compiled in its own library with warnings
# off — it is large third-party C. Its include dir is SYSTEM so our own code
# (audio.cpp) can use miniaudio.h while staying under strict -Werror.
find_package(Threads REQUIRED)
add_library(md_miniaudio STATIC miniaudio_impl.cpp)
target_include_directories(md_miniaudio SYSTEM PUBLIC
  "${CMAKE_SOURCE_DIR}/third_party/miniaudio")
target_compile_options(md_miniaudio PRIVATE -w)
target_link_libraries(md_miniaudio PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)

add_executable(md_app
  main.cpp
  game_window.cpp
  renderer.cpp
  audio.cpp
  ${shader_headers})

target_include_directories(md_app PRIVATE "${shader_hdr_dir}")

# Qt/Vulkan headers are imported targets → their includes are SYSTEM, so our
# strict -Werror only holds our own code to account.
target_link_libraries(md_app
  PRIVATE md::core md::warnings Qt6::Gui Vulkan::Vulkan md_miniaudio)

set_target_properties(md_app PROPERTIES AUTOMOC ON)
