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

# macOS: miniaudio's only backend here is CoreAudio, which it reaches through
# these three frameworks. Unlike the Windows COM case there is no source-level
# quirk to match (see audio.cpp) — the device just needs the frameworks linked.
if(APPLE)
  target_link_libraries(md_miniaudio PUBLIC
    "-framework CoreFoundation" "-framework CoreAudio" "-framework AudioToolbox")
endif()

# MACOSX_BUNDLE is ignored off Apple. On macOS it is not cosmetic: a bare Mach-O
# executable never becomes a foreground GUI process, so it gets no Dock icon and
# cannot reliably take keyboard focus — fatal for a game. Building the bundle in
# the dev tree too (not just at install time) keeps what you play identical to
# what ships; the locators in tools/_util.py and md/ui/runner.py know the layout.
add_executable(md_app MACOSX_BUNDLE
  main.cpp
  game_window.cpp
  renderer.cpp
  audio.cpp
  highscores.cpp
  ${shader_headers})

target_include_directories(md_app PRIVATE "${shader_hdr_dir}")

# Qt/Vulkan headers are imported targets → their includes are SYSTEM, so our
# strict -Werror only holds our own code to account.
# md::agent supplies the scripted baseline for the app's WATCH AI mode. It is a
# headless sibling of the core — no Qt, no Vulkan — so this costs nothing but a
# static link.
target_link_libraries(md_app
  PRIVATE md::core md::agent md::replay md::warnings Qt6::Gui Vulkan::Vulkan md_miniaudio)

# macOS: fill in the bundle manifest (app/Info.plist.in) and drop the icon into
# Contents/Resources. The bundle directory stays `md_app.app` so the build tree
# looks the same on every platform; the name a user sees comes from CFBundleName
# / CFBundleDisplayName, not from the directory.
if(APPLE)
  set(MD_ICON_ICNS "${CMAKE_SOURCE_DIR}/packaging/missile-defense.icns")
  target_sources(md_app PRIVATE "${MD_ICON_ICNS}")
  set_source_files_properties("${MD_ICON_ICNS}" PROPERTIES
    MACOSX_PACKAGE_LOCATION Resources)
  set_target_properties(md_app PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
    MACOSX_BUNDLE_BUNDLE_NAME "Missile Defense"
    MACOSX_BUNDLE_GUI_IDENTIFIER "de.koehler-speyer.missile-defense"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_ICON_FILE "missile-defense.icns"
    MACOSX_BUNDLE_COPYRIGHT "Copyright (c) 2026 Jens Köhler. MIT License.")
endif()

# Windows: the audio device init explicitly manages a COM apartment (audio.cpp).
if(WIN32)
  target_link_libraries(md_app PRIVATE ole32)
  # Embed the application icon (shown by Explorer, the taskbar and shortcuts).
  enable_language(RC)
  set(MD_ICON_ICO "${CMAKE_SOURCE_DIR}/packaging/missile-defense.ico")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/app.rc.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/app.rc" @ONLY)
  target_sources(md_app PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/app.rc")
endif()

set_target_properties(md_app PROPERTIES AUTOMOC ON)

# Vulkan validation layers are opt-in (dev builds); never requested in a
# release/package build, so shipped binaries don't depend on the layer package.
if(MD_VULKAN_VALIDATION)
  target_compile_definitions(md_app PRIVATE MD_VULKAN_VALIDATION)
endif()

# ---- Install / packaging -----------------------------------------------------
if(WIN32)
  # Self-contained layout for the CPack NSIS installer / ZIP: the exe plus its
  # bundled Qt + runtime DLLs (gathered by tools/windeploy.sh at install time)
  # and the licence notices, all under the install root.
  install(TARGETS md_app RUNTIME DESTINATION .)
  find_program(BASH_PROGRAM bash REQUIRED)
  install(CODE "
    message(STATUS \"windeploy: bundling Qt + runtime DLLs ...\")
    execute_process(
      COMMAND \"${BASH_PROGRAM}\" \"${CMAKE_SOURCE_DIR}/tools/windeploy.sh\"
              \"\${CMAKE_INSTALL_PREFIX}/md_app.exe\"
      RESULT_VARIABLE _md_deploy_rc)
    if(NOT _md_deploy_rc EQUAL 0)
      message(FATAL_ERROR \"tools/windeploy.sh failed (code \${_md_deploy_rc})\")
    endif()
  ")
  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
                "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION .)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses" DESTINATION .)
elseif(APPLE)
  # macOS: the .app for the DMG, the same shape as the Windows layout — the
  # bundle carries its own Qt, and the licence texts sit beside it. It does not
  # carry MoltenVK: macdeployqt bundles what the binary *links*, and the Metal
  # driver is dlopened by the Vulkan loader from an ICD manifest outside the
  # bundle. So this .app still needs MoltenVK on the target machine — the one
  # gap between it and the Windows/Debian packages (docs/MACOS.md).
  install(TARGETS md_app BUNDLE DESTINATION .)

  # macdeployqt is qtbase's equivalent of tools/windeploy.sh: it copies the Qt
  # frameworks and the cocoa platform plugin into Contents/, then rewrites the
  # install names so the binary loads them from inside the bundle.
  if(TARGET Qt6::qmake)
    get_target_property(md_qmake Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(md_qt_bin "${md_qmake}" DIRECTORY)
  endif()
  find_program(MACDEPLOYQT macdeployqt HINTS "${md_qt_bin}")
  if(NOT MACDEPLOYQT)
    # Not fatal here: building and running from the build tree needs no
    # deployment at all. Only installing/packaging does, so fail there instead.
    message(STATUS "md_app: macdeployqt not found — `cmake --install` and "
                   "`cpack` will fail until Qt's bin directory is on PATH")
  endif()
  install(CODE "
    set(md_macdeployqt \"${MACDEPLOYQT}\")
    if(NOT md_macdeployqt)
      message(FATAL_ERROR \"macdeployqt not found: cannot bundle Qt into the .app\")
    endif()
    set(md_bundle \"\${CMAKE_INSTALL_PREFIX}/md_app.app\")
    message(STATUS \"macdeployqt: bundling Qt into \${md_bundle} ...\")
    execute_process(
      COMMAND \"\${md_macdeployqt}\" \"\${md_bundle}\" -always-overwrite
      RESULT_VARIABLE md_deploy_rc)
    if(NOT md_deploy_rc EQUAL 0)
      message(FATAL_ERROR \"macdeployqt failed (code \${md_deploy_rc})\")
    endif()
    # Re-sign ad-hoc. The linker ad-hoc signs every arm64 binary it produces, and
    # macdeployqt then rewrites the load commands, which invalidates that
    # signature — an arm64 binary with a broken signature is killed on launch.
    # This is not Developer ID signing: the DMG is still unsigned and unnotarised
    # (see docs/MACOS.md), it just runs once the user clears quarantine.
    execute_process(
      COMMAND codesign --force --deep --sign - \"\${md_bundle}\"
      RESULT_VARIABLE md_sign_rc)
    if(NOT md_sign_rc EQUAL 0)
      message(FATAL_ERROR \"ad-hoc codesign failed (code \${md_sign_rc})\")
    endif()
  ")

  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
                "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION .)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses" DESTINATION .)
else()
  # Debian: install to the games dir as `missile-defense`, with a desktop entry.
  include(GNUInstallDirs)
  install(PROGRAMS "$<TARGET_FILE:md_app>"
    DESTINATION games
    RENAME missile-defense)
  install(FILES "${CMAKE_SOURCE_DIR}/packaging/missile-defense.desktop"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications")
  # Icons for GNOME/KDE menus (hicolor theme; the .desktop's Icon= resolves here).
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/packaging/icons/hicolor"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons")
  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
                "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/missile-defense")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/missile-defense")
endif()
