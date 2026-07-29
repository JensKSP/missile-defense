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
# Silence a vendored single-header we do not maintain. The two drivers spell it
# differently, and MSVC treats an unknown `-w` as a source file to compile.
target_compile_options(md_miniaudio PRIVATE $<IF:$<BOOL:${MSVC}>,/w,-w>)
# `m` is the Unix maths library. MSVC has no `m.lib` — the maths lives in the
# CRT — so linking it unconditionally is a hard `LNK1104: cannot open m.lib`,
# which is what the app hit the first time it was built with MSVC rather than
# MinGW. `CMAKE_DL_LIBS` is already empty on Windows and needs no such guard.
target_link_libraries(md_miniaudio PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(NOT MSVC)
  target_link_libraries(md_miniaudio PUBLIC m)
endif()

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
# what ships; the locators in tools/_util.py and missile_defense/runs/runner.py know the layout.
add_executable(md_app MACOSX_BUNDLE
  main.cpp
  trainer.cpp
  install.cpp
  instance.cpp
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

if(APPLE)
  # CFUserNotificationDisplayAlert, for the "this Mac has no Vulkan driver"
  # alert in main.cpp. Qt drags CoreFoundation in already and the link would
  # very likely succeed without this — which is the reason to name it: a
  # transitive dependency that happens to be there is not the same as one this
  # binary asked for, and the difference only shows up on the day Qt stops.
  target_link_libraries(md_app PRIVATE "-framework CoreFoundation")
endif()

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
  # The counterpart of MACOSX_BUNDLE above, and no more cosmetic than it is.
  # Without this the binary is linked for the console subsystem, so Windows opens
  # a black command window behind the game and leaves it there for as long as it
  # runs — reported from a real download of the nightly on 2026-07-27. Qt6::Core
  # links its WinMain shim in on exactly this property, so `main()` stays where it
  # is; what a terminal still needs is in main.cpp's attach_parent_console().
  set_target_properties(md_app PROPERTIES WIN32_EXECUTABLE ON)
  target_link_libraries(md_app PRIVATE ole32)
  # Embed the application icon (shown by Explorer, the taskbar and shortcuts).
  enable_language(RC)
  set(MD_ICON_ICO "${CMAKE_SOURCE_DIR}/packaging/missile-defense.ico")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/app.rc.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/app.rc" @ONLY)
  target_sources(md_app PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/app.rc")
endif()

set_target_properties(md_app PROPERTIES AUTOMOC ON)

if(BUILD_TESTING)
  add_subdirectory(tests)

  # The upstream-ownership witness for the xcb fallback in `app/main.cpp`. A bare
  # QVulkanWindow with none of this project in it, which still segfaults during
  # Qt's window teardown under Wayland; `test_wayland_teardown.py` runs it so the
  # fallback fails loudly the day Qt no longer needs it.
  #
  # Unlike `md_vulkan_baseline` this needs no validation layer — the exit status
  # is the measurement — so it is built wherever tests are. Never installed.
  if(UNIX AND NOT APPLE)
    add_executable(md_wayland_teardown tests/wayland_teardown.cpp)
    set_target_properties(md_wayland_teardown PROPERTIES AUTOMOC ON)
    target_link_libraries(md_wayland_teardown PRIVATE Qt6::Gui md::warnings)
  endif()
endif()

# Vulkan validation layers are opt-in (dev builds); never requested in a
# release/package build, so shipped binaries don't depend on the layer package.
if(MD_VULKAN_VALIDATION)
  target_compile_definitions(md_app PRIVATE MD_VULKAN_VALIDATION)

  # The upstream-ownership witness for the `vkQueueWaitIdle` in
  # `Renderer::submit`. A bare QVulkanWindow with none of this project in it;
  # `test_vulkan_validation.py` runs it to prove the VUID that workaround exists
  # for is Qt's, and to notice the day Qt stops raising it. Never installed, and
  # built only here, where the layer it needs is already a build-time choice.
  add_executable(md_vulkan_baseline tests/vulkan_baseline.cpp)
  set_target_properties(md_vulkan_baseline PROPERTIES AUTOMOC ON)
  target_link_libraries(md_vulkan_baseline PRIVATE Qt6::Gui md::warnings)
endif()

# ---- What TRAIN AI needs beside the game -------------------------------------
# Both are `game` component: they are what the *game* shows and hands to pip when
# someone chooses TRAIN AI, not part of a trainer that is no longer installed
# here at all.
#
# The page is everywhere, including Debian, because the notice screen opens it on
# every platform — the arcade font has no lower case, so a command or a URL
# cannot be shown on screen without being wrong to type.
if(WIN32)
  set(md_doc_dir ".")
elseif(APPLE)
  set(md_doc_dir "md_app.app/Contents/Resources")
else()
  include(GNUInstallDirs)
  set(md_doc_dir "${CMAKE_INSTALL_DATAROOTDIR}/missile-defense")
endif()
install(FILES "${CMAKE_SOURCE_DIR}/packaging/HOW-TO-TRAIN.html"
  DESTINATION "${md_doc_dir}" COMPONENT game)

# The wheel, on the two platforms where the game installs the trainer itself.
#
# Named rather than built: it is the artifact the `wheels` CI job already
# produces with cibuildwheel and publishes to PyPI, so what ships beside the game
# is the same file that was import-tested there. Building a second one here would
# produce something nobody has run.
#
# Empty is the ordinary state — a developer build ships no wheel, and `decide()`
# answers NeedsPackage, which is also the right answer on Debian.
set(MD_TRAINER_WHEEL "" CACHE FILEPATH
    "The missile_defense wheel to ship beside the game for TRAIN AI to install")
if(MD_TRAINER_WHEEL)
  if(NOT EXISTS "${MD_TRAINER_WHEEL}")
    message(FATAL_ERROR "MD_TRAINER_WHEEL names a file that does not exist: ${MD_TRAINER_WHEEL}")
  endif()
  # `app/install.cpp` looks for `missile_defense-*.whl` beside the binary, so the
  # name is the contract and a renamed copy would simply not be found.
  get_filename_component(md_wheel_name "${MD_TRAINER_WHEEL}" NAME)
  if(NOT md_wheel_name MATCHES "^missile_defense-.*\\.whl$")
    message(FATAL_ERROR
      "MD_TRAINER_WHEEL must be a file called missile_defense-<version>-<tags>.whl; "
      "app/install.cpp finds it by that name. Got: ${md_wheel_name}")
  endif()
  if(WIN32)
    install(FILES "${MD_TRAINER_WHEEL}" DESTINATION . COMPONENT game)
  elseif(APPLE)
    install(FILES "${MD_TRAINER_WHEEL}"
      DESTINATION "md_app.app/Contents/MacOS" COMPONENT game)
  else()
    message(WARNING
      "MD_TRAINER_WHEEL is set on a platform where the trainer is a package. "
      "Ignoring it: pip into a distribution's interpreter is what PEP 668 refuses.")
  endif()
endif()

# ---- The bundled learned agent -----------------------------------------------
# `models/pretrained.mdp` ships *with the game* — component `game`, not
# `python` — because playing against it needs no interpreter at all: it is a
# data-only file (docs/API.md §7) and `md::agent::Policy` reads it natively.
# That is the entire reason the format exists.
#
# **Optional on purpose**, and currently present: `Relational 620`, 1959-feature
# observation, matching this simulation. It was briefly absent — the observation
# encoding gained a feature in a100aec and every checkpoint trained before it
# became unrunnable — which is why this is a conditional rather than a hard
# dependency. A source tree with no model still builds and still ships; the game
# handles that state by leaving MODELS out of the WATCH AI menu rather than
# offering a list with nothing in it. `GameWindow::pretrained_path()` knows
# these three layouts, and lists this file alongside anything the trainer has
# promoted into `models/`.
include(GNUInstallDirs)
# Every `.mdp` in models/, not one named file: the game ships the same policy at
# three points in its training — novice, skilled, expert — so a player can watch
# one network acquire ammunition discipline rather than take the claim on trust.
# Discovered rather than listed, so adding a fourth is a file and not an edit.
file(GLOB MD_MODELS "${CMAKE_SOURCE_DIR}/models/*.mdp")
if(MD_MODELS)
  message(STATUS "md_app: bundling ${MD_MODELS}")
  if(WIN32)
    install(FILES ${MD_MODELS} DESTINATION models COMPONENT game)
  elseif(APPLE)
    # Inside the bundle: nothing beside an .app is guaranteed to travel with it.
    install(FILES ${MD_MODELS}
      DESTINATION "md_app.app/Contents/Resources/models" COMPONENT game)
  else()
    install(FILES ${MD_MODELS}
      DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/missile-defense/models" COMPONENT game)
  endif()
else()
  message(STATUS "md_app: no models/*.mdp — the game ships the scripted agent only")
endif()

# ---- Install / packaging -----------------------------------------------------
# Everything the game installs is tagged `COMPONENT game`, and everything the
# training half installs is tagged `python` (bindings/CMakeLists.txt and the
# MD_INSTALL_PYTHON_PACKAGE block in the top-level file). That split is not
# cosmetic: it is what lets `cmake --install --component game` stage the exact
# game-only product from a build tree that also built the trainer, which is both
# what the Windows/macOS installers offer as a choice and what
# python/tests/e2e/test_packages.py asserts is genuinely Python-free. Leaving
# these untagged put them in "Unspecified", where a component install of the
# game would silently have carried the native binding along with it.
if(WIN32)
  # Self-contained layout for the CPack NSIS installer / ZIP: the exe plus its
  # bundled Qt + C runtime DLLs (gathered by windeployqt at install time)
  # and the licence notices, all under the install root.
  #
  # Installed as `missile-defense.exe`, not as the target's own name. The build
  # tree keeps `md_app` — every locator that walks a checkout expects it
  # (tools/_util.py, missile_defense/runs/runner.py) and it is the same name on every platform
  # there — but what a person sees in Explorer, in the taskbar and in Task
  # Manager is this file, and `md_app` is an abbreviation nothing else in the
  # product ever shows them. install(PROGRAMS ... RENAME) rather than
  # install(TARGETS), which has no RENAME; the game links nothing that needs an
  # install-time RPATH, so the two are equivalent here apart from the name.
  install(PROGRAMS "$<TARGET_FILE:md_app>"
    DESTINATION .
    RENAME "missile-defense.exe"
    COMPONENT game)
  # `windeployqt` does the whole job under MSVC, which is why the shell script
  # that used to stand here is gone. It copied Qt's own DLLs and then walked the
  # dependency graph to a fixpoint for everything else that resolved inside the
  # MSYS2 CLANG64 prefix — libc++, ICU, HarfBuzz, FreeType, glib, PCRE2, zlib —
  # because a MinGW build drags its toolchain's runtime along and windeployqt
  # did not copy it. An MSVC build has no such tail: Qt's DLLs are the
  # dependency set, and `--compiler-runtime` covers the CRT.
  #
  # `vulkan-1.dll` is still deliberately not bundled: the loader ships with the
  # OS and the GPU driver and has to match them.
  if(TARGET Qt6::qmake)
    get_target_property(md_qmake Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(md_qt_bin "${md_qmake}" DIRECTORY)
  endif()
  find_program(WINDEPLOYQT NAMES windeployqt6 windeployqt HINTS "${md_qt_bin}" REQUIRED)
  install(CODE "
    message(STATUS \"windeployqt: bundling Qt + the C runtime ...\")
    execute_process(
      COMMAND \"${WINDEPLOYQT}\" --release --no-translations --no-opengl-sw
              --compiler-runtime \"\${CMAKE_INSTALL_PREFIX}/missile-defense.exe\"
      RESULT_VARIABLE _md_deploy_rc)
    if(NOT _md_deploy_rc EQUAL 0)
      message(FATAL_ERROR \"windeployqt failed (code \${_md_deploy_rc})\")
    endif()
  " COMPONENT game)
  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
                "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION . COMPONENT game)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses" DESTINATION . COMPONENT game)
elseif(APPLE)
  # macOS: a self-contained .app for the DMG, the same shape as the Windows
  # layout — the bundle carries its own Qt *and* its own Metal driver, and the
  # licence texts sit beside it. The work itself is in deploy_macos.cmake.in,
  # which runs at install time; everything here is finding what it needs.
  #
  # The installed bundle is renamed, for the reason the Windows branch renames
  # the exe. It was left as `md_app.app` on the theory that CFBundleName and
  # CFBundleDisplayName are what a user reads — they are not, in the one place
  # that matters most: the Finder labels a bundle with its *filename*, so an icon
  # dragged into /Applications sat there called "md_app" no matter what the
  # manifest said. The Dock and Launchpad do read the manifest, which is why this
  # was invisible to anyone who only ever launched it.
  #
  # A rename after the fact rather than an OUTPUT_NAME, because the build tree
  # must keep `md_app.app`: the locators in tools/_util.py and missile_defense/runs/runner.py
  # walk a checkout looking for that name, and a bundle with a space in it would
  # make every documented `./build/release/app/...` command need quoting.
  set(MD_MACOS_BUNDLE "Missile Defense.app")
  install(TARGETS md_app BUNDLE DESTINATION . COMPONENT game)
  install(CODE "
    set(md_built \"\${CMAKE_INSTALL_PREFIX}/md_app.app\")
    set(md_installed \"\${CMAKE_INSTALL_PREFIX}/${MD_MACOS_BUNDLE}\")
    if(NOT md_built STREQUAL md_installed AND IS_DIRECTORY \"\${md_built}\")
      file(REMOVE_RECURSE \"\${md_installed}\")
      file(RENAME \"\${md_built}\" \"\${md_installed}\")
      message(STATUS \"installed the game as ${MD_MACOS_BUNDLE}\")
    endif()
  " COMPONENT game)

  # macdeployqt is qtbase's equivalent of windeployqt: it copies the Qt
  # frameworks and the cocoa platform plugin into Contents/, then rewrites the
  # install names so the binary loads them from inside the bundle.
  if(TARGET Qt6::qmake)
    get_target_property(md_qmake Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(md_qt_bin "${md_qmake}" DIRECTORY)
  endif()
  find_program(MACDEPLOYQT macdeployqt HINTS "${md_qt_bin}")

  # The driver the loader dlopens. Frameworks are searched last on purpose:
  # molten-vk ships MoltenVK.xcframework beside the dylib, and find_library
  # prefers frameworks on Apple — the bundle needs the plain dylib.
  set(md_saved_find_framework "${CMAKE_FIND_FRAMEWORK}")
  set(CMAKE_FIND_FRAMEWORK NEVER)
  find_library(MD_MOLTENVK_LIBRARY NAMES MoltenVK)
  set(CMAKE_FIND_FRAMEWORK "${md_saved_find_framework}")

  # Its ICD manifest, taken from the same installation as the dylib rather than
  # searched for independently, so the two can never come from different copies.
  if(MD_MOLTENVK_LIBRARY)
    get_filename_component(md_mvk_libdir "${MD_MOLTENVK_LIBRARY}" DIRECTORY)
    get_filename_component(md_mvk_prefix "${md_mvk_libdir}" DIRECTORY)
    find_file(MD_MOLTENVK_ICD
      NAMES MoltenVK_icd.json
      PATHS "${md_mvk_prefix}/etc/vulkan/icd.d" "${md_mvk_prefix}/share/vulkan/icd.d"
      NO_DEFAULT_PATH)
  endif()

  # None of these is fatal here: building and running from the build tree needs
  # no deployment at all. Only installing and packaging do, and the script says
  # so there, where the message can be about the thing you were actually doing.
  if(NOT MACDEPLOYQT OR NOT MD_MOLTENVK_LIBRARY OR NOT MD_MOLTENVK_ICD)
    message(STATUS "md_app: macdeployqt and/or MoltenVK not found — the build is "
                   "fine, but `cmake --install` and `cpack` will fail")
  endif()

  configure_file(deploy_macos.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/deploy_macos.cmake" @ONLY)
  install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/deploy_macos.cmake" COMPONENT game)

  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
                "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION . COMPONENT game)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses" DESTINATION . COMPONENT game)
else()
  # Debian: install to the games dir as `missile-defense`, with a desktop entry.
  include(GNUInstallDirs)
  install(PROGRAMS "$<TARGET_FILE:md_app>"
    DESTINATION games
    RENAME missile-defense
    COMPONENT game)
  install(FILES "${CMAKE_SOURCE_DIR}/packaging/missile-defense.desktop"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/applications" COMPONENT game)
  # Icons for GNOME/KDE menus (hicolor theme; the .desktop's Icon= resolves here).
  #
  # The trainer's emblem lives in the same theme directory and is *not* the
  # game's to install: it belongs to the `python` component, which the top-level
  # file installs beside the launchers it goes with. A game-only install must
  # carry no trace of the trainer, and `usr/share/icons/hicolor` claimed whole
  # would have put its icon in the game's .deb.
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/packaging/icons/hicolor"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/icons" COMPONENT game
    PATTERN "missile-defense-trainer.png" EXCLUDE)
  install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
                "${CMAKE_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/missile-defense" COMPONENT game)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/licenses"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/missile-defense" COMPONENT game)
endif()
