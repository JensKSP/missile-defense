# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
# Two things live here:
#   md_rl       — the RL environment (pure C++, no Python). Always built.
#   _md_native  — the nanobind extension wrapping it. Optional.
#
# md_rl is deliberately unconditional: the training loop sits on it, so it carries
# the same test gate as the core. Gating it behind MD_BUILD_BINDINGS would have
# meant the batch environment never compiled — let alone ran its tests — in a
# normal build.
add_library(md_rl)
add_library(md::rl ALIAS md_rl)

target_sources(md_rl PRIVATE vec_env.cpp)
target_include_directories(md_rl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(md_rl PUBLIC cxx_std_23)
# Threads::Threads arrives transitively (md_core links it PUBLIC).
target_link_libraries(md_rl PUBLIC md::core md::replay md::agent PRIVATE md::warnings)

# nanobind_add_module builds a shared object, so everything it links must be PIC.
if(CMAKE_POSITION_INDEPENDENT_CODE OR MD_BUILD_BINDINGS)
  set_target_properties(md_rl PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# ---- The Python extension (optional) -----------------------------------------
# Only built when MD_BUILD_BINDINGS is ON *and* Python + nanobind are actually
# present, so a plain C++ checkout never needs a Python toolchain.
if(NOT MD_BUILD_BINDINGS)
  return()
endif()

# Development.SABIModule is what creates the `Python::SABIModule` target, and
# that target is one of the three things nanobind checks before honouring
# STABLE_ABI below — the others being CPython and >= 3.12. Ask for it and the
# module is `_md_native.abi3.so`; leave it out and nanobind *silently* drops back
# to a version-tagged `_md_native.cpython-313-x86_64-linux-gnu.so`, which is what
# it had been doing here since the day STABLE_ABI was written. Nothing failed:
# every build worked, on the interpreter that built it, and the claim in
# docs/PACKAGING.md about surviving a Python upgrade was simply not true.
#
# OPTIONAL_COMPONENTS, because the floor is 3.11 and the stable ABI starts at
# 3.12. A machine below that must still be able to build the extension; it just
# gets one tied to its own interpreter. `md_native_is_abi3` below is how the
# packaging build tells the difference, instead of trusting that it got it.
find_package(Python 3.11
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
  QUIET)
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
  module.cpp)

# What STABLE_ABI actually produced. nanobind decides silently and records the
# answer in the target's suffix, so that is what gets read — a build that
# *asked* for abi3 proves nothing, and asking was the bug. The top-level file
# turns this into a hard error for a packaging build; here it is only reported,
# because a developer on 3.11 has nothing to fix.
#
# Compared against nanobind's own `NB_SUFFIX_S` rather than matched for the
# string "abi3", because that string is not in the name on every platform: the
# limited-API suffix is `.abi3.so` on Linux and macOS but a bare `.pyd` on
# Windows, which is what CPython's importer looks for there. Matching the
# literal would have reported every stable-ABI Windows build as a failure.
get_target_property(md_native_suffix _md_native SUFFIX)
if(md_native_suffix STREQUAL NB_SUFFIX_S)
  set(MD_NATIVE_IS_ABI3 ON CACHE INTERNAL "The extension was built against the stable ABI")
  message(STATUS "md bindings: _md_native${md_native_suffix} (stable ABI)")
else()
  set(MD_NATIVE_IS_ABI3 OFF CACHE INTERNAL "The extension was built against the stable ABI")
  message(STATUS
    "md bindings: _md_native${md_native_suffix} — tied to this interpreter. The "
    "stable ABI needs CPython >= 3.12 with its SABI development files; this is "
    "Python ${Python_VERSION} (${Python_INTERPRETER_ID}).")
endif()

target_link_libraries(_md_native PRIVATE md::rl)

# The strict project warnings are deliberately *not* applied here: nanobind's
# generated glue trips several of them, and they are not our code to fix.
set_target_properties(_md_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/missile_defense")

# ---- Beside the Python package, so a checkout is importable ------------------
# `missile_defense.sim.env` imports this as `missile_defense._md_native`, so the object has to sit next to
# python/missile_defense/. That copy used to be a Python script run after the fact, which made
# `cmake --build` produce a module nothing could import until a second, separate
# tool was run over it. It is a build step, so it belongs to the build. The
# destination is gitignored: writing into the source tree is the point, because
# it is what makes pytest and an interactive session work from a plain checkout
# with no install step and no PYTHONPATH.
set(md_native_built "_md_native${md_native_suffix}")
set(md_native_beside "${CMAKE_SOURCE_DIR}/python/missile_defense")

# What this build *supersedes* — never simply "everything called _md_native*".
# A module carrying a different ABI tag is a second toolchain's, deliberately
# kept: on Windows an MSYS2 build and an MSVC one coexist (docs/TRAINING.md) and
# an interpreter imports only its own tag, so deleting the other one breaks that
# toolchain and buys nothing. Two names do have to go:
#   * the untagged `_md_native.so`, a stale shadow of this same module;
#   * the version-tagged name, but only when this build is abi3 — importlib tries
#     the version-tagged suffix first, so a checkout that once built the tagged
#     module would keep importing it for ever, including the run meant to prove
#     the stable ABI works.
set(md_native_superseded "_md_native${CMAKE_SHARED_MODULE_SUFFIX}")
if(MD_NATIVE_IS_ABI3)
  list(APPEND md_native_superseded
       "_md_native.${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
endif()
list(REMOVE_ITEM md_native_superseded "${md_native_built}")
list(REMOVE_DUPLICATES md_native_superseded)

set(md_native_cleanup "")
foreach(stale IN LISTS md_native_superseded)
  list(APPEND md_native_cleanup
       COMMAND "${CMAKE_COMMAND}" -E rm -f "${md_native_beside}/${stale}")
endforeach()

# The placed file is declared as an OUTPUT rather than done in a POST_BUILD step,
# and the difference is the whole reliability of this: POST_BUILD runs only when
# the target relinks, so deleting python/missile_defense/_md_native.* and building again would
# leave it deleted and report success. As an output with the module as its
# dependency, it is something the build knows it owes you — removed by hand, it
# comes back on the next build, like any other artefact.
add_custom_command(
  OUTPUT "${md_native_beside}/${md_native_built}"
  DEPENDS _md_native
  ${md_native_cleanup}
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "$<TARGET_FILE:_md_native>" "${md_native_beside}/"
  COMMENT "Placing ${md_native_built} beside python/missile_defense/"
  VERBATIM)
add_custom_target(md_native_beside_package ALL
  DEPENDS "${md_native_beside}/${md_native_built}")

# ---- Installing it -----------------------------------------------------------
# The module only ever landed in the build tree, which is right for in-place
# development and leaves nothing for a *package* to install. Consumers want the
# same thing in different places, so the destination is a variable:
#
#   scikit-build-core  takes the install tree as the wheel root, so a relative
#                      `md` puts the module inside the package.
#   debian/rules       passes an absolute dist-packages path.
#
# STABLE_ABI above is what makes this worth installing at all: the object is
# abi3, so it survives the distribution's Python moving a minor version and does
# not have to be rebuilt in lockstep with it (docs/PACKAGING.md).
# `MD_PYTHON_INSTALL_DIR` is declared by the top-level CMakeLists, before this
# directory is added. It has to be: the install rule below is registered when
# this file is read, so a destination set afterwards never reaches it.
#
# Empty means nobody has said where the Python half goes — an ordinary
# `cmake --install` on Linux, where only the distribution knows. Then there is
# no rule to register: the extension is built and placed beside python/missile_defense/ for
# development, and installing the tree quietly leaves it alone. Guessing instead
# is what produced `/usr/local/md/_md_native.abi3.so`, a module in a directory
# no interpreter looks in. Everything below is the install; nothing follows it,
# so this returns rather than nesting the file in an `if`.
if(NOT MD_PYTHON_INSTALL_DIR)
  message(STATUS
    "md bindings: _md_native is built but will not be installed — set "
    "MD_PYTHON_INSTALL_DIR to the interpreter's site-packages to install it "
    "(debian/rules does; a wheel does not need to).")
  return()
endif()

# ---- Installing a module built somewhere else --------------------------------
# Windows needs this, and only Windows. The game is built in MSYS2/CLANG64 —
# that is what Qt and Vulkan are there — so the extension built alongside it is a
# mingw object linked against MSYS2's libc++ and MSYS2's interpreter. The
# installed trainer does not run in MSYS2: `missile-defense-trainer.cmd` execs whatever
# `python` is on PATH, which for anyone who followed docs/WINDOWS.md is a
# python.org CPython, because that is where the PySide6 and torch wheels go.
# Those two do not mix, and the failure is a DLL load error inside an import.
#
# So the Windows job builds `_md_native` a second time against that interpreter
# (the `win-native` preset, which exists for the same reason) and names the
# result here. The stable ABI makes the swap total rather than partial: both
# builds produce the identical filename, so nothing downstream has to know which
# one it got.
set(MD_PREBUILT_PYTHON_MODULE "" CACHE FILEPATH
    "A _md_native built elsewhere to install instead of this build's own")

if(MD_PREBUILT_PYTHON_MODULE)
  if(NOT EXISTS "${MD_PREBUILT_PYTHON_MODULE}")
    message(FATAL_ERROR
      "MD_PREBUILT_PYTHON_MODULE names a file that does not exist: "
      "${MD_PREBUILT_PYTHON_MODULE}")
  endif()
  # Installed under the name it already has. That name is not decoration: an
  # extension's suffix is how CPython decides whether the file is a module for
  # this interpreter at all, and the build that produced it is the only thing
  # that knows the right one — `_md_native.pyd` for a stable-ABI Windows build,
  # `_md_native.abi3.so` elsewhere. Rewriting it here from the last extension
  # was worse than doing nothing: it turned an `.abi3.so` into a `.so`, which
  # CPython loads last if at all.
  #
  # So the check is on the name rather than a correction of it. A path pointing
  # at something that is not this module is a mistake in the job that set it,
  # and it should stop the configure rather than install quietly.
  get_filename_component(md_prebuilt_name "${MD_PREBUILT_PYTHON_MODULE}" NAME)
  if(NOT md_prebuilt_name MATCHES "^_md_native\\.")
    message(FATAL_ERROR
      "MD_PREBUILT_PYTHON_MODULE must name a file called _md_native.<suffix> — "
      "CPython finds an extension by its filename, and ${md_prebuilt_name} is "
      "not one it would import as missile_defense._md_native.")
  endif()
  message(STATUS
    "md bindings: installing the prebuilt ${MD_PREBUILT_PYTHON_MODULE}")
  install(FILES "${MD_PREBUILT_PYTHON_MODULE}"
          DESTINATION "${MD_PYTHON_INSTALL_DIR}"
          COMPONENT python)
else()
  # COMPONENT is repeated inside each artifact clause rather than written once at
  # the end. A trailing COMPONENT after per-artifact destinations binds only to the
  # clause before it, so the one form that looks tidiest quietly files the module
  # under "Unspecified" and `--component python` installs the stub alone.
  install(TARGETS _md_native
          LIBRARY DESTINATION "${MD_PYTHON_INSTALL_DIR}" COMPONENT python
          RUNTIME DESTINATION "${MD_PYTHON_INSTALL_DIR}" COMPONENT python)
endif()

# The stub travels with the extension rather than with the .py sources: it
# describes *this* module, and a package that installs one without the other
# leaves an editor with no idea what `_md_native` exposes.
install(FILES "${CMAKE_SOURCE_DIR}/python/missile_defense/_md_native.pyi"
        DESTINATION "${MD_PYTHON_INSTALL_DIR}"
        COMPONENT python)
