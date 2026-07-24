# app/ — the human client (Qt 6 + Vulkan). Depends on the pure md::core sim.
find_package(Qt6 QUIET COMPONENTS Gui)
find_package(Vulkan QUIET)

if(NOT Qt6_FOUND OR NOT Vulkan_FOUND)
  message(STATUS "md_app: Qt6 or Vulkan not found — skipping the app target "
                 "(build core-only with -DMD_BUILD_APP=OFF to silence this)")
  return()
endif()

add_executable(md_app
  main.cpp
  game_window.cpp
  renderer.cpp)

# Qt/Vulkan headers are imported targets → their includes are SYSTEM, so our
# strict -Werror only holds our own code to account.
target_link_libraries(md_app
  PRIVATE md::core md::warnings Qt6::Gui Vulkan::Vulkan)

set_target_properties(md_app PROPERTIES AUTOMOC ON)
