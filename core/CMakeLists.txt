# md_core — the pure, headless game simulation library.
# No Qt, no rendering, no Python. This is the single source of truth for game rules.
add_library(md_core)
add_library(md::core ALIAS md_core)

target_sources(md_core
  PRIVATE
    src/version.cpp)

target_include_directories(md_core PUBLIC include)
target_compile_features(md_core PUBLIC cxx_std_23)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
