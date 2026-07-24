# Smoke test: proves the toolchain + library + CTest pipeline works end-to-end.
# Real unit tests (Catch2 via FetchContent) arrive with the actual sim in Step 1.
add_executable(md_core_smoke smoke.cpp)
target_link_libraries(md_core_smoke PRIVATE md::core)

add_test(NAME core_smoke COMMAND md_core_smoke)
