#include "md/version.hpp"

#include <cstdio>
#include <string_view>

// Minimal, dependency-free smoke test. Its only job is to prove the build,
// link, and CTest wiring are correct before we introduce a real test framework.
int main() {
    const std::string_view v = md::version();
    if (v != "0.1.0") {
        std::fprintf(stderr, "unexpected version: %.*s\n", static_cast<int>(v.size()), v.data());
        return 1;
    }
    std::printf("md::core version %.*s — build OK\n", static_cast<int>(v.size()), v.data());
    return 0;
}
