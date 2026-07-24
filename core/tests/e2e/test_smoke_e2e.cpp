#include "md/version.hpp"

#include <catch2/catch_test_macros.hpp>

// End-to-end tests exercise the whole simulation through a real episode:
// reset(seed) -> repeated step(action) -> terminal state, asserting invariants
// (determinism, score/ammo/city bookkeeping, termination).
//
// The Sim class lands in the next increment. For now this placeholder proves the
// e2e harness builds, links md::core, and runs under CTest so the very first
// real episode test can be dropped in with zero setup.
TEST_CASE("e2e harness is wired up", "[e2e][smoke]") {
    REQUIRE(md::version() == "0.1.0");
}
