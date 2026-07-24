#!/usr/bin/env bash
# LLVM source-based coverage for the pure simulation (md::core).
#
# Builds the coverage preset (core only, instrumented), runs the test binaries,
# merges their profiles, and reports line coverage of core/ (production sources
# only — tests and third-party code are excluded). Exits non-zero if total line
# coverage is below the threshold, so it can gate CI.
#
# Usage: scripts/coverage.sh [threshold-percent]   (default 80)
set -euo pipefail

THRESHOLD="${1:-80}"
BUILD="build/coverage"
PROF="$BUILD/prof"

PROFDATA_BIN="$(command -v llvm-profdata-21 || command -v llvm-profdata)"
COV_BIN="$(command -v llvm-cov-21 || command -v llvm-cov)"

cmake --preset coverage >/dev/null
cmake --build --preset coverage >/dev/null

UNIT="$BUILD/core/tests/md_core_unit_tests"
E2E="$BUILD/core/tests/md_core_e2e_tests"

rm -rf "$PROF"; mkdir -p "$PROF"
LLVM_PROFILE_FILE="$PROF/unit.profraw" "$UNIT" >/dev/null
LLVM_PROFILE_FILE="$PROF/e2e.profraw" "$E2E" >/dev/null

"$PROFDATA_BIN" merge -sparse "$PROF"/*.profraw -o "$PROF/merged.profdata"

# Only md::core and the tests are instrumented, so ignoring tests + fetched deps
# leaves exactly the production sources (core/src + core/include).
OBJECTS=(-object "$UNIT" -object "$E2E")
IGNORE='-ignore-filename-regex=(tests/|_deps/|build/|catch2)'

echo
"$COV_BIN" report "${OBJECTS[@]}" -instr-profile="$PROF/merged.profdata" "$IGNORE"
echo

# Parse total line coverage from the machine-readable export.
PCT="$("$COV_BIN" export "${OBJECTS[@]}" -instr-profile="$PROF/merged.profdata" \
  "$IGNORE" -summary-only \
  | python3 -c 'import json,sys; print("%.2f" % json.load(sys.stdin)["data"][0]["totals"]["lines"]["percent"])')"

printf 'Total line coverage: %s%% (threshold %s%%)\n' "$PCT" "$THRESHOLD"
if awk -v p="$PCT" -v t="$THRESHOLD" 'BEGIN { exit (p+0 >= t+0) ? 0 : 1 }'; then
  echo "OK: coverage gate passed."
else
  echo "FAIL: line coverage ${PCT}% is below the ${THRESHOLD}% threshold." >&2
  exit 1
fi
