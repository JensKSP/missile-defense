# Testing & Code Quality

This project is developed **test-first**. Everything below runs locally via `poe`
(after `. .venv/bin/activate`), and the same tasks form the quality gate.

## Test layers

| Layer | Location | Framework | CTest label | Run with |
|-------|----------|-----------|-------------|----------|
| C++ unit | `core/tests/unit/` | Catch2 v3 | `unit` | `poe test-unit` |
| C++ e2e / integration | `core/tests/e2e/` | Catch2 v3 | `e2e` | `poe test-e2e` |
| Python | `python/tests/` | pytest | — | `poe pytest` |

- **Unit** tests are fast and isolated — the TDD inner loop is just `poe test-unit`
  (it auto-builds first).
- **e2e** tests drive a whole simulation episode (`reset` → `step`… → terminal) and
  assert end-to-end invariants (determinism, scoring, termination).
- Catch2 test names must **not contain `[` `]`** — those are reserved for tags.

## The TDD loop

```bash
. .venv/bin/activate
poe test-unit      # write a failing test, watch it fail, implement, watch it pass
```

## Quality gates (zero-warning policy)

Production code (the sim / game / AI) must compile and lint **perfectly clean**.
Tests and glue code share the same warnings but are not held to clang-tidy.

| Gate | Task | Enforcement |
|------|------|-------------|
| C++ format | `poe format-check` | clang-format-21, `--Werror` |
| C++ warnings | (build) | `-Wall -Wextra -Wpedantic …` **`-Werror`** on `md::warnings` |
| C++ static analysis | `poe tidy` | clang-tidy-21, `--warnings-as-errors=*` (production sources only) |
| Python format | `poe fmt-py-chk` | ruff format `--check` |
| Python lint | `poe lint` | ruff check |
| Python types | `poe typecheck` | mypy `--strict` |

Run **everything** with:

```bash
poe check     # format + lint + types + tidy + all tests — the full local CI gate
```

## Optional: git pre-commit hook

```bash
poe install-hooks   # runs format + lint + build + unit tests before each commit
```

## Toggles

- `-DMD_WERROR=OFF` — disable warnings-as-errors (e.g. trying a new compiler).
- `-DMD_SANITIZE=OFF` — disable ASan/UBSan (on by default in the `debug` preset).
