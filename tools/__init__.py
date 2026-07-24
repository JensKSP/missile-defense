# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
"""Cross-platform developer tooling for missile-defense (invoked via ``poe``).

Everything here is plain Python (no shell), so the build/test/quality workflow
works the same on Linux, macOS, and Windows. Screen/video capture is inherently
platform-specific and lives in :mod:`tools.capture` with per-OS backends.
"""
