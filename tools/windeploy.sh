#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jens Köhler
# Assisted-by: Claude Code (Anthropic)
#
# Bundle md_app.exe with its full Windows runtime so it runs on a machine without
# MSYS2/Qt installed. Two steps, because windeployqt on MSYS2 only handles Qt's
# own DLLs and plugins:
#   1. windeployqt6  -> Qt DLLs + plugins (platforms/qwindows.dll, tls, ...).
#   2. transitive-closure copy of every remaining dependency that resolves inside
#      the CLANG64 prefix (Qt's deps + the C/C++ runtime: libc++, ICU, HarfBuzz,
#      FreeType, glib, PCRE2, zlib, ...), which windeployqt does not copy here.
# The Vulkan loader (vulkan-1.dll) is intentionally NOT bundled: it ships with the
# OS/GPU driver and must match it. MSYS2 CLANG64 only.
#
# Usage: tools/windeploy.sh <path-to-md_app.exe>
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: windeploy.sh <path-to-exe>" >&2; exit 2; }

exe_dir="$(cd "$(dirname "$1")" && pwd)"
exe="$exe_dir/$(basename "$1")"
[ -f "$exe" ] || { echo "windeploy: no such file: $exe" >&2; exit 1; }

srcbin="/clang64/bin"
windeployqt="$(command -v windeployqt6 || command -v windeployqt)"

"$windeployqt" --release --no-translations --no-opengl-sw --compiler-runtime "$exe"

# Fixpoint: keep copying newly-discovered CLANG64 dependencies of the exe and of
# every DLL already in the bundle (plugins included) until nothing new appears.
changed=1
while [ "$changed" = 1 ]; do
  changed=0
  while IFS= read -r target; do
    while read -r dep; do
      case "$dep" in
        "$srcbin"/*)
          base="$(basename "$dep")"
          if [ ! -f "$exe_dir/$base" ]; then
            cp "$dep" "$exe_dir/"
            changed=1
          fi
          ;;
      esac
    done < <(ldd "$target" 2>/dev/null | awk '{print $3}')
  done < <(printf '%s\n' "$exe"; find "$exe_dir" -name '*.dll')
done

echo "windeploy: bundled $(find "$exe_dir" -name '*.dll' | wc -l) DLLs alongside $(basename "$exe")"
