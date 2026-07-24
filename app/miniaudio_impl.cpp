// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
// The single translation unit that compiles the vendored miniaudio library.
// Built with warnings disabled (it is large third-party C); our own audio code
// lives in audio.cpp and is held to the normal strict flags.
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
