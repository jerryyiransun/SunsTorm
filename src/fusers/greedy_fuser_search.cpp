/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "greedy_fuser_search.h"

// Greedy search internals live in fuser_common.cpp to keep the tightly coupled beam
// search state in one translation unit while exposing a small strategy entry point.
