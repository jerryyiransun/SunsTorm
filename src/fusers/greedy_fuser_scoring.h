/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "fuser.h"

#include <cstddef>
#include <cstdint>

namespace mlsys {

auto ScoreGreedyPotentialForTesting(GreedyPotentialScorer potential_scorer, int64_t saved_elements,
                                    size_t shared_tensor_count, bool is_retain, size_t retain_span,
                                    int64_t retained_elements, int64_t fast_memory_capacity,
                                    int64_t direct_boundary_elements) -> int64_t;

} // namespace mlsys
