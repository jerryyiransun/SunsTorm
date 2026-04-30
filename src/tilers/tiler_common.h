/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "mlsys.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlsys::tiler_internal {

struct CostGuidedDivisorState {
    size_t width_idx = 0;
    size_t height_idx = 0;
    size_t depth_idx = 0;
};

auto HalvingCandidates(int64_t start) -> std::vector<int64_t>;
auto CandidateTileSizes(int64_t dimension, int64_t cap) -> std::vector<int64_t>;
auto SplitKMatMulDepth(const Problem& problem, const Subgraph& subgraph) -> std::optional<int64_t>;
auto MaxFinalOutputShape(const Problem& problem, const Subgraph& subgraph) -> Tensor;
auto BuildProducerMap(const Problem& problem) -> std::vector<int>;
auto BuildSnakeTraversalOrder(const Problem& problem, const Subgraph& subgraph)
    -> std::optional<TraversalOrder>;
auto EstimateCandidateScore(const Problem& problem, const Subgraph& subgraph) -> double;
auto BuildCacheKey(const Solution& solution, size_t sg_idx,
                   const std::set<size_t>& prev_retained_tensors) -> std::string;
auto FitsFastMemory(const Problem& problem, const Solution& solution, size_t sg_idx,
                    const std::set<size_t>& prev_retained_tensors,
                    const std::vector<int>& producer_op) -> bool;
auto GranularityFromState(const std::vector<int64_t>& width_candidates,
                          const std::vector<int64_t>& height_candidates,
                          const std::vector<int64_t>& depth_candidates,
                          const CostGuidedDivisorState& state) -> Granularity;
void ApplyGranularityAndTraversal(const Problem& problem, Solution& solution, size_t sg_idx,
                                  const Granularity& granularity);

} // namespace mlsys::tiler_internal
