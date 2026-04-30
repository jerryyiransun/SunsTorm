/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "tiler.h"

#include "cost_model.h"
#include "tiler_common.h"

#include "absl/status/status.h"

#include <algorithm>
#ifdef DEBUG
#include <iostream>
#endif
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <vector>

namespace mlsys {
namespace {

struct CostGuidedDivisorState {
    size_t width_idx = 0;
    size_t height_idx = 0;
    size_t depth_idx = 0;
};

#ifdef DEBUG
// Prints a compact vector representation for debug traces.
template <typename T> void DebugPrintVector(const std::vector<T>& values) {
    std::cout << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx > 0) {
            std::cout << ", ";
        }
        std::cout << values[idx];
    }
    std::cout << "]";
}

// Prints a granularity tuple for debug traces.
void DebugPrintGranularity(const Granularity& granularity) {
    std::cout << "{w=" << granularity.width << ", h=" << granularity.height
              << ", k=" << granularity.depth << "}";
}

// Prints candidate indices plus the corresponding granularity for debug traces.
void DebugPrintState(const CostGuidedDivisorState& state, const Granularity& granularity) {
    std::cout << "{idx_w=" << state.width_idx << ", idx_h=" << state.height_idx
              << ", idx_k=" << state.depth_idx << ", granularity=";
    DebugPrintGranularity(granularity);
    std::cout << "}";
}

// Prints an optional candidate latency for debug traces.
void DebugPrintOptionalCost(const std::optional<double>& cost) {
    if (!cost.has_value()) {
        std::cout << "unavailable";
        return;
    }
    std::cout << cost.value();
}
#endif

// Converts candidate-list indices into the concrete granularity they select.
auto GranularityFromState(const std::vector<int64_t>& width_candidates,
                          const std::vector<int64_t>& height_candidates,
                          const std::vector<int64_t>& depth_candidates,
                          const CostGuidedDivisorState& state) -> Granularity {
    return {.width = width_candidates[state.width_idx],
            .height = height_candidates[state.height_idx],
            .depth = depth_candidates[state.depth_idx]};
}

// Advances the next spatial candidate, preferring the currently larger spatial side.
auto NextSpatialState(const std::vector<int64_t>& width_candidates,
                      const std::vector<int64_t>& height_candidates,
                      const std::vector<int64_t>& depth_candidates,
                      const CostGuidedDivisorState& state)
    -> std::optional<CostGuidedDivisorState> {
    Granularity const current =
        GranularityFromState(width_candidates, height_candidates, depth_candidates, state);
    CostGuidedDivisorState next = state;

    auto shrink_width = [&]() -> bool {
        if (next.width_idx + 1 >= width_candidates.size()) {
            return false;
        }
        ++next.width_idx;
        return true;
    };
    auto shrink_height = [&]() -> bool {
        if (next.height_idx + 1 >= height_candidates.size()) {
            return false;
        }
        ++next.height_idx;
        return true;
    };

    if (current.height > current.width) {
        if (shrink_height() || shrink_width()) {
            return next;
        }
    } else if (shrink_width() || shrink_height()) {
        return next;
    }

    return std::nullopt;
}

// Advances the split-k candidate by one step when another depth candidate exists.
auto NextDepthState(const std::vector<int64_t>& depth_candidates,
                    const CostGuidedDivisorState& state) -> std::optional<CostGuidedDivisorState> {
    if (state.depth_idx + 1 >= depth_candidates.size()) {
        return std::nullopt;
    }

    CostGuidedDivisorState next = state;
    ++next.depth_idx;
    return next;
}

// Applies a granularity and refreshes the matching traversal order.
void ApplyGranularityAndTraversal(const Problem& problem, Solution& solution, size_t sg_idx,
                                  const Granularity& granularity) {
    Subgraph& subgraph = solution.subgraphs[sg_idx];
    subgraph.granularity = granularity;
    subgraph.traversal_order = tiler_internal::BuildSnakeTraversalOrder(problem, subgraph);
}

// Estimates a candidate subgraph latency, returning nullopt if the cost model rejects it.
auto EstimateCandidateLatency(const Problem& problem, const Solution& solution, size_t sg_idx,
                              const std::set<size_t>& prev_retained_tensors, CostModel& cost_model,
                              const Granularity& granularity) -> std::optional<double> {
    Solution candidate_solution = solution;
    ApplyGranularityAndTraversal(problem, candidate_solution, sg_idx, granularity);

    auto latency =
        cost_model.estimate_subgraph_latency(candidate_solution, sg_idx, prev_retained_tensors);
    if (!latency.ok()) {
#ifdef DEBUG
        std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx
                  << " candidate cost unavailable for ";
        DebugPrintGranularity(granularity);
        std::cout << ": " << latency.status().message() << "\n";
#endif
        return std::nullopt;
    }
    return latency.value();
}

// Tiles one subgraph by comparing greedy-style spatial and split-k candidates.
auto TileSubgraphWithCostGuidedCandidates(const Problem& problem, Solution& solution, size_t sg_idx,
                                          const std::set<size_t>& prev_retained_tensors,
                                          const std::vector<int>& producer_op,
                                          CostModel& cost_model) -> Status {
    Subgraph& subgraph = solution.subgraphs[sg_idx];
    Tensor const output_shape = tiler_internal::MaxFinalOutputShape(problem, subgraph);

    std::vector<int64_t> width_candidates =
        tiler_internal::CandidateTileSizes(output_shape.width, problem.native_granularity.width);
    std::vector<int64_t> height_candidates =
        tiler_internal::CandidateTileSizes(output_shape.height, problem.native_granularity.height);
    std::optional<int64_t> const split_k_matmul_depth =
        tiler_internal::SplitKMatMulDepth(problem, subgraph);
    std::vector<int64_t> depth_candidates =
        split_k_matmul_depth.has_value()
            ? tiler_internal::CandidateTileSizes(split_k_matmul_depth.value(),
                                                 problem.native_granularity.depth)
            : std::vector<int64_t>{1};

#ifdef DEBUG
    std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " begin\n";
    std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " ops=";
    DebugPrintVector(subgraph.ops);
    std::cout << ", output_shape={w=" << output_shape.width << ", h=" << output_shape.height
              << "}, depth_affects_split_k="
              << (split_k_matmul_depth.has_value() ? "true" : "false") << ", prev_retained=";
    DebugPrintVector(
        std::vector<size_t>(prev_retained_tensors.begin(), prev_retained_tensors.end()));
    std::cout << "\n";
    std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " width_candidates=";
    DebugPrintVector(width_candidates);
    std::cout << ", height_candidates=";
    DebugPrintVector(height_candidates);
    std::cout << ", depth_candidates=";
    DebugPrintVector(depth_candidates);
    std::cout << "\n";
#endif

    CostGuidedDivisorState state;
#ifdef DEBUG
    int64_t step = 0;
#endif

    while (true) {
        Granularity const current =
            GranularityFromState(width_candidates, height_candidates, depth_candidates, state);
        ApplyGranularityAndTraversal(problem, solution, sg_idx, current);

#ifdef DEBUG
        std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                  << " current=";
        DebugPrintState(state, current);
        std::cout << "\n";
#endif

        bool const fits_fast_memory =
#ifdef DEBUG
            DebugSubgraphFitsFastMemory(problem, solution, sg_idx, prev_retained_tensors,
                                        producer_op);
#else
            tiler_internal::FitsFastMemory(problem, solution, sg_idx, prev_retained_tensors,
                                           producer_op);
#endif
#ifdef DEBUG
        std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                  << " fits_fast_memory=" << (fits_fast_memory ? "true" : "false") << "\n";
#endif

        if (fits_fast_memory) {
            auto current_latency =
                cost_model.estimate_subgraph_latency(solution, sg_idx, prev_retained_tensors);
            if (current_latency.ok()) {
#ifdef DEBUG
                std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step "
                          << step << " accepted ";
                DebugPrintGranularity(current);
                std::cout << " with exact_cost=" << current_latency.value() << "\n";
#endif
                return absl::OkStatus();
            }
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " current cost unavailable despite fitting: "
                      << current_latency.status().message() << "\n";
#endif
        }

        std::optional<CostGuidedDivisorState> spatial_state =
            NextSpatialState(width_candidates, height_candidates, depth_candidates, state);
        std::optional<CostGuidedDivisorState> depth_state = NextDepthState(depth_candidates, state);

        if (!spatial_state.has_value() && !depth_state.has_value()) {
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " terminating: no shrink candidate remains\n";
#endif
            return absl::ResourceExhaustedError("Cannot fit working set even at minimum tile size");
        }

        if (spatial_state.has_value() && !depth_state.has_value()) {
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " choose spatial shrink because it is the only remaining move\n";
#endif
            state = *spatial_state;
#ifdef DEBUG
            ++step;
#endif
            continue;
        }

        if (!spatial_state.has_value() && depth_state.has_value()) {
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " choose depth shrink because it is the only remaining move\n";
#endif
            state = *depth_state;
#ifdef DEBUG
            ++step;
#endif
            continue;
        }

        std::optional<double> spatial_latency;
        spatial_latency =
            EstimateCandidateLatency(problem, solution, sg_idx, prev_retained_tensors, cost_model,
                                     GranularityFromState(width_candidates, height_candidates,
                                                          depth_candidates, *spatial_state));

        std::optional<double> depth_latency;
        depth_latency =
            EstimateCandidateLatency(problem, solution, sg_idx, prev_retained_tensors, cost_model,
                                     GranularityFromState(width_candidates, height_candidates,
                                                          depth_candidates, *depth_state));

#ifdef DEBUG
        std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                  << " next spatial=";
        if (spatial_state.has_value()) {
            DebugPrintState(*spatial_state,
                            GranularityFromState(width_candidates, height_candidates,
                                                 depth_candidates, *spatial_state));
        } else {
            std::cout << "none";
        }
        std::cout << ", spatial_cost=";
        DebugPrintOptionalCost(spatial_latency);
        std::cout << "; next depth=";
        if (depth_state.has_value()) {
            DebugPrintState(*depth_state, GranularityFromState(width_candidates, height_candidates,
                                                               depth_candidates, *depth_state));
        } else {
            std::cout << "none";
        }
        std::cout << ", depth_cost=";
        DebugPrintOptionalCost(depth_latency);
        std::cout << "\n";
#endif

        if (!spatial_latency.has_value() && !depth_latency.has_value()) {
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " terminating: no shrink candidate has an exact cost\n";
#endif
            return absl::ResourceExhaustedError("Cannot fit working set even at minimum tile size");
        }

        if (depth_latency.has_value() &&
            (!spatial_latency.has_value() || depth_latency.value() < spatial_latency.value())) {
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " choose depth shrink because depth_cost=" << depth_latency.value();
            if (spatial_latency.has_value()) {
                std::cout << " < spatial_cost=" << spatial_latency.value();
            } else {
                std::cout << " and spatial_cost is unavailable";
            }
            std::cout << "\n";
#endif
            state = *depth_state;
        } else {
#ifdef DEBUG
            std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " step " << step
                      << " choose spatial shrink";
            if (spatial_latency.has_value() && depth_latency.has_value()) {
                std::cout << " because spatial_cost=" << spatial_latency.value()
                          << " <= depth_cost=" << depth_latency.value();
            } else if (spatial_latency.has_value()) {
                std::cout << " because depth_cost is unavailable and spatial_cost="
                          << spatial_latency.value();
            }
            std::cout << "\n";
#endif
            state = *spatial_state;
        }
#ifdef DEBUG
        ++step;
#endif
    }
}

} // namespace

auto CostGuidedDivisorTiler::tile(const Problem& problem, const Solution& solution)
    -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    std::vector<int> producer_op = tiler_internal::BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;
    CostModel cost_model(problem);

    for (size_t sg_idx = 0; sg_idx < tiled_solution.subgraphs.size(); ++sg_idx) {
        auto status = TileSubgraphWithCostGuidedCandidates(
            problem, tiled_solution, sg_idx, prev_retained_tensors, producer_op, cost_model);
        if (!status.ok()) {
            return status;
        }

        const auto& sg = tiled_solution.subgraphs[sg_idx];
        prev_retained_tensors =
            std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
    }

    return tiled_solution;
}

// Retiles a single subgraph with the cost-guided divisor search.
auto CostGuidedDivisorTiler::tile_subgraph(const Problem& problem, const Solution& solution,
                                           size_t sg_idx) -> StatusOr<Solution> {
    if (sg_idx >= solution.subgraphs.size()) {
        return absl::InvalidArgumentError("CostGuidedDivisorTiler: subgraph index out of range");
    }

    Solution tiled_solution = solution;
    std::vector<int> producer_op = tiler_internal::BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;
    if (sg_idx > 0) {
        const auto& prev_sg = tiled_solution.subgraphs[sg_idx - 1];
        prev_retained_tensors =
            std::set<size_t>(prev_sg.tensors_to_retain.begin(), prev_sg.tensors_to_retain.end());
    }

    CostModel cost_model(problem);
    auto status = TileSubgraphWithCostGuidedCandidates(
        problem, tiled_solution, sg_idx, prev_retained_tensors, producer_op, cost_model);
    if (!status.ok()) {
        return status;
    }

    return tiled_solution;
}

} // namespace mlsys
