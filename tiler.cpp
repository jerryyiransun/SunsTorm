#include "tiler.h"

#include "cost_model.h"

#include "absl/status/status.h"

#include <algorithm>
#include <functional>
#ifdef DEBUG
#include <iostream>
#endif
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace mlsys {

namespace {

// The current tiler search space mirrors the old greedy shrink path:
// start from the native granularity and repeatedly halve dimensions.
auto HalvingCandidates(int64_t start) -> std::vector<int64_t> {
    std::vector<int64_t> candidates;
    start = std::max<int64_t>(1, start);

    while (true) {
        candidates.push_back(start);
        if (start == 1) {
            break;
        }
        start = (start + 1) / 2;
    }

    return candidates;
}

auto DivisorCandidates(int64_t value, int64_t cap) -> std::vector<int64_t> {
    value = std::max<int64_t>(1, value);
    cap = std::max<int64_t>(1, cap);

    std::vector<int64_t> candidates;
    for (int64_t divisor = 1; divisor <= value / divisor; ++divisor) {
        if ((value % divisor) != 0) {
            continue;
        }
        int64_t const paired = value / divisor;
        if (divisor <= cap) {
            candidates.push_back(divisor);
        }
        if (paired != divisor && paired <= cap) {
            candidates.push_back(paired);
        }
    }

    std::sort(candidates.begin(), candidates.end(), std::greater<>());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    if (candidates.empty()) {
        candidates.push_back(1);
    }
    return candidates;
}

auto MaxMatMulDepth(const Problem& problem, const Subgraph& subgraph) -> int64_t {
    int64_t max_depth = 1;
    for (size_t op_idx : subgraph.ops) {
        if (problem.ops[op_idx].op_type != "MatMul") {
            continue;
        }
        size_t lhs_idx = problem.ops[op_idx].inputs[0];
        max_depth = std::max<int64_t>(max_depth, problem.tensors[lhs_idx].width);
    }
    return max_depth;
}

auto FinalMatMulDepth(const Problem& problem, const Subgraph& subgraph) -> std::optional<int64_t> {
    std::set<size_t> consumed;
    for (size_t op_idx : subgraph.ops) {
        for (size_t input_idx : problem.ops[op_idx].inputs) {
            consumed.insert(input_idx);
        }
    }

    std::optional<int64_t> max_depth;
    for (size_t op_idx : subgraph.ops) {
        const Op& op = problem.ops[op_idx];
        if (op.op_type != "MatMul" || consumed.contains(op.outputs[0])) {
            continue;
        }

        int64_t const depth = problem.tensors[op.inputs[0]].width;
        max_depth = max_depth.has_value() ? std::max(max_depth.value(), depth) : depth;
    }

    return max_depth;
}

auto MaxFinalOutputShape(const Problem& problem, const Subgraph& subgraph) -> Tensor {
    std::set<size_t> produced;
    std::set<size_t> consumed;
    for (size_t op_idx : subgraph.ops) {
        produced.insert(problem.ops[op_idx].outputs[0]);
        for (size_t input_idx : problem.ops[op_idx].inputs) {
            consumed.insert(input_idx);
        }
    }

    Tensor shape{.width = 1, .height = 1};
    bool found_final_output = false;
    for (size_t tensor_idx : produced) {
        if (consumed.contains(tensor_idx)) {
            continue;
        }
        found_final_output = true;
        shape.width = std::max<int64_t>(shape.width, problem.tensors[tensor_idx].width);
        shape.height = std::max<int64_t>(shape.height, problem.tensors[tensor_idx].height);
    }

    if (found_final_output) {
        return shape;
    }

    for (size_t tensor_idx : produced) {
        shape.width = std::max<int64_t>(shape.width, problem.tensors[tensor_idx].width);
        shape.height = std::max<int64_t>(shape.height, problem.tensors[tensor_idx].height);
    }
    return shape;
}

auto BuildProducerMap(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        producer[problem.ops[op_idx].outputs[0]] = static_cast<int>(op_idx);
    }
    return producer;
}

auto BuildSnakeTraversalOrder(const Problem& problem, const Subgraph& subgraph)
    -> std::optional<TraversalOrder> {
    std::set<size_t> subgraph_produced;
    std::set<size_t> subgraph_consumed;
    for (size_t op_idx : subgraph.ops) {
        size_t const out = problem.ops[op_idx].outputs[0];
        subgraph_produced.insert(out);
        for (size_t in_idx : problem.ops[op_idx].inputs) {
            subgraph_consumed.insert(in_idx);
        }
    }

    std::vector<size_t> final_outputs;
    final_outputs.reserve(subgraph_produced.size());
    for (size_t t_idx : subgraph_produced) {
        if (!subgraph_consumed.contains(t_idx)) {
            final_outputs.push_back(t_idx);
        }
    }

    if (final_outputs.empty()) {
        return std::nullopt;
    }

    auto tile_grid_for_output = [&](size_t tensor_idx) -> std::pair<int64_t, int64_t> {
        int64_t const out_w = problem.tensors[tensor_idx].width;
        int64_t const out_h = problem.tensors[tensor_idx].height;
        int64_t const tiles_w =
            (out_w + subgraph.granularity.width - 1) / subgraph.granularity.width;
        int64_t const tiles_h =
            (out_h + subgraph.granularity.height - 1) / subgraph.granularity.height;
        return {tiles_w, tiles_h};
    };

    auto [tiles_w, tiles_h] = tile_grid_for_output(final_outputs[0]);
    for (size_t out_idx : final_outputs) {
        auto [curr_tiles_w, curr_tiles_h] = tile_grid_for_output(out_idx);
        if (curr_tiles_w * curr_tiles_h != tiles_w * tiles_h) {
            return std::nullopt;
        }
    }

    TraversalOrder order;
    order.reserve(static_cast<size_t>(tiles_w * tiles_h));

    if (tiles_w >= tiles_h) {
        for (int64_t ty = 0; ty < tiles_h; ++ty) {
            if ((ty % 2) == 0) {
                for (int64_t tx = 0; tx < tiles_w; ++tx) {
                    order.push_back(ty * tiles_w + tx);
                }
            } else {
                for (int64_t tx = tiles_w - 1; tx >= 0; --tx) {
                    order.push_back(ty * tiles_w + tx);
                }
            }
        }
        return order;
    }

    for (int64_t tx = 0; tx < tiles_w; ++tx) {
        if ((tx % 2) == 0) {
            for (int64_t ty = 0; ty < tiles_h; ++ty) {
                order.push_back(ty * tiles_w + tx);
            }
        } else {
            for (int64_t ty = tiles_h - 1; ty >= 0; --ty) {
                order.push_back(ty * tiles_w + tx);
            }
        }
    }
    return order;
}

// Cheap ranking heuristic for exhaustive tiling search.
// We keep the exact CostModel at the full-plan level in the solver, but use this
// score to compare subgraph-local tile candidates without exploding runtime.
auto EstimateCandidateScore(const Problem& problem, const Subgraph& subgraph) -> double {
    std::set<size_t> produced;
    std::set<size_t> consumed;
    for (size_t op_idx : subgraph.ops) {
        produced.insert(problem.ops[op_idx].outputs[0]);
        for (size_t input_idx : problem.ops[op_idx].inputs) {
            consumed.insert(input_idx);
        }
    }

    int64_t max_out_w = 1;
    int64_t max_out_h = 1;
    for (size_t tensor_idx : produced) {
        if (consumed.contains(tensor_idx)) {
            continue;
        }
        max_out_w = std::max<int64_t>(max_out_w, problem.tensors[tensor_idx].width);
        max_out_h = std::max<int64_t>(max_out_h, problem.tensors[tensor_idx].height);
    }

    int64_t const tiles_w =
        (max_out_w + subgraph.granularity.width - 1) / subgraph.granularity.width;
    int64_t const tiles_h =
        (max_out_h + subgraph.granularity.height - 1) / subgraph.granularity.height;

    int64_t k_steps = 1;
    for (size_t op_idx : subgraph.ops) {
        if (problem.ops[op_idx].op_type != "MatMul") {
            continue;
        }
        size_t lhs_idx = problem.ops[op_idx].inputs[0];
        int64_t const full_k = problem.tensors[lhs_idx].width;
        k_steps = std::max<int64_t>(k_steps, (full_k + subgraph.granularity.depth - 1) /
                                                 subgraph.granularity.depth);
    }

    double const native_w = static_cast<double>(problem.native_granularity.width);
    double const native_h = static_cast<double>(problem.native_granularity.height);
    double const padding_ratio = (native_w / static_cast<double>(subgraph.granularity.width)) *
                                 (native_h / static_cast<double>(subgraph.granularity.height));

    double compute_score = 0.0;
    for (size_t op_idx : subgraph.ops) {
        compute_score += static_cast<double>(problem.ops[op_idx].base_cost);
    }

    return static_cast<double>(tiles_w) * static_cast<double>(tiles_h) *
           static_cast<double>(k_steps) * padding_ratio * compute_score;
}

// The cache key includes the suffix schedule because boundary-output behavior can
// change when later subgraphs recompute or consume tensors differently.
auto BuildCacheKey(const Solution& solution, size_t sg_idx,
                   const std::set<size_t>& prev_retained_tensors) -> std::string {
    std::string key;
    for (size_t idx = sg_idx; idx < solution.subgraphs.size(); ++idx) {
        const auto& subgraph = solution.subgraphs[idx];
        key += "ops:";
        for (size_t op_idx : subgraph.ops) {
            key += std::to_string(op_idx);
            key += ",";
        }

        key += "|retain:";
        for (size_t tensor_idx : subgraph.tensors_to_retain) {
            key += std::to_string(tensor_idx);
            key += ",";
        }
        key += ";";
    }

    key += "|prev:";
    for (size_t tensor_idx : prev_retained_tensors) {
        key += std::to_string(tensor_idx);
        key += ",";
    }

    return key;
}

// Fast-path feasibility check that mirrors Evaluate's working-set accounting.
// This lets the tiler prune impossible candidates before the solver pays the
// cost of full plan evaluation and exact latency estimation.
auto FitsFastMemory(const Problem& problem, const Solution& solution, size_t sg_idx,
                    const std::set<size_t>& prev_retained_tensors,
                    const std::vector<int>& producer_op) -> bool {
    return SubgraphFitsFastMemory(problem, solution, sg_idx, prev_retained_tensors, producer_op);
}

struct CostGuidedDivisorState {
    size_t width_idx = 0;
    size_t height_idx = 0;
    size_t depth_idx = 0;
};

#ifdef DEBUG
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

void DebugPrintGranularity(const Granularity& granularity) {
    std::cout << "{w=" << granularity.width << ", h=" << granularity.height
              << ", k=" << granularity.depth << "}";
}

void DebugPrintState(const CostGuidedDivisorState& state, const Granularity& granularity) {
    std::cout << "{idx_w=" << state.width_idx << ", idx_h=" << state.height_idx
              << ", idx_k=" << state.depth_idx << ", granularity=";
    DebugPrintGranularity(granularity);
    std::cout << "}";
}

void DebugPrintOptionalCost(const std::optional<double>& cost) {
    if (!cost.has_value()) {
        std::cout << "unavailable";
        return;
    }
    std::cout << cost.value();
}
#endif

auto GranularityFromState(const std::vector<int64_t>& width_candidates,
                          const std::vector<int64_t>& height_candidates,
                          const std::vector<int64_t>& depth_candidates,
                          const CostGuidedDivisorState& state) -> Granularity {
    return {.width = width_candidates[state.width_idx],
            .height = height_candidates[state.height_idx],
            .depth = depth_candidates[state.depth_idx]};
}

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

auto NextDepthState(const std::vector<int64_t>& depth_candidates,
                    const CostGuidedDivisorState& state) -> std::optional<CostGuidedDivisorState> {
    if (state.depth_idx + 1 >= depth_candidates.size()) {
        return std::nullopt;
    }

    CostGuidedDivisorState next = state;
    ++next.depth_idx;
    return next;
}

void ApplyGranularityAndTraversal(const Problem& problem, Solution& solution, size_t sg_idx,
                                  const Granularity& granularity) {
    Subgraph& subgraph = solution.subgraphs[sg_idx];
    subgraph.granularity = granularity;
    subgraph.traversal_order = BuildSnakeTraversalOrder(problem, subgraph);
}

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

auto TileSubgraphWithCostGuidedDivisors(const Problem& problem, Solution& solution, size_t sg_idx,
                                        const std::set<size_t>& prev_retained_tensors,
                                        const std::vector<int>& producer_op, CostModel& cost_model)
    -> Status {
    Subgraph& subgraph = solution.subgraphs[sg_idx];
    Tensor const output_shape = MaxFinalOutputShape(problem, subgraph);

    std::vector<int64_t> width_candidates =
        DivisorCandidates(output_shape.width, problem.native_granularity.width);
    std::vector<int64_t> height_candidates =
        DivisorCandidates(output_shape.height, problem.native_granularity.height);
    std::optional<int64_t> const final_matmul_depth = FinalMatMulDepth(problem, subgraph);
    std::vector<int64_t> depth_candidates =
        final_matmul_depth.has_value()
            ? DivisorCandidates(final_matmul_depth.value(), final_matmul_depth.value())
            : std::vector<int64_t>{1};

#ifdef DEBUG
    std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " begin\n";
    std::cout << "[DEBUG][CostGuidedDivisorTiler] subgraph " << sg_idx << " ops=";
    DebugPrintVector(subgraph.ops);
    std::cout << ", output_shape={w=" << output_shape.width << ", h=" << output_shape.height
              << "}, depth_affects_split_k=" << (final_matmul_depth.has_value() ? "true" : "false")
              << ", prev_retained=";
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
            FitsFastMemory(problem, solution, sg_idx, prev_retained_tensors, producer_op);
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
        spatial_latency = EstimateCandidateLatency(
            problem, solution, sg_idx, prev_retained_tensors, cost_model,
            GranularityFromState(width_candidates, height_candidates, depth_candidates,
                                 *spatial_state));

        std::optional<double> depth_latency;
        depth_latency = EstimateCandidateLatency(
            problem, solution, sg_idx, prev_retained_tensors, cost_model,
            GranularityFromState(width_candidates, height_candidates, depth_candidates,
                                 *depth_state));

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

auto BruteForceTiler::tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    std::vector<int> producer_op = BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;

    for (size_t sg_idx = 0; sg_idx < tiled_solution.subgraphs.size(); ++sg_idx) {
        auto& sg = tiled_solution.subgraphs[sg_idx];
        std::string const cache_key = BuildCacheKey(tiled_solution, sg_idx, prev_retained_tensors);
        auto cache_it = best_granularity_cache_.find(cache_key);
        if (cache_it != best_granularity_cache_.end()) {
            if (!cache_it->second.has_value()) {
                return absl::ResourceExhaustedError(
                    "Cannot fit working set even at minimum tile size");
            }

            sg.granularity = *cache_it->second;
            sg.traversal_order = BuildSnakeTraversalOrder(problem, sg);
            prev_retained_tensors =
                std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
            continue;
        }

        std::vector<int64_t> width_candidates = HalvingCandidates(problem.native_granularity.width);
        std::vector<int64_t> height_candidates =
            HalvingCandidates(problem.native_granularity.height);
        std::vector<int64_t> depth_candidates = HalvingCandidates(MaxMatMulDepth(problem, sg));
        bool has_matmul = false;
        for (size_t op_idx : sg.ops) {
            if (problem.ops[op_idx].op_type == "MatMul") {
                has_matmul = true;
                break;
            }
        }

        Granularity best_granularity{.width = 0, .height = 0, .depth = 0};
        double best_score = std::numeric_limits<double>::infinity();

        // Exhaustively enumerate the halving-based candidate space for this subgraph.
        for (int64_t width : width_candidates) {
            for (int64_t height : height_candidates) {
                for (int64_t depth : depth_candidates) {
                    if (depth != 1 && !has_matmul) {
                        continue;
                    }

                    sg.granularity = {.width = width, .height = height, .depth = depth};
                    if (!FitsFastMemory(problem, tiled_solution, sg_idx, prev_retained_tensors,
                                        producer_op)) {
                        continue;
                    }

                    double const score = EstimateCandidateScore(problem, sg);

                    if (score < best_score) {
                        best_score = score;
                        best_granularity = sg.granularity;
                    }
                }
            }
        }

        if (best_granularity.width == 0 || best_granularity.height == 0 ||
            best_granularity.depth == 0) {
            best_granularity_cache_[cache_key] = std::nullopt;
            return absl::ResourceExhaustedError("Cannot fit working set even at minimum tile size");
        }

        sg.granularity = best_granularity;
        sg.traversal_order = BuildSnakeTraversalOrder(problem, sg);
        best_granularity_cache_[cache_key] = best_granularity;
        prev_retained_tensors =
            std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
    }

    return tiled_solution;
}

auto GreedyTiler::tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    std::vector<int> producer_op = BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;

    for (size_t sg_idx = 0; sg_idx < tiled_solution.subgraphs.size(); ++sg_idx) {
        auto& sg = tiled_solution.subgraphs[sg_idx];
        sg.granularity.width = problem.native_granularity.width;
        sg.granularity.height = problem.native_granularity.height;
        sg.granularity.depth = MaxMatMulDepth(problem, sg);

        while (
            !FitsFastMemory(problem, tiled_solution, sg_idx, prev_retained_tensors, producer_op)) {
            int64_t const w = sg.granularity.width;
            int64_t const h = sg.granularity.height;
            int64_t const k = sg.granularity.depth;
            int64_t const max_dim = std::max({w, h, k});

            if (max_dim <= 1) {
                return absl::ResourceExhaustedError(
                    "Cannot fit working set even at minimum tile size");
            }

            // Match the previous strategy: always shrink the largest active dimension.
            if (k == max_dim) {
                sg.granularity.depth = (k + 1) / 2;
            } else if (w == max_dim) {
                sg.granularity.width = (w + 1) / 2;
            } else {
                sg.granularity.height = (h + 1) / 2;
            }
        }

        sg.traversal_order = BuildSnakeTraversalOrder(problem, sg);
        prev_retained_tensors =
            std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
    }

    return tiled_solution;
}

auto CostGuidedDivisorTiler::tile(const Problem& problem, const Solution& solution)
    -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    std::vector<int> producer_op = BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;
    CostModel cost_model(problem);

    for (size_t sg_idx = 0; sg_idx < tiled_solution.subgraphs.size(); ++sg_idx) {
        auto status = TileSubgraphWithCostGuidedDivisors(
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

auto CostGuidedDivisorTiler::tile_subgraph(const Problem& problem, const Solution& solution,
                                           size_t sg_idx) -> StatusOr<Solution> {
    if (sg_idx >= solution.subgraphs.size()) {
        return absl::InvalidArgumentError("CostGuidedDivisorTiler: subgraph index out of range");
    }

    Solution tiled_solution = solution;
    std::vector<int> producer_op = BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;
    if (sg_idx > 0) {
        const auto& prev_sg = tiled_solution.subgraphs[sg_idx - 1];
        prev_retained_tensors =
            std::set<size_t>(prev_sg.tensors_to_retain.begin(), prev_sg.tensors_to_retain.end());
    }

    CostModel cost_model(problem);
    auto status = TileSubgraphWithCostGuidedDivisors(
        problem, tiled_solution, sg_idx, prev_retained_tensors, producer_op, cost_model);
    if (!status.ok()) {
        return status;
    }

    return tiled_solution;
}

} // namespace mlsys
