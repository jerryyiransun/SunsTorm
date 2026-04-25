#include "tiler_common.h"

#include <algorithm>
#include <functional>
#include <set>
#include <string>

namespace mlsys::tiler_internal {

// BruteForceTiler's legacy search space mirrors the old greedy shrink path:
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

// Builds descending ceil(D / n) tile sizes, capped to the hardware/native limit.
auto CandidateTileSizes(int64_t dimension, int64_t cap) -> std::vector<int64_t> {
    dimension = std::max<int64_t>(1, dimension);
    cap = std::max<int64_t>(1, cap);

    std::vector<int64_t> candidates;
    for (int64_t tile_count = 1; tile_count <= dimension; ++tile_count) {
        int64_t const tile_size = std::min<int64_t>(cap, (dimension + tile_count - 1) / tile_count);
        if (candidates.empty() || candidates.back() != tile_size) {
            candidates.push_back(tile_size);
        }
    }

    if (candidates.empty()) {
        candidates.push_back(1);
    }
    return candidates;
}

// Finds the split-k depth that can affect final outputs of a subgraph.
//
// A final MatMul output can split directly. A final Pointwise output can also
// split when it is an epilogue over an upstream MatMul: the MatMul k-loop
// completes inside the subgraph, then the Pointwise fires on the completed
// spatial tile.
auto SplitKMatMulDepth(const Problem& problem, const Subgraph& subgraph) -> std::optional<int64_t> {
    std::set<size_t> op_set(subgraph.ops.begin(), subgraph.ops.end());
    std::set<size_t> produced;
    std::set<size_t> consumed;
    std::vector<int> producer(problem.tensors.size(), -1);

    for (size_t op_idx : subgraph.ops) {
        const Op& op = problem.ops[op_idx];
        produced.insert(op.outputs[0]);
        producer[op.outputs[0]] = static_cast<int>(op_idx);
        for (size_t input_idx : op.inputs) {
            consumed.insert(input_idx);
        }
    }

    std::optional<int64_t> max_depth;
    std::set<size_t> visited_tensors;
    std::function<void(size_t)> collect_depth_from_tensor = [&](size_t tensor_idx) {
        if (!visited_tensors.insert(tensor_idx).second) {
            return;
        }
        if (tensor_idx >= producer.size()) {
            return;
        }
        int const producer_idx = producer[tensor_idx];
        if (producer_idx < 0 || !op_set.contains(static_cast<size_t>(producer_idx))) {
            return;
        }

        const Op& op = problem.ops[static_cast<size_t>(producer_idx)];
        if (op.op_type == "MatMul") {
            int64_t const depth = problem.tensors[op.inputs[0]].width;
            max_depth = max_depth.has_value() ? std::max(max_depth.value(), depth) : depth;
            return;
        }

        if (op.op_type != "Pointwise") {
            return;
        }

        for (size_t input_idx : op.inputs) {
            if (produced.contains(input_idx)) {
                collect_depth_from_tensor(input_idx);
            }
        }
    };

    for (size_t tensor_idx : produced) {
        if (!consumed.contains(tensor_idx)) {
            collect_depth_from_tensor(tensor_idx);
        }
    }

    return max_depth;
}

// Returns the largest final-output shape that defines the subgraph tile grid.
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

// Maps each produced tensor to the op that creates it.
auto BuildProducerMap(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        producer[problem.ops[op_idx].outputs[0]] = static_cast<int>(op_idx);
    }
    return producer;
}

// Builds a snake traversal order over the final-output tile grid.
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

auto GranularityFromState(const std::vector<int64_t>& width_candidates,
                          const std::vector<int64_t>& height_candidates,
                          const std::vector<int64_t>& depth_candidates,
                          const CostGuidedDivisorState& state) -> Granularity {
    return {.width = width_candidates[state.width_idx],
            .height = height_candidates[state.height_idx],
            .depth = depth_candidates[state.depth_idx]};
}

void ApplyGranularityAndTraversal(const Problem& problem, Solution& solution, size_t sg_idx,
                                  const Granularity& granularity) {
    Subgraph& subgraph = solution.subgraphs[sg_idx];
    subgraph.granularity = granularity;
    subgraph.traversal_order = BuildSnakeTraversalOrder(problem, subgraph);
}

} // namespace mlsys::tiler_internal
