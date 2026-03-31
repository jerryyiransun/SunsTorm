#include "tiler.h"

#include <algorithm>
#include <limits>
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

auto BuildProducerMap(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        producer[problem.ops[op_idx].outputs[0]] = static_cast<int>(op_idx);
    }
    return producer;
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
    const auto& subgraph = solution.subgraphs[sg_idx];
    std::set<size_t> subgraph_ops(subgraph.ops.begin(), subgraph.ops.end());

    FastMemoryCapacity fast_memory_usage = 0;
    std::set<size_t> visited_tensors;

    for (size_t const t : prev_retained_tensors) {
        fast_memory_usage += problem.tensors[t].width * problem.tensors[t].height;
        visited_tensors.insert(t);
    }

    std::vector<std::tuple<size_t, Tensor, bool>> q;
    std::set<size_t> subgraph_produced;
    std::set<size_t> subgraph_consumed;
    std::set<size_t> to_be_retained_tensors(subgraph.tensors_to_retain.begin(),
                                            subgraph.tensors_to_retain.end());
    std::set<size_t> final_output_tensors;

    for (size_t const op_idx : subgraph.ops) {
        size_t const out = problem.ops[op_idx].outputs[0];
        subgraph_produced.insert(out);

        for (size_t const in : problem.ops[op_idx].inputs) {
            subgraph_consumed.insert(in);
        }
    }

    for (size_t const t_idx : to_be_retained_tensors) {
        if (visited_tensors.contains(t_idx)) {
            continue;
        }

        fast_memory_usage += problem.tensors[t_idx].width * problem.tensors[t_idx].height;
        visited_tensors.insert(t_idx);
        if (fast_memory_usage > problem.fast_memory_capacity) {
            return false;
        }
    }

    for (size_t const t_idx : subgraph_produced) {
        bool const is_final_output = !subgraph_consumed.contains(t_idx);
        if (!is_final_output) {
            continue;
        }

        final_output_tensors.insert(t_idx);
        if (!visited_tensors.contains(t_idx)) {
            fast_memory_usage += subgraph.granularity.width * subgraph.granularity.height;
            visited_tensors.insert(t_idx);
            if (fast_memory_usage > problem.fast_memory_capacity) {
                return false;
            }
        }

        Tensor tensor{.width = subgraph.granularity.width, .height = subgraph.granularity.height};
        q.emplace_back(t_idx, tensor, true);
    }

    size_t head = 0;
    auto is_ignored_tensor_in_backward = [&](size_t tensor_idx) {
        return final_output_tensors.contains(tensor_idx) ||
               to_be_retained_tensors.contains(tensor_idx) ||
               prev_retained_tensors.contains(tensor_idx);
    };

    while (head < q.size()) {
        auto [curr_tensor_idx, curr_tensor_dim, is_final] = q[head++];

        int const op_idx = producer_op[curr_tensor_idx];
        if (op_idx == -1 || !subgraph_ops.contains(static_cast<size_t>(op_idx))) {
            continue;
        }

        const Op& op = problem.ops[static_cast<size_t>(op_idx)];

        if (op.op_type == "MatMul") {
            size_t const lhs_tensor_idx = op.inputs[0];
            size_t const rhs_tensor_idx = op.inputs[1];

            int64_t const inner_k =
                is_final ? subgraph.granularity.depth : problem.tensors[lhs_tensor_idx].width;

            Tensor lhs_tensor{
                .width = inner_k,
                .height = curr_tensor_dim.height,
            };
            if (!is_ignored_tensor_in_backward(lhs_tensor_idx) &&
                !visited_tensors.contains(lhs_tensor_idx)) {
                visited_tensors.insert(lhs_tensor_idx);

                bool const lhs_is_ephemeral = subgraph_produced.contains(lhs_tensor_idx);
                bool const lhs_is_retained = prev_retained_tensors.contains(lhs_tensor_idx);
                if (!lhs_is_ephemeral && !lhs_is_retained) {
                    fast_memory_usage += lhs_tensor.width * lhs_tensor.height;
                    if (fast_memory_usage > problem.fast_memory_capacity) {
                        return false;
                    }
                }
                q.emplace_back(lhs_tensor_idx, lhs_tensor, false);
            }

            Tensor rhs_tensor{
                .width = curr_tensor_dim.width,
                .height = inner_k,
            };
            if (!is_ignored_tensor_in_backward(rhs_tensor_idx) &&
                !visited_tensors.contains(rhs_tensor_idx)) {
                visited_tensors.insert(rhs_tensor_idx);

                bool const rhs_is_ephemeral = subgraph_produced.contains(rhs_tensor_idx);
                bool const rhs_is_retained = prev_retained_tensors.contains(rhs_tensor_idx);
                if (!rhs_is_ephemeral && !rhs_is_retained) {
                    fast_memory_usage += rhs_tensor.width * rhs_tensor.height;
                    if (fast_memory_usage > problem.fast_memory_capacity) {
                        return false;
                    }
                }
                q.emplace_back(rhs_tensor_idx, rhs_tensor, false);
            }
            continue;
        }

        if (op.inputs.size() == 1) {
            size_t const in_tensor_idx = op.inputs[0];
            if (!is_ignored_tensor_in_backward(in_tensor_idx) &&
                !visited_tensors.contains(in_tensor_idx)) {
                visited_tensors.insert(in_tensor_idx);
                q.emplace_back(in_tensor_idx, curr_tensor_dim, false);
            }
            continue;
        }

        for (size_t const in_tensor_idx : op.inputs) {
            if (is_ignored_tensor_in_backward(in_tensor_idx) ||
                visited_tensors.contains(in_tensor_idx)) {
                continue;
            }

            visited_tensors.insert(in_tensor_idx);
            bool const in_is_ephemeral = subgraph_produced.contains(in_tensor_idx);
            bool const in_is_retained = prev_retained_tensors.contains(in_tensor_idx);
            if (!in_is_ephemeral && !in_is_retained) {
                fast_memory_usage += curr_tensor_dim.width * curr_tensor_dim.height;
                if (fast_memory_usage > problem.fast_memory_capacity) {
                    return false;
                }
            }
            q.emplace_back(in_tensor_idx, curr_tensor_dim, false);
        }
    }

    return true;
}

} // namespace

auto Tiler::tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> {
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
        best_granularity_cache_[cache_key] = best_granularity;
        prev_retained_tensors =
            std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
    }

    return tiled_solution;
}

} // namespace mlsys
