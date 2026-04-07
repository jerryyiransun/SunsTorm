#include "fuser.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "cost_model.h"
#include "tiler.h"

namespace mlsys {

namespace {

struct TopologyInfo {
    std::vector<size_t> order;
    std::vector<size_t> rank;
};

struct SubgraphTensorUsage {
    std::set<size_t> produced;
    std::set<size_t> consumed;
};

enum class GreedyMoveType {
    kFuse,
    kRetain,
};

struct GreedyCandidate {
    GreedyMoveType type;
    Solution solution;
    int64_t potential_score = 0;
    std::string key;
};

struct ExactEvaluation {
    bool valid = false;
    Solution solution;
    double total_latency = 0.0;
};

struct SearchContext {
    const Problem& problem;
    const std::vector<int>& producer_op;
    const std::vector<std::vector<size_t>>& consumers_by_tensor;
    const TopologyInfo& topo;
    GreedyTiler tiler;
    CostModel cost_model;
    std::unordered_map<std::string, ExactEvaluation> evaluation_cache;
    bool has_best = false;
    double best_cost = std::numeric_limits<double>::infinity();
    Solution best_solution;
};

auto BuildProducerOpIndex(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer_op(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        for (size_t output_tensor : problem.ops[op_idx].outputs) {
            producer_op[output_tensor] = static_cast<int>(op_idx);
        }
    }
    return producer_op;
}

auto BuildConsumersByTensor(const Problem& problem) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> consumers(problem.tensors.size());
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        for (size_t tensor_idx : problem.ops[op_idx].inputs) {
            consumers[tensor_idx].push_back(op_idx);
        }
    }
    return consumers;
}

auto BuildTopologicalOrder(const Problem& problem) -> StatusOr<TopologyInfo> {
    size_t const num_ops = problem.ops.size();
    TopologyInfo topo;
    topo.rank.resize(num_ops, 0);

    std::vector<int> producer_op = BuildProducerOpIndex(problem);
    std::vector<int> indegree(num_ops, 0);
    std::vector<std::vector<size_t>> edges(num_ops);

    for (size_t op_idx = 0; op_idx < num_ops; ++op_idx) {
        for (size_t tensor_idx : problem.ops[op_idx].inputs) {
            int const producer = producer_op[tensor_idx];
            if (producer < 0) {
                continue;
            }
            edges[static_cast<size_t>(producer)].push_back(op_idx);
            ++indegree[op_idx];
        }
    }

    std::priority_queue<size_t, std::vector<size_t>, std::greater<>> ready;
    for (size_t op_idx = 0; op_idx < num_ops; ++op_idx) {
        if (indegree[op_idx] == 0) {
            ready.push(op_idx);
        }
    }

    while (!ready.empty()) {
        size_t const op_idx = ready.top();
        ready.pop();
        topo.rank[op_idx] = topo.order.size();
        topo.order.push_back(op_idx);

        for (size_t consumer_idx : edges[op_idx]) {
            --indegree[consumer_idx];
            if (indegree[consumer_idx] == 0) {
                ready.push(consumer_idx);
            }
        }
    }

    if (topo.order.size() != num_ops) {
        return absl::InvalidArgumentError("Problem graph contains a cycle");
    }

    return topo;
}

void CanonicalizeOps(std::vector<size_t>& ops, const std::vector<size_t>& topo_rank) {
    std::sort(ops.begin(), ops.end(), [&](size_t lhs, size_t rhs) {
        if (topo_rank[lhs] != topo_rank[rhs]) {
            return topo_rank[lhs] < topo_rank[rhs];
        }
        return lhs < rhs;
    });
    ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
}

void CanonicalizeTensors(std::vector<size_t>& tensors) {
    std::sort(tensors.begin(), tensors.end());
    tensors.erase(std::unique(tensors.begin(), tensors.end()), tensors.end());
}

auto BuildSubgraphUsages(const Problem& problem, const Solution& solution)
    -> std::vector<SubgraphTensorUsage> {
    std::vector<SubgraphTensorUsage> usages;
    usages.reserve(solution.subgraphs.size());

    for (const auto& subgraph : solution.subgraphs) {
        SubgraphTensorUsage usage;
        for (size_t op_idx : subgraph.ops) {
            usage.produced.insert(problem.ops[op_idx].outputs[0]);
            for (size_t tensor_idx : problem.ops[op_idx].inputs) {
                usage.consumed.insert(tensor_idx);
            }
        }
        usages.push_back(std::move(usage));
    }

    return usages;
}

auto MakeStateKey(const Solution& solution) -> std::string {
    std::string key;
    for (const auto& subgraph : solution.subgraphs) {
        key += "ops:";
        for (size_t op_idx : subgraph.ops) {
            key += std::to_string(op_idx);
            key.push_back(',');
        }
        key += "|retain:";
        for (size_t tensor_idx : subgraph.tensors_to_retain) {
            key += std::to_string(tensor_idx);
            key.push_back(',');
        }
        key.push_back(';');
    }
    return key;
}

auto TensorSize(const Problem& problem, size_t tensor_idx) -> int64_t {
    return problem.tensors[tensor_idx].width * problem.tensors[tensor_idx].height;
}

auto NormalizeAndValidateSolution(const Problem& problem, const std::vector<size_t>& topo_rank,
                                  const std::vector<int>& producer_op, Solution& solution) -> bool {
    for (auto& subgraph : solution.subgraphs) {
        CanonicalizeOps(subgraph.ops, topo_rank);
        CanonicalizeTensors(subgraph.tensors_to_retain);
        subgraph.traversal_order = std::nullopt;
        subgraph.subgraph_latency = 0.0;
    }

    solution.subgraphs.erase(
        std::remove_if(solution.subgraphs.begin(), solution.subgraphs.end(),
                       [](const Subgraph& subgraph) { return subgraph.ops.empty(); }),
        solution.subgraphs.end());

    std::vector<bool> globally_available(problem.tensors.size(), true);
    for (size_t tensor_idx = 0; tensor_idx < problem.tensors.size(); ++tensor_idx) {
        if (producer_op[tensor_idx] >= 0) {
            globally_available[tensor_idx] = false;
        }
    }

    std::vector<bool> retained_available(problem.tensors.size(), false);

    for (auto& subgraph : solution.subgraphs) {
        std::vector<bool> local_available(problem.tensors.size(), false);
        std::set<size_t> produced;
        std::set<size_t> consumed;

        for (size_t op_idx : subgraph.ops) {
            if (op_idx >= problem.ops.size()) {
                return false;
            }

            for (size_t tensor_idx : problem.ops[op_idx].inputs) {
                bool const available = globally_available[tensor_idx] ||
                                       retained_available[tensor_idx] ||
                                       local_available[tensor_idx];
                if (!available) {
                    return false;
                }
                consumed.insert(tensor_idx);
            }

            size_t const output_tensor = problem.ops[op_idx].outputs[0];
            local_available[output_tensor] = true;
            produced.insert(output_tensor);
        }

        for (size_t tensor_idx : produced) {
            if (!consumed.contains(tensor_idx)) {
                globally_available[tensor_idx] = true;
            }
        }

        std::vector<size_t> normalized_retains;
        normalized_retains.reserve(subgraph.tensors_to_retain.size());
        for (size_t tensor_idx : subgraph.tensors_to_retain) {
            if (tensor_idx >= problem.tensors.size()) {
                return false;
            }

            bool const touched = produced.contains(tensor_idx) || consumed.contains(tensor_idx) ||
                                 retained_available[tensor_idx];
            bool const available = globally_available[tensor_idx] ||
                                   retained_available[tensor_idx] || local_available[tensor_idx];
            if (touched && available) {
                normalized_retains.push_back(tensor_idx);
            }
        }

        subgraph.tensors_to_retain = std::move(normalized_retains);

        std::fill(retained_available.begin(), retained_available.end(), false);
        for (size_t tensor_idx : subgraph.tensors_to_retain) {
            retained_available[tensor_idx] = true;
        }
    }

    return true;
}

auto BuildDirectMergeSolution(const Solution& solution, size_t producer_sg_idx,
                              size_t consumer_sg_idx, const std::vector<size_t>& topo_rank)
    -> Solution {
    Solution merged_solution;
    merged_solution.subgraphs.reserve(solution.subgraphs.size() - 1);

    Subgraph merged_subgraph = solution.subgraphs[consumer_sg_idx];
    merged_subgraph.ops.insert(merged_subgraph.ops.end(),
                               solution.subgraphs[producer_sg_idx].ops.begin(),
                               solution.subgraphs[producer_sg_idx].ops.end());
    merged_subgraph.tensors_to_retain.insert(
        merged_subgraph.tensors_to_retain.end(),
        solution.subgraphs[producer_sg_idx].tensors_to_retain.begin(),
        solution.subgraphs[producer_sg_idx].tensors_to_retain.end());
    CanonicalizeOps(merged_subgraph.ops, topo_rank);
    CanonicalizeTensors(merged_subgraph.tensors_to_retain);

    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        if (sg_idx == producer_sg_idx) {
            continue;
        }
        if (sg_idx == consumer_sg_idx) {
            merged_solution.subgraphs.push_back(merged_subgraph);
            continue;
        }
        merged_solution.subgraphs.push_back(solution.subgraphs[sg_idx]);
    }

    return merged_solution;
}

auto BuildCloneFuseSolution(const Solution& solution, size_t producer_sg_idx,
                            size_t consumer_sg_idx, const std::vector<size_t>& topo_rank)
    -> Solution {
    Solution cloned_solution = solution;
    Subgraph& fused_subgraph = cloned_solution.subgraphs[consumer_sg_idx];
    fused_subgraph.ops.insert(fused_subgraph.ops.end(),
                              solution.subgraphs[producer_sg_idx].ops.begin(),
                              solution.subgraphs[producer_sg_idx].ops.end());
    CanonicalizeOps(fused_subgraph.ops, topo_rank);
    CanonicalizeTensors(fused_subgraph.tensors_to_retain);
    return cloned_solution;
}

auto BuildRetainSolution(const Solution& solution, size_t producer_sg_idx, size_t consumer_sg_idx,
                         size_t tensor_idx) -> Solution {
    Solution retained_solution = solution;
    for (size_t sg_idx = producer_sg_idx; sg_idx < consumer_sg_idx; ++sg_idx) {
        retained_solution.subgraphs[sg_idx].tensors_to_retain.push_back(tensor_idx);
    }
    return retained_solution;
}

auto SharedProducerConsumerTensors(const SubgraphTensorUsage& producer_usage,
                                   const SubgraphTensorUsage& consumer_usage)
    -> std::vector<size_t> {
    std::vector<size_t> shared;
    std::set_intersection(producer_usage.produced.begin(), producer_usage.produced.end(),
                          consumer_usage.consumed.begin(), consumer_usage.consumed.end(),
                          std::back_inserter(shared));
    return shared;
}

auto FindLatestSubgraphContainingOp(const Solution& solution, size_t op_idx, size_t before_idx)
    -> std::optional<size_t> {
    for (size_t sg_idx = before_idx; sg_idx-- > 0;) {
        const auto& ops = solution.subgraphs[sg_idx].ops;
        if (std::find(ops.begin(), ops.end(), op_idx) != ops.end()) {
            return sg_idx;
        }
    }
    return std::nullopt;
}

void PrefuseUnaryUniquePointwise(const Problem& problem, const std::vector<int>& producer_op,
                                 const std::vector<std::vector<size_t>>& consumers_by_tensor,
                                 const std::vector<size_t>& topo_rank, Solution& solution) {
    auto subgraph_contains_op = [&](const Subgraph& subgraph, size_t op_idx) -> bool {
        return std::find(subgraph.ops.begin(), subgraph.ops.end(), op_idx) != subgraph.ops.end();
    };

    while (true) {
        bool changed = false;
        for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
            const auto& current_subgraph = solution.subgraphs[sg_idx];
            for (size_t pointwise_op_idx : current_subgraph.ops) {
                const Op& pointwise_op = problem.ops[pointwise_op_idx];
                if (pointwise_op.op_type != "Pointwise" || pointwise_op.inputs.size() != 1) {
                    continue;
                }

                // Case A: shared tensor is the unary pointwise input.
                size_t const input_tensor = pointwise_op.inputs[0];
                int const producer = producer_op[input_tensor];
                if (producer >= 0) {
                    auto producer_sg_idx = FindLatestSubgraphContainingOp(
                        solution, static_cast<size_t>(producer), sg_idx);
                    if (producer_sg_idx.has_value()) {
                        // TODO: Add a tiler/cost-aware guard for MatMul->Pointwise hard pre-fuse.
                        // Split-k behavior can make these fusions latency-regressive.
                        Solution candidate = BuildDirectMergeSolution(
                            solution, producer_sg_idx.value(), sg_idx, topo_rank);
                        if (NormalizeAndValidateSolution(problem, topo_rank, producer_op,
                                                         candidate)) {
                            solution = std::move(candidate);
                            changed = true;
                            break;
                        }
                    }
                }

                if (changed) {
                    break;
                }

                // Case B: shared tensor is the unary pointwise output.
                size_t const output_tensor = pointwise_op.outputs[0];
                const auto& output_consumers = consumers_by_tensor[output_tensor];
                if (output_consumers.empty()) {
                    continue;
                }

                bool all_consumers_pointwise = true;
                for (size_t consumer_op_idx : output_consumers) {
                    if (problem.ops[consumer_op_idx].op_type != "Pointwise") {
                        all_consumers_pointwise = false;
                        break;
                    }
                }
                if (!all_consumers_pointwise) {
                    continue;
                }

                std::optional<size_t> target_sg_idx;
                for (size_t candidate_sg_idx = sg_idx + 1;
                     candidate_sg_idx < solution.subgraphs.size(); ++candidate_sg_idx) {
                    bool all_consumers_in_candidate = true;
                    for (size_t consumer_op_idx : output_consumers) {
                        if (!subgraph_contains_op(solution.subgraphs[candidate_sg_idx],
                                                  consumer_op_idx)) {
                            all_consumers_in_candidate = false;
                            break;
                        }
                    }

                    if (all_consumers_in_candidate) {
                        target_sg_idx = candidate_sg_idx;
                        break;
                    }
                }
                if (!target_sg_idx.has_value()) {
                    continue;
                }

                Solution candidate =
                    BuildDirectMergeSolution(solution, sg_idx, target_sg_idx.value(), topo_rank);
                if (!NormalizeAndValidateSolution(problem, topo_rank, producer_op, candidate)) {
                    continue;
                }

                solution = std::move(candidate);
                changed = true;
                break;
            }

            if (changed) {
                break;
            }
        }

        if (!changed) {
            return;
        }
    }
}

auto GenerateGreedyCandidates(const Problem& problem, const std::vector<int>& producer_op,
                              const std::vector<size_t>& topo_rank, const Solution& state)
    -> std::vector<GreedyCandidate> {
    std::vector<GreedyCandidate> candidates;
    std::unordered_set<std::string> seen_keys;
    std::string const parent_key = MakeStateKey(state);
    auto usages = BuildSubgraphUsages(problem, state);

    for (size_t producer_sg_idx = 0; producer_sg_idx < state.subgraphs.size(); ++producer_sg_idx) {
        for (size_t consumer_sg_idx = producer_sg_idx + 1; consumer_sg_idx < state.subgraphs.size();
             ++consumer_sg_idx) {
            std::vector<size_t> const shared_tensors =
                SharedProducerConsumerTensors(usages[producer_sg_idx], usages[consumer_sg_idx]);
            if (shared_tensors.empty()) {
                continue;
            }

            int64_t fuse_score = 0;
            for (size_t tensor_idx : shared_tensors) {
                fuse_score += TensorSize(problem, tensor_idx);
            }

            if (fuse_score > 0) {
                Solution direct_merge =
                    BuildDirectMergeSolution(state, producer_sg_idx, consumer_sg_idx, topo_rank);
                if (NormalizeAndValidateSolution(problem, topo_rank, producer_op, direct_merge)) {
                    std::string const key = MakeStateKey(direct_merge);
                    if (key != parent_key && seen_keys.insert(key).second) {
                        candidates.push_back(GreedyCandidate{.type = GreedyMoveType::kFuse,
                                                             .solution = std::move(direct_merge),
                                                             .potential_score = fuse_score,
                                                             .key = key});
                    }
                } else {
                    Solution clone_fuse =
                        BuildCloneFuseSolution(state, producer_sg_idx, consumer_sg_idx, topo_rank);
                    if (NormalizeAndValidateSolution(problem, topo_rank, producer_op, clone_fuse)) {
                        std::string const key = MakeStateKey(clone_fuse);
                        if (key != parent_key && seen_keys.insert(key).second) {
                            candidates.push_back(GreedyCandidate{.type = GreedyMoveType::kFuse,
                                                                 .solution = std::move(clone_fuse),
                                                                 .potential_score = fuse_score,
                                                                 .key = key});
                        }
                    }
                }
            }

            for (size_t tensor_idx : shared_tensors) {
                int64_t const retain_score = TensorSize(problem, tensor_idx);
                if (retain_score <= 0) {
                    continue;
                }

                Solution retain_candidate =
                    BuildRetainSolution(state, producer_sg_idx, consumer_sg_idx, tensor_idx);
                if (!NormalizeAndValidateSolution(problem, topo_rank, producer_op,
                                                  retain_candidate)) {
                    continue;
                }

                std::string const key = MakeStateKey(retain_candidate);
                if (key == parent_key || !seen_keys.insert(key).second) {
                    continue;
                }

                candidates.push_back(GreedyCandidate{.type = GreedyMoveType::kRetain,
                                                     .solution = std::move(retain_candidate),
                                                     .potential_score = retain_score,
                                                     .key = key});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const GreedyCandidate& lhs, const GreedyCandidate& rhs) {
                  if (lhs.potential_score != rhs.potential_score) {
                      return lhs.potential_score > rhs.potential_score;
                  }
                  return lhs.key < rhs.key;
              });
    return candidates;
}

auto EvaluateCandidateState(SearchContext& context, const GreedyCandidate& candidate)
    -> const ExactEvaluation& {
    auto cache_it = context.evaluation_cache.find(candidate.key);
    if (cache_it != context.evaluation_cache.end()) {
        return cache_it->second;
    }

    ExactEvaluation evaluation;
    auto tiled = context.tiler.tile(context.problem, candidate.solution);
    if (tiled.ok()) {
        auto estimated = context.cost_model.estimate(tiled.value());
        if (estimated.ok()) {
            evaluation.valid = true;
            evaluation.solution = std::get<0>(estimated.value());
            evaluation.total_latency = std::get<1>(estimated.value());
        }
    }

    auto [it, _] = context.evaluation_cache.emplace(candidate.key, std::move(evaluation));
    return it->second;
}

void MaybeUpdateBest(SearchContext& context, const ExactEvaluation& evaluation) {
    if (!evaluation.valid) {
        return;
    }
    if (!context.has_best || evaluation.total_latency < context.best_cost) {
        context.has_best = true;
        context.best_cost = evaluation.total_latency;
        context.best_solution = evaluation.solution;
    }
}

void ExploreGreedyState(const GreedyFuserConfig& config, SearchContext& context,
                        const Solution& state, int depth) {
    if (depth >= config.search_depth) {
        return;
    }

    auto candidates =
        GenerateGreedyCandidates(context.problem, context.producer_op, context.topo.rank, state);

    struct RankedNextState {
        double total_latency;
        Solution solution;
    };

    size_t exact_evaluated = 0;
    std::vector<RankedNextState> next_states;

    for (const auto& candidate : candidates) {
        if (exact_evaluated >= static_cast<size_t>(config.beam_width)) {
            break;
        }

        const ExactEvaluation& evaluation = EvaluateCandidateState(context, candidate);
        ++exact_evaluated;
        if (!evaluation.valid) {
            continue;
        }

        MaybeUpdateBest(context, evaluation);
        next_states.push_back(RankedNextState{.total_latency = evaluation.total_latency,
                                              .solution = candidate.solution});
    }

    std::sort(next_states.begin(), next_states.end(),
              [](const RankedNextState& lhs, const RankedNextState& rhs) {
                  if (lhs.total_latency != rhs.total_latency) {
                      return lhs.total_latency < rhs.total_latency;
                  }
                  return MakeStateKey(lhs.solution) < MakeStateKey(rhs.solution);
              });

    for (const auto& next_state : next_states) {
        ExploreGreedyState(config, context, next_state.solution, depth + 1);
    }
}

auto BuildInitialGreedySolution(const Problem& problem, const TopologyInfo& topo,
                                const std::vector<int>& producer_op,
                                const std::vector<std::vector<size_t>>& consumers_by_tensor)
    -> StatusOr<Solution> {
    Solution initial_solution;
    for (size_t op_idx : topo.order) {
        Subgraph subgraph;
        subgraph.ops = {op_idx};
        subgraph.tensors_to_retain = {};
        subgraph.traversal_order = std::nullopt;
        subgraph.subgraph_latency = 0.0;
        initial_solution.subgraphs.push_back(subgraph);
    }

    PrefuseUnaryUniquePointwise(problem, producer_op, consumers_by_tensor, topo.rank,
                                initial_solution);

    if (!NormalizeAndValidateSolution(problem, topo.rank, producer_op, initial_solution)) {
        return absl::FailedPreconditionError("Failed to build a valid initial greedy solution");
    }

    return initial_solution;
}

auto CanIncludeOp(const Problem& problem, const std::vector<int>& producer_op, size_t op_idx,
                  const std::vector<char>& covered_ops, const std::vector<char>& selected_ops)
    -> bool {
    for (size_t input_tensor : problem.ops[op_idx].inputs) {
        int const producer = producer_op[input_tensor];
        if (producer >= 0 && !covered_ops[producer] && !selected_ops[producer]) {
            return false;
        }
    }
    return true;
}

void EnumerateSubgraphCandidates(const Problem& problem, const std::vector<int>& producer_op,
                                 const std::vector<char>& covered_ops,
                                 const std::vector<size_t>& topo_order, size_t order_pos,
                                 std::vector<char>& selected_ops,
                                 std::vector<size_t>& current_subgraph, bool introduces_new_op,
                                 std::vector<std::vector<size_t>>& candidates) {
    if (order_pos == topo_order.size()) {
        if (!current_subgraph.empty() && introduces_new_op) {
            candidates.push_back(current_subgraph);
        }
        return;
    }

    size_t const op_idx = topo_order[order_pos];

    EnumerateSubgraphCandidates(problem, producer_op, covered_ops, topo_order, order_pos + 1,
                                selected_ops, current_subgraph, introduces_new_op, candidates);

    if (!CanIncludeOp(problem, producer_op, op_idx, covered_ops, selected_ops)) {
        return;
    }

    current_subgraph.push_back(op_idx);
    selected_ops[op_idx] = true;
    EnumerateSubgraphCandidates(problem, producer_op, covered_ops, topo_order, order_pos + 1,
                                selected_ops, current_subgraph,
                                introduces_new_op || !covered_ops[op_idx], candidates);
    selected_ops[op_idx] = false;
    current_subgraph.pop_back();
}

auto BuildSchedulableSubgraphCandidates(const Problem& problem, const std::vector<int>& producer_op,
                                        const std::vector<char>& covered_ops,
                                        const std::vector<size_t>& topo_order)
    -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> candidates;
    std::vector<char> selected_ops(problem.ops.size(), false);
    std::vector<size_t> current_subgraph;
    EnumerateSubgraphCandidates(problem, producer_op, covered_ops, topo_order, 0, selected_ops,
                                current_subgraph, false, candidates);
    return candidates;
}

void EnumerateSchedules(const Problem& problem, const std::vector<int>& producer_op,
                        const std::vector<size_t>& topo_order, std::vector<char>& covered_ops,
                        size_t covered_count, std::vector<std::vector<size_t>>& current_schedule,
                        std::vector<std::vector<std::vector<size_t>>>& schedules) {
    if (covered_count == problem.ops.size()) {
        schedules.push_back(current_schedule);
        return;
    }

    for (const auto& candidate :
         BuildSchedulableSubgraphCandidates(problem, producer_op, covered_ops, topo_order)) {
        current_schedule.push_back(candidate);

        std::vector<size_t> newly_covered;
        for (size_t op_idx : candidate) {
            if (!covered_ops[op_idx]) {
                covered_ops[op_idx] = true;
                newly_covered.push_back(op_idx);
            }
        }

        EnumerateSchedules(problem, producer_op, topo_order, covered_ops,
                           covered_count + newly_covered.size(), current_schedule, schedules);

        for (size_t op_idx : newly_covered) {
            covered_ops[op_idx] = false;
        }
        current_schedule.pop_back();
    }
}

} // namespace

GreedyFuser::GreedyFuser(GreedyFuserConfig config) : config_(config) {}

auto GreedyFuser::fuse(const Problem& problem) -> StatusOr<Solution> {
    if (config_.search_depth < 0) {
        return absl::InvalidArgumentError("search_depth must be non-negative");
    }
    if (config_.beam_width <= 0) {
        return absl::InvalidArgumentError("beam_width must be positive");
    }
    if (problem.ops.empty()) {
        return Solution{};
    }

    auto topo_or = BuildTopologicalOrder(problem);
    if (!topo_or.ok()) {
        return topo_or.status();
    }
    TopologyInfo topo = topo_or.value();

    std::vector<int> producer_op = BuildProducerOpIndex(problem);
    std::vector<std::vector<size_t>> consumers_by_tensor = BuildConsumersByTensor(problem);
    auto initial_solution_or =
        BuildInitialGreedySolution(problem, topo, producer_op, consumers_by_tensor);
    if (!initial_solution_or.ok()) {
        return initial_solution_or.status();
    }

    Solution initial_solution = initial_solution_or.value();
    SearchContext context{
        .problem = problem,
        .producer_op = producer_op,
        .consumers_by_tensor = consumers_by_tensor,
        .topo = topo,
        .tiler = GreedyTiler(),
        .cost_model = CostModel(problem),
    };

    GreedyCandidate root_candidate{
        .type = GreedyMoveType::kFuse,
        .solution = initial_solution,
        .potential_score = 0,
        .key = MakeStateKey(initial_solution),
    };
    const ExactEvaluation& root_evaluation = EvaluateCandidateState(context, root_candidate);
    MaybeUpdateBest(context, root_evaluation);

    ExploreGreedyState(config_, context, initial_solution, 0);

    if (!context.has_best) {
        return absl::InternalError("Greedy fuser found no valid plans");
    }

    return context.best_solution;
}

auto BruteForceFuser::fuse(const Problem& problem) -> StatusOr<std::vector<Solution>> {
    std::vector<Solution> results;
    size_t const num_ops = problem.ops.size();
    if (num_ops == 0) {
        return results;
    }

    auto topo_or = BuildTopologicalOrder(problem);
    if (!topo_or.ok()) {
        return topo_or.status();
    }

    std::vector<int> producer_op = BuildProducerOpIndex(problem);
    std::vector<char> covered_ops(num_ops, false);
    std::vector<std::vector<size_t>> current_schedule;
    std::vector<std::vector<std::vector<size_t>>> schedules;
    EnumerateSchedules(problem, producer_op, topo_or.value().order, covered_ops, 0,
                       current_schedule, schedules);

    for (const auto& subgraphs_ops : schedules) {
        std::vector<std::vector<size_t>> all_cands;
        for (const auto& ops_list : subgraphs_ops) {
            std::set<size_t> cands;
            for (size_t op_idx : ops_list) {
                for (size_t in_idx : problem.ops[op_idx].inputs) {
                    cands.insert(in_idx);
                }
                for (size_t out_idx : problem.ops[op_idx].outputs) {
                    cands.insert(out_idx);
                }
            }
            all_cands.push_back(std::vector<size_t>(cands.begin(), cands.end()));
        }

        size_t const num_subgraphs = subgraphs_ops.size();
        std::vector<size_t> num_subsets;
        size_t total_combinations = 1;
        for (const auto& cands : all_cands) {
            size_t const subsets = 1ULL << cands.size();
            num_subsets.push_back(subsets);
            total_combinations *= subsets;
        }

        for (size_t combo = 0; combo < total_combinations; ++combo) {
            Solution solution;
            size_t current_combo = combo;

            for (size_t sg_idx = 0; sg_idx < num_subgraphs; ++sg_idx) {
                size_t const subset_mask = current_combo % num_subsets[sg_idx];
                current_combo /= num_subsets[sg_idx];

                Subgraph subgraph;
                subgraph.ops = subgraphs_ops[sg_idx];
                for (size_t tensor_pos = 0; tensor_pos < all_cands[sg_idx].size(); ++tensor_pos) {
                    if (((subset_mask >> tensor_pos) & 1) != 0) {
                        subgraph.tensors_to_retain.push_back(all_cands[sg_idx][tensor_pos]);
                    }
                }
                solution.subgraphs.push_back(subgraph);
            }

            results.push_back(std::move(solution));
        }
    }

    return results;
}

} // namespace mlsys
