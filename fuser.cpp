#include "fuser.h"
#include "fuser_logging.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
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

#if MLSYS_ENABLE_FUSER_LOGGING
#define MLSYS_FUSER_LOG_ARGS(...) , __VA_ARGS__
#else
#define MLSYS_FUSER_LOG_ARGS(...)
#endif

struct TopologyInfo {
    std::vector<size_t> order;
    std::vector<size_t> rank;
};

struct SubgraphTensorUsage {
    std::set<size_t> produced;
    std::set<size_t> consumed;
};

enum class GreedyMoveType {
    kDirectFuse,
    kCloneFuse,
    kRetain,
};

auto MoveTypeName(GreedyMoveType type) -> const char* {
    switch (type) {
    case GreedyMoveType::kDirectFuse:
        return "DirectFuse";
    case GreedyMoveType::kCloneFuse:
        return "CloneFuse";
    case GreedyMoveType::kRetain:
        return "Retain";
    }
    return "Unknown";
}

struct GreedyCandidate {
    GreedyMoveType type;
    Solution solution;
    int64_t potential_score = 0;
    std::string key;
    std::vector<size_t> producer_subgraph_ops;
    std::vector<size_t> consumer_subgraph_ops;
#if MLSYS_ENABLE_FUSER_LOGGING
    size_t producer_sg_idx = 0;
    size_t consumer_sg_idx = 0;
    std::vector<size_t> touched_tensors;
#endif
};

struct RankedGreedyCandidate {
    const GreedyCandidate* candidate = nullptr;
    double penalized_score = 0.0;
    double op_hit_sum = 0.0;
};

void LogTopKCandidates(const std::vector<RankedGreedyCandidate>& ranked_candidates, int depth,
                       int beam_width) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserTopKLoggingEnabled()) {
        return;
    }

    size_t const beam_limit = beam_width <= 0 ? 0 : static_cast<size_t>(beam_width);
    size_t const limit = std::min(beam_limit, ranked_candidates.size());
    std::vector<FuserTopKCandidateLogData> top_candidates;
    top_candidates.reserve(limit);

    for (size_t idx = 0; idx < limit; ++idx) {
        RankedGreedyCandidate const& ranked = ranked_candidates[idx];
        GreedyCandidate const& candidate = *ranked.candidate;
        FuserTopKCandidateLogData log_entry;
        log_entry.move_name = MoveTypeName(candidate.type);
        log_entry.potential_score = candidate.potential_score;
        log_entry.penalized_score = ranked.penalized_score;
        log_entry.op_hit_sum = ranked.op_hit_sum;
        log_entry.producer_sg_idx = candidate.producer_sg_idx;
        log_entry.consumer_sg_idx = candidate.consumer_sg_idx;
        log_entry.touched_tensors = candidate.touched_tensors;
        log_entry.producer_subgraph_ops = candidate.producer_subgraph_ops;
        log_entry.consumer_subgraph_ops = candidate.consumer_subgraph_ops;
        top_candidates.push_back(std::move(log_entry));
    }

    LogFuserTopKCandidates(depth, ranked_candidates.size(), beam_limit, top_candidates);
#else
    (void)ranked_candidates;
    (void)depth;
    (void)beam_width;
#endif
}

struct ExactEvaluation {
    bool valid = false;
    Solution solution;
    double total_latency = 0.0;
};

struct EvaluatedCandidateResult {
    const GreedyCandidate* candidate = nullptr;
    bool improved = false;
};

struct SearchContext {
    const Problem& problem;
    const std::vector<int>& producer_op;
    const std::vector<std::vector<size_t>>& consumers_by_tensor;
    const TopologyInfo& topo;
    GreedyTiler tiler;
    CostModel cost_model;
    std::unordered_map<std::string, std::vector<GreedyCandidate>> scored_candidates_cache;
    std::unordered_map<size_t, double> topk_hits_by_op;
    bool has_best = false;
    double best_cost = std::numeric_limits<double>::infinity();
    Solution best_solution;
};

struct SubgraphEdgeDegree {
    size_t incoming = 0;
    size_t outgoing = 0;
};

struct GreedySearchFrame {
    Solution state;
    double current_cost = std::numeric_limits<double>::infinity();
    double base_cost = std::numeric_limits<double>::infinity();
    int remaining_lookahead = 0;
    int depth = 0;
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
                         const std::vector<size_t>& tensor_indices) -> Solution {
    Solution retained_solution = solution;
    for (size_t sg_idx = producer_sg_idx; sg_idx < consumer_sg_idx; ++sg_idx) {
        auto& retains = retained_solution.subgraphs[sg_idx].tensors_to_retain;
        retains.insert(retains.end(), tensor_indices.begin(), tensor_indices.end());
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

auto OpConsumesTensor(const Problem& problem, size_t op_idx, size_t tensor_idx) -> bool {
    const Op& op = problem.ops[op_idx];
    return std::find(op.inputs.begin(), op.inputs.end(), tensor_idx) != op.inputs.end();
}

auto TensorConsumerOpsInSubgraph(const Problem& problem, const Subgraph& subgraph,
                                 size_t tensor_idx) -> std::vector<size_t> {
    std::vector<size_t> consumers;
    for (size_t op_idx : subgraph.ops) {
        if (OpConsumesTensor(problem, op_idx, tensor_idx)) {
            consumers.push_back(op_idx);
        }
    }
    return consumers;
}

auto SubgraphContainsAllOps(const Subgraph& subgraph, const std::vector<size_t>& ops) -> bool {
    for (size_t op_idx : ops) {
        if (std::find(subgraph.ops.begin(), subgraph.ops.end(), op_idx) == subgraph.ops.end()) {
            return false;
        }
    }
    return true;
}

auto ComputeNewlyInternalizedBoundaryScore(const Problem& problem,
                                           const std::vector<int>& producer_op,
                                           const Solution& parent_state, size_t producer_sg_idx,
                                           size_t consumer_sg_idx,
                                           const std::vector<size_t>& shared_tensors,
                                           const Solution& candidate_state) -> int64_t {
    int64_t score = 0;
    const Subgraph& parent_consumer = parent_state.subgraphs[consumer_sg_idx];

    for (size_t tensor_idx : shared_tensors) {
        int const producer = producer_op[tensor_idx];
        if (producer < 0) {
            continue;
        }

        if (parent_state.subgraphs[producer_sg_idx].ops.end() ==
            std::find(parent_state.subgraphs[producer_sg_idx].ops.begin(),
                      parent_state.subgraphs[producer_sg_idx].ops.end(),
                      static_cast<size_t>(producer))) {
            continue;
        }

        std::vector<size_t> consumer_ops =
            TensorConsumerOpsInSubgraph(problem, parent_consumer, tensor_idx);
        if (consumer_ops.empty()) {
            continue;
        }

        bool internalized = false;
        for (const auto& candidate_subgraph : candidate_state.subgraphs) {
            if (!SubgraphContainsAllOps(candidate_subgraph, consumer_ops)) {
                continue;
            }
            if (std::find(candidate_subgraph.ops.begin(), candidate_subgraph.ops.end(),
                          static_cast<size_t>(producer)) != candidate_subgraph.ops.end()) {
                internalized = true;
                break;
            }
        }

        if (internalized) {
            score += TensorSize(problem, tensor_idx);
        }
    }

    return score;
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
                if (pointwise_op.inputs.size() != 1) {
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
    std::unordered_map<std::string, size_t> candidate_idx_by_key;
    std::string const parent_key = MakeStateKey(state);
    auto usages = BuildSubgraphUsages(problem, state);

    auto AddOrUpdateCandidate =
        [&](GreedyMoveType type, Solution solution, int64_t potential_score,
            const std::vector<size_t>& producer_subgraph_ops,
            const std::vector<size_t>& consumer_subgraph_ops MLSYS_FUSER_LOG_ARGS(
                size_t producer_sg_idx, size_t consumer_sg_idx,
                const std::vector<size_t>& touched_tensors)) {
            if (potential_score <= 0) {
                return;
            }
            std::string const key = MakeStateKey(solution);
            if (key == parent_key) {
                return;
            }

            auto existing = candidate_idx_by_key.find(key);
            if (existing == candidate_idx_by_key.end()) {
                candidate_idx_by_key[key] = candidates.size();
                candidates.push_back(GreedyCandidate {
                    .type = type, .solution = std::move(solution),
                    .potential_score = potential_score, .key = key,
                    .producer_subgraph_ops = producer_subgraph_ops,
                    .consumer_subgraph_ops = consumer_subgraph_ops,
#if MLSYS_ENABLE_FUSER_LOGGING
                    .producer_sg_idx = producer_sg_idx, .consumer_sg_idx = consumer_sg_idx,
                    .touched_tensors = touched_tensors,
#endif
                });
                return;
            }

            GreedyCandidate& kept = candidates[existing->second];
            if (potential_score > kept.potential_score) {
                kept.type = type;
                kept.solution = std::move(solution);
                kept.potential_score = potential_score;
                kept.key = key;
                kept.producer_subgraph_ops = producer_subgraph_ops;
                kept.consumer_subgraph_ops = consumer_subgraph_ops;
#if MLSYS_ENABLE_FUSER_LOGGING
                kept.producer_sg_idx = producer_sg_idx;
                kept.consumer_sg_idx = consumer_sg_idx;
                kept.touched_tensors = touched_tensors;
#endif
            }
        };

    for (size_t producer_sg_idx = 0; producer_sg_idx < state.subgraphs.size(); ++producer_sg_idx) {
        for (size_t consumer_sg_idx = producer_sg_idx + 1; consumer_sg_idx < state.subgraphs.size();
             ++consumer_sg_idx) {
            std::vector<size_t> const shared_tensors =
                SharedProducerConsumerTensors(usages[producer_sg_idx], usages[consumer_sg_idx]);
            if (shared_tensors.empty()) {
                continue;
            }

            bool consumer_is_producer = false;
            // for (size_t downstream_sg_idx = consumer_sg_idx + 1;
            //      downstream_sg_idx < state.subgraphs.size(); ++downstream_sg_idx) {
            //     if (!SharedProducerConsumerTensors(usages[consumer_sg_idx],
            //                                        usages[downstream_sg_idx])
            //              .empty()) {
            //         consumer_is_producer = true;
            //         break;
            //     }
            // }
            int64_t const potential_saving_multiplier = consumer_is_producer ? 2 : 1;

            Solution direct_merge =
                BuildDirectMergeSolution(state, producer_sg_idx, consumer_sg_idx, topo_rank);
            if (NormalizeAndValidateSolution(problem, topo_rank, producer_op, direct_merge)) {
                int64_t const direct_score = potential_saving_multiplier *
                                             ComputeNewlyInternalizedBoundaryScore(
                                                 problem, producer_op, state, producer_sg_idx,
                                                 consumer_sg_idx, shared_tensors, direct_merge);
                AddOrUpdateCandidate(GreedyMoveType::kDirectFuse, std::move(direct_merge),
                                     direct_score, state.subgraphs[producer_sg_idx].ops,
                                     state.subgraphs[consumer_sg_idx].ops MLSYS_FUSER_LOG_ARGS(
                                         producer_sg_idx, consumer_sg_idx, shared_tensors));
            }

            Solution clone_fuse =
                BuildCloneFuseSolution(state, producer_sg_idx, consumer_sg_idx, topo_rank);
            if (NormalizeAndValidateSolution(problem, topo_rank, producer_op, clone_fuse)) {
                int64_t const clone_score =
                    potential_saving_multiplier * ComputeNewlyInternalizedBoundaryScore(
                                                      problem, producer_op, state, producer_sg_idx,
                                                      consumer_sg_idx, shared_tensors, clone_fuse);
                AddOrUpdateCandidate(GreedyMoveType::kCloneFuse, std::move(clone_fuse), clone_score,
                                     state.subgraphs[producer_sg_idx].ops,
                                     state.subgraphs[consumer_sg_idx].ops MLSYS_FUSER_LOG_ARGS(
                                         producer_sg_idx, consumer_sg_idx, shared_tensors));
            }
            int64_t retain_score = 0;
            for (size_t tensor_idx : shared_tensors) {
                int64_t const tensor_score = TensorSize(problem, tensor_idx);
                if (tensor_score <= 0) {
                    continue;
                }
                retain_score += tensor_score;
            }

            if (retain_score > 0) {
                Solution retain_candidate =
                    BuildRetainSolution(state, producer_sg_idx, consumer_sg_idx, shared_tensors);
                if (NormalizeAndValidateSolution(problem, topo_rank, producer_op,
                                                 retain_candidate)) {
                    AddOrUpdateCandidate(GreedyMoveType::kRetain, std::move(retain_candidate),
                                         potential_saving_multiplier * retain_score,
                                         state.subgraphs[producer_sg_idx].ops,
                                         state.subgraphs[consumer_sg_idx].ops MLSYS_FUSER_LOG_ARGS(
                                             producer_sg_idx, consumer_sg_idx, shared_tensors));
                }
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

auto GenerateGreedyCandidatesAverage(const Problem& problem, const std::vector<int>& producer_op,
                                     const std::vector<size_t>& topo_rank, const Solution& state)
    -> std::vector<GreedyCandidate> {
    std::vector<GreedyCandidate> candidates;
    std::unordered_map<std::string, size_t> candidate_idx_by_key;
    std::string const parent_key = MakeStateKey(state);
    auto usages = BuildSubgraphUsages(problem, state);

    auto AddOrUpdateCandidate =
        [&](GreedyMoveType type, Solution solution, int64_t potential_score,
            const std::vector<size_t>& producer_subgraph_ops,
            const std::vector<size_t>& consumer_subgraph_ops MLSYS_FUSER_LOG_ARGS(
                size_t producer_sg_idx, size_t consumer_sg_idx,
                const std::vector<size_t>& touched_tensors)) {
            if (potential_score <= 0) {
                return;
            }
            std::string const key = MakeStateKey(solution);
            if (key == parent_key) {
                return;
            }

            auto existing = candidate_idx_by_key.find(key);
            if (existing == candidate_idx_by_key.end()) {
                candidate_idx_by_key[key] = candidates.size();
                candidates.push_back(GreedyCandidate {
                    .type = type, .solution = std::move(solution),
                    .potential_score = potential_score, .key = key,
                    .producer_subgraph_ops = producer_subgraph_ops,
                    .consumer_subgraph_ops = consumer_subgraph_ops,
#if MLSYS_ENABLE_FUSER_LOGGING
                    .producer_sg_idx = producer_sg_idx, .consumer_sg_idx = consumer_sg_idx,
                    .touched_tensors = touched_tensors,
#endif
                });
                return;
            }

            GreedyCandidate& kept = candidates[existing->second];
            if (potential_score > kept.potential_score) {
                kept.type = type;
                kept.solution = std::move(solution);
                kept.potential_score = potential_score;
                kept.key = key;
                kept.producer_subgraph_ops = producer_subgraph_ops;
                kept.consumer_subgraph_ops = consumer_subgraph_ops;
#if MLSYS_ENABLE_FUSER_LOGGING
                kept.producer_sg_idx = producer_sg_idx;
                kept.consumer_sg_idx = consumer_sg_idx;
                kept.touched_tensors = touched_tensors;
#endif
            }
        };

    auto AverageScoreByTensorCount = [](int64_t raw_score, size_t tensor_count) -> int64_t {
        if (raw_score <= 0 || tensor_count == 0) {
            return 0;
        }
        return raw_score / static_cast<int64_t>(tensor_count);
    };

    for (size_t producer_sg_idx = 0; producer_sg_idx < state.subgraphs.size(); ++producer_sg_idx) {
        for (size_t consumer_sg_idx = producer_sg_idx + 1; consumer_sg_idx < state.subgraphs.size();
             ++consumer_sg_idx) {
            std::vector<size_t> const shared_tensors =
                SharedProducerConsumerTensors(usages[producer_sg_idx], usages[consumer_sg_idx]);
            if (shared_tensors.empty()) {
                continue;
            }

            size_t const shared_tensor_count = shared_tensors.size();

            bool consumer_is_producer = false;
            for (size_t downstream_sg_idx = consumer_sg_idx + 1;
                 downstream_sg_idx < state.subgraphs.size(); ++downstream_sg_idx) {
                if (!SharedProducerConsumerTensors(usages[consumer_sg_idx],
                                                   usages[downstream_sg_idx])
                         .empty()) {
                    consumer_is_producer = true;
                    break;
                }
            }
            int64_t const potential_saving_multiplier = consumer_is_producer ? 2 : 1;

            Solution direct_merge =
                BuildDirectMergeSolution(state, producer_sg_idx, consumer_sg_idx, topo_rank);
            if (NormalizeAndValidateSolution(problem, topo_rank, producer_op, direct_merge)) {
                int64_t const direct_raw_score = potential_saving_multiplier *
                                                 ComputeNewlyInternalizedBoundaryScore(
                                                     problem, producer_op, state, producer_sg_idx,
                                                     consumer_sg_idx, shared_tensors, direct_merge);
                int64_t const direct_score =
                    AverageScoreByTensorCount(direct_raw_score, shared_tensor_count);
                AddOrUpdateCandidate(GreedyMoveType::kDirectFuse, std::move(direct_merge),
                                     direct_score, state.subgraphs[producer_sg_idx].ops,
                                     state.subgraphs[consumer_sg_idx].ops MLSYS_FUSER_LOG_ARGS(
                                         producer_sg_idx, consumer_sg_idx, shared_tensors));
            }

            Solution clone_fuse =
                BuildCloneFuseSolution(state, producer_sg_idx, consumer_sg_idx, topo_rank);
            if (NormalizeAndValidateSolution(problem, topo_rank, producer_op, clone_fuse)) {
                int64_t const clone_raw_score =
                    potential_saving_multiplier * ComputeNewlyInternalizedBoundaryScore(
                                                      problem, producer_op, state, producer_sg_idx,
                                                      consumer_sg_idx, shared_tensors, clone_fuse);
                int64_t const clone_score =
                    AverageScoreByTensorCount(clone_raw_score, shared_tensor_count);
                AddOrUpdateCandidate(GreedyMoveType::kCloneFuse, std::move(clone_fuse), clone_score,
                                     state.subgraphs[producer_sg_idx].ops,
                                     state.subgraphs[consumer_sg_idx].ops MLSYS_FUSER_LOG_ARGS(
                                         producer_sg_idx, consumer_sg_idx, shared_tensors));
            }

            int64_t retain_total = 0;
            for (size_t tensor_idx : shared_tensors) {
                int64_t const tensor_score = TensorSize(problem, tensor_idx);
                if (tensor_score <= 0) {
                    continue;
                }
                retain_total += tensor_score;
            }

            if (retain_total > 0) {
                Solution retain_candidate =
                    BuildRetainSolution(state, producer_sg_idx, consumer_sg_idx, shared_tensors);
                if (NormalizeAndValidateSolution(problem, topo_rank, producer_op,
                                                 retain_candidate)) {
                    int64_t const retain_raw_score = potential_saving_multiplier * retain_total;
                    int64_t const retain_score =
                        AverageScoreByTensorCount(retain_raw_score, shared_tensor_count);
                    AddOrUpdateCandidate(GreedyMoveType::kRetain, std::move(retain_candidate),
                                         retain_score, state.subgraphs[producer_sg_idx].ops,
                                         state.subgraphs[consumer_sg_idx].ops MLSYS_FUSER_LOG_ARGS(
                                             producer_sg_idx, consumer_sg_idx, shared_tensors));
                }
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
    -> ExactEvaluation {
    ExactEvaluation evaluation;
    auto tiled = context.tiler.tile(context.problem, candidate.solution);
    if (!tiled.ok()) {
        return evaluation;
    }

    auto estimated = context.cost_model.estimate(tiled.value());
    if (!estimated.ok()) {
        return evaluation;
    }

    evaluation.valid = true;
    evaluation.solution = std::get<0>(estimated.value());
    evaluation.total_latency = std::get<1>(estimated.value());
    return evaluation;
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

auto BuildSubgraphEdgeDegrees(const Problem& problem, const Solution& state)
    -> std::vector<SubgraphEdgeDegree> {
    auto usages = BuildSubgraphUsages(problem, state);
    std::vector<SubgraphEdgeDegree> degrees(state.subgraphs.size());

    for (size_t producer_sg_idx = 0; producer_sg_idx < state.subgraphs.size(); ++producer_sg_idx) {
        for (size_t consumer_sg_idx = producer_sg_idx + 1; consumer_sg_idx < state.subgraphs.size();
             ++consumer_sg_idx) {
            std::vector<size_t> const shared_tensors =
                SharedProducerConsumerTensors(usages[producer_sg_idx], usages[consumer_sg_idx]);
            if (shared_tensors.empty()) {
                continue;
            }

            degrees[producer_sg_idx].outgoing += shared_tensors.size();
            degrees[consumer_sg_idx].incoming += shared_tensors.size();
        }
    }

    return degrees;
}

void PrefuseLinearUniqueChains(const Problem& problem, const std::vector<int>& producer_op,
                               const std::vector<size_t>& topo_rank, Solution& solution) {
    while (true) {
        bool changed = false;
        auto usages = BuildSubgraphUsages(problem, solution);
        auto edge_degrees = BuildSubgraphEdgeDegrees(problem, solution);

        for (size_t producer_sg_idx = 0; producer_sg_idx < solution.subgraphs.size();
             ++producer_sg_idx) {
            if (edge_degrees[producer_sg_idx].outgoing != 1) {
                continue;
            }

            for (size_t consumer_sg_idx = producer_sg_idx + 1;
                 consumer_sg_idx < solution.subgraphs.size(); ++consumer_sg_idx) {
                std::vector<size_t> const shared_tensors =
                    SharedProducerConsumerTensors(usages[producer_sg_idx], usages[consumer_sg_idx]);
                if (shared_tensors.size() != 1) {
                    continue;
                }

                if (edge_degrees[consumer_sg_idx].incoming != 1 ||
                    edge_degrees[consumer_sg_idx].outgoing != 1) {
                    continue;
                }

                Solution merged =
                    BuildDirectMergeSolution(solution, producer_sg_idx, consumer_sg_idx, topo_rank);
                if (!NormalizeAndValidateSolution(problem, topo_rank, producer_op, merged)) {
                    continue;
                }

                solution = std::move(merged);
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

void PrefuseFreeUnaryChains(const Problem& problem, const std::vector<int>& producer_op,
                            const std::vector<std::vector<size_t>>& consumers_by_tensor,
                            const std::vector<size_t>& topo_rank, Solution& solution) {
    auto subgraph_all_unary = [&](const Subgraph& subgraph) -> bool {
        for (size_t op_idx : subgraph.ops) {
            if (problem.ops[op_idx].inputs.size() != 1) {
                return false;
            }
        }
        return true;
    };

    while (true) {
        bool changed = false;
        auto usages = BuildSubgraphUsages(problem, solution);

        for (size_t producer_sg_idx = 0; producer_sg_idx < solution.subgraphs.size();
             ++producer_sg_idx) {
            if (!subgraph_all_unary(solution.subgraphs[producer_sg_idx])) {
                continue;
            }

            for (size_t consumer_sg_idx = producer_sg_idx + 1;
                 consumer_sg_idx < solution.subgraphs.size(); ++consumer_sg_idx) {
                if (!subgraph_all_unary(solution.subgraphs[consumer_sg_idx])) {
                    continue;
                }

                std::vector<size_t> const shared_tensors =
                    SharedProducerConsumerTensors(usages[producer_sg_idx], usages[consumer_sg_idx]);
                if (shared_tensors.size() != 1) {
                    continue;
                }

                size_t const boundary_tensor = shared_tensors[0];
                if (consumers_by_tensor[boundary_tensor].size() != 1) {
                    continue;
                }

                Solution merged =
                    BuildDirectMergeSolution(solution, producer_sg_idx, consumer_sg_idx, topo_rank);
                if (!NormalizeAndValidateSolution(problem, topo_rank, producer_op, merged)) {
                    continue;
                }

                solution = std::move(merged);
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

auto ComputeCandidateOpHitSum(const GreedyCandidate& candidate,
                              const std::unordered_map<size_t, double>& topk_hits_by_op) -> double {
    double op_hit_sum = 0.0;
    auto add_hit_sum_for_ops = [&](const std::vector<size_t>& ops) {
        for (size_t op_idx : ops) {
            auto hit_it = topk_hits_by_op.find(op_idx);
            if (hit_it == topk_hits_by_op.end()) {
                continue;
            }
            op_hit_sum += hit_it->second;
        }
    };
    add_hit_sum_for_ops(candidate.producer_subgraph_ops);
    add_hit_sum_for_ops(candidate.consumer_subgraph_ops);
    return op_hit_sum;
}

auto RankCandidatesByOpHitPenalty(const std::vector<GreedyCandidate>& candidates,
                                  const std::unordered_map<size_t, double>& topk_hits_by_op,
                                  double topk_failure_penalty)
    -> std::vector<RankedGreedyCandidate> {
    (void)topk_hits_by_op;
    (void)topk_failure_penalty;
    std::vector<RankedGreedyCandidate> ranked_candidates;
    ranked_candidates.reserve(candidates.size());
    for (const GreedyCandidate& candidate : candidates) {
        // Temporarily disable hit-rate penalty; rank by raw potential score.
        double const op_hit_sum = 0.0;
        double const penalized_score = static_cast<double>(candidate.potential_score);
        ranked_candidates.push_back(RankedGreedyCandidate{
            .candidate = &candidate, .penalized_score = penalized_score, .op_hit_sum = op_hit_sum});
    }

    std::sort(ranked_candidates.begin(), ranked_candidates.end(),
              [](const RankedGreedyCandidate& lhs, const RankedGreedyCandidate& rhs) {
                  if (lhs.penalized_score != rhs.penalized_score) {
                      return lhs.penalized_score > rhs.penalized_score;
                  }
                  return lhs.candidate->key < rhs.candidate->key;
              });
    return ranked_candidates;
}

auto TopScoringFusions(SearchContext& context, const Solution& state)
    -> const std::vector<GreedyCandidate>& {
    std::string const state_key = MakeStateKey(state);
    auto cache_it = context.scored_candidates_cache.find(state_key);
    if (cache_it != context.scored_candidates_cache.end()) {
        return cache_it->second;
    }

    std::vector<GreedyCandidate> generated = GenerateGreedyCandidatesAverage(
        context.problem, context.producer_op, context.topo.rank, state);
    auto [inserted_it, _] =
        context.scored_candidates_cache.emplace(state_key, std::move(generated));
    return inserted_it->second;
}

void RunGreedyLookaheadSearch(const GreedyFuserConfig& config, SearchContext& context,
                              const Solution& initial_state, double initial_cost) {
    std::stack<GreedySearchFrame> fusion_stack;
    fusion_stack.push(GreedySearchFrame{
        .state = initial_state,
        .current_cost = initial_cost,
        .base_cost = initial_cost,
        .remaining_lookahead = 0,
        .depth = 0,
    });

    while (!fusion_stack.empty()) {
        GreedySearchFrame frame = std::move(fusion_stack.top());
        fusion_stack.pop();

        if (frame.depth >= config.search_depth) {
            continue;
        }

        const std::vector<GreedyCandidate>& candidates = TopScoringFusions(context, frame.state);
        std::vector<RankedGreedyCandidate> ranked_candidates = RankCandidatesByOpHitPenalty(
            candidates, context.topk_hits_by_op, config.topk_failure_penalty);
#if MLSYS_ENABLE_FUSER_LOGGING
        LogTopKCandidates(ranked_candidates, frame.depth, config.beam_width);
#endif
        std::vector<GreedySearchFrame> next_frames;
        next_frames.reserve(ranked_candidates.size());
        size_t valid_evaluated = 0;
        std::vector<EvaluatedCandidateResult> evaluated_candidates;
        evaluated_candidates.reserve(ranked_candidates.size());

        for (const RankedGreedyCandidate& ranked : ranked_candidates) {
            if (valid_evaluated >= static_cast<size_t>(config.beam_width)) {
                break;
            }
            const GreedyCandidate& candidate = *ranked.candidate;
            ExactEvaluation evaluation = EvaluateCandidateState(context, candidate);
            bool candidate_improved = false;
            if (!evaluation.valid) {
                evaluated_candidates.push_back(
                    EvaluatedCandidateResult{.candidate = &candidate, .improved = false});
                continue;
            }
            ++valid_evaluated;
            MaybeUpdateBest(context, evaluation);
            if (evaluation.total_latency < frame.current_cost) {
                candidate_improved = true;
            }
            evaluated_candidates.push_back(
                EvaluatedCandidateResult{.candidate = &candidate, .improved = candidate_improved});

            GreedySearchFrame next_frame{
                .state = candidate.solution,
                .current_cost = evaluation.total_latency,
                .base_cost = frame.base_cost,
                .remaining_lookahead = frame.remaining_lookahead,
                .depth = frame.depth + 1,
            };

            if (frame.remaining_lookahead == 0) {
                if (next_frame.current_cost < frame.current_cost) {
                    next_frame.base_cost = next_frame.current_cost;
                    next_frame.remaining_lookahead = 0;
                } else {
                    next_frame.base_cost = frame.current_cost;
                    next_frame.remaining_lookahead = 2;
                }
                next_frames.push_back(std::move(next_frame));
                continue;
            }

            if (next_frame.current_cost < frame.base_cost) {
                next_frame.base_cost = next_frame.current_cost;
                next_frame.remaining_lookahead = 0;
                next_frames.push_back(std::move(next_frame));
                continue;
            }

            if (frame.remaining_lookahead > 1) {
                next_frame.base_cost = frame.base_cost;
                next_frame.remaining_lookahead = frame.remaining_lookahead - 1;
                next_frames.push_back(std::move(next_frame));
            }
        }

        (void)evaluated_candidates;

        for (auto next_it = next_frames.rbegin(); next_it != next_frames.rend(); ++next_it) {
            fusion_stack.push(std::move(*next_it));
        }
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

    PrefuseFreeUnaryChains(problem, producer_op, consumers_by_tensor, topo.rank, initial_solution);

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

#undef MLSYS_FUSER_LOG_ARGS

} // namespace

GreedyFuser::GreedyFuser(GreedyFuserConfig config) : config_(config) {}

auto GreedyFuser::fuse(const Problem& problem) -> StatusOr<Solution> {
    if (config_.search_depth < 0) {
        return absl::InvalidArgumentError("search_depth must be non-negative");
    }
    if (config_.beam_width <= 0) {
        return absl::InvalidArgumentError("beam_width must be positive");
    }
    if (config_.topk_failure_penalty < 0.0) {
        return absl::InvalidArgumentError("topk_failure_penalty must be non-negative");
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
        .type = GreedyMoveType::kDirectFuse,
        .solution = initial_solution,
        .potential_score = 0,
        .key = MakeStateKey(initial_solution),
    };
    const ExactEvaluation& root_evaluation = EvaluateCandidateState(context, root_candidate);
    MaybeUpdateBest(context, root_evaluation);

    double const initial_cost = root_evaluation.valid ? root_evaluation.total_latency
                                                      : std::numeric_limits<double>::infinity();
    RunGreedyLookaheadSearch(config_, context, initial_solution, initial_cost);

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
