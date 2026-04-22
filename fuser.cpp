#include "fuser.h"
#include "fuser_logging.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
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
    if (!IsFuserLoggingEnabled()) {
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

struct FrameCandidateEvaluationLog {
    std::string explore_id;
    size_t rank = 0;
    GreedyMoveType move = GreedyMoveType::kDirectFuse;
    size_t producer_sg_idx = 0;
    size_t consumer_sg_idx = 0;
    bool valid = false;
    double evaluated_cost = std::numeric_limits<double>::infinity();
    double baseline_cost = std::numeric_limits<double>::infinity();
    double delta_vs_baseline = std::numeric_limits<double>::infinity();
};

struct BestUpdateEvent {
    int depth = 0;
    size_t frame_seq = 0;
    std::string explore_id;
    double selected_latency = std::numeric_limits<double>::infinity();
    Solution selected_solution;
};

struct EvaluatedCandidateResult {
    const GreedyCandidate* candidate = nullptr;
    bool improved = false;
};

struct SearchContext {
    const Problem& problem;
    const std::vector<int>& producer_op;
    const TopologyInfo& topo;
    Tiler* tiler;
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
    Solution evaluated_state;
    bool has_evaluated_state = false;
    double current_cost = std::numeric_limits<double>::infinity();
    double base_cost = std::numeric_limits<double>::infinity();
    int remaining_lookahead = 0;
    int depth = 0;
};

auto FormatIndexListForLog(const std::vector<size_t>& values) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx > 0) {
            out << ",";
        }
        out << values[idx];
    }
    out << "]";
    return out.str();
}

auto FormatSubgraphOpsListForLog(const Solution& solution) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        if (sg_idx > 0) {
            out << ",";
        }
        out << FormatIndexListForLog(solution.subgraphs[sg_idx].ops);
    }
    out << "]";
    return out.str();
}

auto FormatSubgraphRetainsListForLog(const Solution& solution) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        if (sg_idx > 0) {
            out << ",";
        }
        out << FormatIndexListForLog(solution.subgraphs[sg_idx].tensors_to_retain);
    }
    out << "]";
    return out.str();
}

auto FormatInt64ListForLog(const std::vector<int64_t>& values) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx > 0) {
            out << ",";
        }
        out << values[idx];
    }
    out << "]";
    return out.str();
}

auto FormatTraversalOrderForLog(const std::optional<TraversalOrder>& traversal_order)
    -> std::string {
    if (!traversal_order.has_value()) {
        return "null";
    }
    return FormatInt64ListForLog(*traversal_order);
}

auto FormatGranularityForLog(const Granularity& granularity) -> std::string {
    std::ostringstream out;
    out << "(" << granularity.width << "x" << granularity.height << "x" << granularity.depth << ")";
    return out.str();
}

auto FormatSubgraphGranularityListForLog(const Solution& solution) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        if (sg_idx > 0) {
            out << ",";
        }
        out << FormatGranularityForLog(solution.subgraphs[sg_idx].granularity);
    }
    out << "]";
    return out.str();
}

auto FormatSubgraphTraversalOrdersListForLog(const Solution& solution) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        if (sg_idx > 0) {
            out << ",";
        }
        out << FormatTraversalOrderForLog(solution.subgraphs[sg_idx].traversal_order);
    }
    out << "]";
    return out.str();
}

auto FormatSubgraphLatencyListForLog(const Solution& solution) -> std::string {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "[";
    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        if (sg_idx > 0) {
            out << ",";
        }
        out << solution.subgraphs[sg_idx].subgraph_latency;
    }
    out << "]";
    return out.str();
}

void LogSolutionArraysForLog(const std::string& label, const Solution& solution) {
    std::ostringstream subgraphs_line;
    subgraphs_line << label << "_subgraphs=" << FormatSubgraphOpsListForLog(solution);
    LogFuserDebugLine(subgraphs_line.str());

    std::ostringstream retain_line;
    retain_line << label << "_tensors_to_retain=" << FormatSubgraphRetainsListForLog(solution);
    LogFuserDebugLine(retain_line.str());

    std::ostringstream granularity_line;
    granularity_line << label << "_granularity=" << FormatSubgraphGranularityListForLog(solution);
    LogFuserDebugLine(granularity_line.str());

    std::ostringstream traversal_line;
    traversal_line << label
                   << "_traversal_order=" << FormatSubgraphTraversalOrdersListForLog(solution);
    LogFuserDebugLine(traversal_line.str());

    std::ostringstream latency_line;
    latency_line << label << "_subgraph_latency=" << FormatSubgraphLatencyListForLog(solution);
    LogFuserDebugLine(latency_line.str());
}

auto GetBaselineSolutionForFrame(const GreedySearchFrame& frame) -> const Solution& {
    if (frame.has_evaluated_state) {
        return frame.evaluated_state;
    }
    return frame.state;
}

auto BuildExploreId(int depth, size_t frame_seq, size_t rank_idx) -> std::string {
    std::ostringstream out;
    out << "exp_d" << depth << "_f" << frame_seq << "_r" << rank_idx;
    return out.str();
}

auto SubgraphContainsAnyOp(const Subgraph& subgraph, const std::unordered_set<size_t>& impacted_ops)
    -> bool {
    for (size_t op_idx : subgraph.ops) {
        if (impacted_ops.find(op_idx) != impacted_ops.end()) {
            return true;
        }
    }
    return false;
}

auto FormatImpactedSubgraphOutputTileSizesForLog(const GreedyCandidate& candidate,
                                                 const Solution& evaluated_solution)
    -> std::string {
    std::unordered_set<size_t> impacted_ops;
    impacted_ops.insert(candidate.producer_subgraph_ops.begin(),
                        candidate.producer_subgraph_ops.end());
    impacted_ops.insert(candidate.consumer_subgraph_ops.begin(),
                        candidate.consumer_subgraph_ops.end());

    std::ostringstream out;
    out << "[";
    bool first = true;
    for (size_t sg_idx = 0; sg_idx < evaluated_solution.subgraphs.size(); ++sg_idx) {
        const Subgraph& subgraph = evaluated_solution.subgraphs[sg_idx];
        if (!SubgraphContainsAnyOp(subgraph, impacted_ops)) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        first = false;
        out << "{sg=" << sg_idx << ",ops=" << FormatIndexListForLog(subgraph.ops)
            << ",output_tile=(" << subgraph.granularity.width << "x" << subgraph.granularity.height
            << "x" << subgraph.granularity.depth << ")}";
    }
    out << "]";
    return out.str();
}

void LogSearchFrameState(const GreedySearchFrame& frame) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    const Solution& baseline_solution = GetBaselineSolutionForFrame(frame);
    std::ostringstream line;
    line << std::setprecision(17);
    line << "[SearchFrameState] depth=" << frame.depth << ", current_cost=" << frame.current_cost
         << ", base_cost=" << frame.base_cost
         << ", remaining_lookahead=" << frame.remaining_lookahead
         << ", subgraph_ops=" << FormatSubgraphOpsListForLog(baseline_solution);
    LogFuserDebugLine(line.str());
#else
    (void)frame;
#endif
}

void LogCandidateSkippedByBeamLimit(const std::string& explore_id, int depth, size_t rank_idx,
                                    const RankedGreedyCandidate& ranked, size_t beam_limit) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    const GreedyCandidate& candidate = *ranked.candidate;
    std::ostringstream line;
    line << std::setprecision(17);
    line << "[BeamCandidateResult] explore_id=" << explore_id << ", depth=" << depth
         << ", rank=" << rank_idx << ", move=" << MoveTypeName(candidate.type) << ", pair=("
         << candidate.producer_sg_idx << "->" << candidate.consumer_sg_idx
         << "), result=not_evaluated_beam_limit_reached, beam_width=" << beam_limit;
    LogFuserDebugLine(line.str());
#else
    (void)explore_id;
    (void)depth;
    (void)rank_idx;
    (void)ranked;
    (void)beam_limit;
#endif
}

void LogCandidateEvaluationResult(const std::string& explore_id, int depth, size_t rank_idx,
                                  const RankedGreedyCandidate& ranked,
                                  const ExactEvaluation& evaluation, double current_cost,
                                  double base_cost) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    const GreedyCandidate& candidate = *ranked.candidate;
    std::ostringstream line;
    line << "[BeamCandidateResult] explore_id=" << explore_id << ", depth=" << depth
         << ", rank=" << rank_idx << ", move=" << MoveTypeName(candidate.type) << ", pair=("
         << candidate.producer_sg_idx << "->" << candidate.consumer_sg_idx << ")";

    if (!evaluation.valid) {
        line << ", result=invalid_cost_model_estimate";
        LogFuserDebugLine(line.str());
        return;
    }

    double const delta_vs_current = evaluation.total_latency - current_cost;
    double const delta_vs_base = evaluation.total_latency - base_cost;
    if (delta_vs_current < 0.0) {
        line << ", result=better, better_by=" << std::abs(delta_vs_current);
    } else if (delta_vs_current > 0.0) {
        line << ", result=worse, worse_by=" << delta_vs_current;
    } else {
        line << ", result=equal, change_by=0";
    }
    line << ", cost_model_total_latency=" << evaluation.total_latency
         << ", delta_vs_current=" << delta_vs_current << ", delta_vs_base=" << delta_vs_base;
    LogFuserDebugLine(line.str());
#else
    (void)explore_id;
    (void)depth;
    (void)rank_idx;
    (void)ranked;
    (void)evaluation;
    (void)current_cost;
    (void)base_cost;
#endif
}

void LogCandidatePruned(const std::string& explore_id, int depth, size_t rank_idx,
                        const RankedGreedyCandidate& ranked, const char* reason) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    const GreedyCandidate& candidate = *ranked.candidate;
    std::ostringstream line;
    line << "[BeamCandidatePruned] explore_id=" << explore_id << ", depth=" << depth
         << ", rank=" << rank_idx << ", move=" << MoveTypeName(candidate.type) << ", pair=("
         << candidate.producer_sg_idx << "->" << candidate.consumer_sg_idx
         << "), reason=" << reason;
    LogFuserDebugLine(line.str());
#else
    (void)explore_id;
    (void)depth;
    (void)rank_idx;
    (void)ranked;
    (void)reason;
#endif
}

void LogExplorativeFusion(const std::string& explore_id, const GreedySearchFrame& parent_frame,
                          size_t rank_idx, const RankedGreedyCandidate& ranked,
                          const GreedySearchFrame& next_frame, const Solution& evaluated_solution) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    const GreedyCandidate& candidate = *ranked.candidate;
    double const selected_minus_parent_cost = next_frame.current_cost - parent_frame.current_cost;
    bool const improved = (selected_minus_parent_cost < 0.0);
    const Solution& baseline_solution = GetBaselineSolutionForFrame(parent_frame);

    LogFuserDebugLine("===EXPLORATION_GROUP=============================================");

    std::ostringstream section1_header;
    section1_header << std::setprecision(17);
    section1_header << "[EXPLORATION_SECTION_1_BASELINE] explore_id=" << explore_id
                    << ", depth=" << parent_frame.depth << ", rank=" << rank_idx
                    << ", baseline_score=" << parent_frame.current_cost;
    LogFuserDebugLine(section1_header.str());

    LogSolutionArraysForLog("baseline_solution", baseline_solution);

    std::ostringstream section2_header;
    section2_header << std::setprecision(17);
    section2_header << "[EXPLORATION_SECTION_2_EXPLORED] explore_id=" << explore_id
                    << ", depth=" << parent_frame.depth << ", rank=" << rank_idx
                    << ", move=" << MoveTypeName(candidate.type) << ", pair=("
                    << candidate.producer_sg_idx << "->" << candidate.consumer_sg_idx
                    << "), explored_score=" << next_frame.current_cost
                    << ", next_remaining_lookahead=" << next_frame.remaining_lookahead;
    LogFuserDebugLine(section2_header.str());

    LogSolutionArraysForLog("explored_solution", evaluated_solution);

    std::ostringstream pair_ops_line;
    pair_ops_line << "explored_pair_ops=[" << FormatIndexListForLog(candidate.producer_subgraph_ops)
                  << "," << FormatIndexListForLog(candidate.consumer_subgraph_ops) << "]";
    LogFuserDebugLine(pair_ops_line.str());

    if (candidate.type == GreedyMoveType::kRetain || candidate.type == GreedyMoveType::kCloneFuse) {
        std::ostringstream impacted_line;
        impacted_line << "impacted_subgraph_output_tile_sizes="
                      << FormatImpactedSubgraphOutputTileSizesForLog(candidate, evaluated_solution);
        LogFuserDebugLine(impacted_line.str());
    }

    std::ostringstream section3_line;
    section3_line << std::setprecision(17);
    section3_line << "[EXPLORATION_SECTION_3_DELTA] explore_id=" << explore_id
                  << ", baseline_score=" << parent_frame.current_cost
                  << ", explored_score=" << next_frame.current_cost
                  << ", delta_cost=" << selected_minus_parent_cost << ", improved=" << improved;
    LogFuserDebugLine(section3_line.str());
    LogFuserDebugLine("");
#else
    (void)explore_id;
    (void)parent_frame;
    (void)rank_idx;
    (void)ranked;
    (void)next_frame;
    (void)evaluated_solution;
#endif
}

void LogSelectedSolution(const BestUpdateEvent& event,
                         const std::vector<FrameCandidateEvaluationLog>& frame_candidate_logs) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    LogFuserDebugLine("===SELECTED_SOLUTION=============================================");

    std::ostringstream header;
    header << std::setprecision(17);
    header << "[SELECTED_SOLUTION] depth=" << event.depth << ", frame_seq=" << event.frame_seq
           << ", selected_explore_id=" << event.explore_id
           << ", selected_latency_cost=" << event.selected_latency;
    LogFuserDebugLine(header.str());

    LogSolutionArraysForLog("selected_solution", event.selected_solution);

    std::vector<const FrameCandidateEvaluationLog*> valid_logs;
    valid_logs.reserve(frame_candidate_logs.size());
    for (const FrameCandidateEvaluationLog& record : frame_candidate_logs) {
        if (!record.valid) {
            continue;
        }
        valid_logs.push_back(&record);
    }

    std::sort(valid_logs.begin(), valid_logs.end(),
              [](const FrameCandidateEvaluationLog* lhs, const FrameCandidateEvaluationLog* rhs) {
                  if (lhs->evaluated_cost != rhs->evaluated_cost) {
                      return lhs->evaluated_cost < rhs->evaluated_cost;
                  }
                  return lhs->rank < rhs->rank;
              });

    size_t const limit = std::min<size_t>(5, valid_logs.size());
    std::ostringstream top_header;
    top_header << "[SELECTED_SOLUTION_TOP_CANDIDATES] count=" << limit;
    LogFuserDebugLine(top_header.str());

    for (size_t idx = 0; idx < limit; ++idx) {
        const FrameCandidateEvaluationLog& record = *valid_logs[idx];
        std::ostringstream line;
        line << std::setprecision(17);
        line << "top_idx=" << idx << ", explore_id=" << record.explore_id
             << ", rank=" << record.rank << ", move=" << MoveTypeName(record.move) << ", pair=("
             << record.producer_sg_idx << "->" << record.consumer_sg_idx
             << "), evaluated_cost=" << record.evaluated_cost
             << ", baseline_cost=" << record.baseline_cost
             << ", delta_vs_baseline=" << record.delta_vs_baseline;
        LogFuserDebugLine(line.str());
    }
    LogFuserDebugLine("");
#else
    (void)event;
    (void)frame_candidate_logs;
#endif
}

void LogFinalBestSelection(const SearchContext& context) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    if (!context.has_best) {
        LogFuserDebugLine("[SEARCH_FINAL_BEST] has_best=0");
        return;
    }

    LogFuserDebugLine("===SEARCH_FINAL_BEST=============================================");

    std::ostringstream line;
    line << std::setprecision(17);
    line << "[SEARCH_FINAL_BEST] has_best=1"
         << ", final_latency_cost=" << context.best_cost;
    LogFuserDebugLine(line.str());
    LogSolutionArraysForLog("final_best", context.best_solution);
    LogFuserDebugLine("");
#else
    (void)context;
#endif
}

auto BuildProducerOpIndex(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer_op(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        for (size_t output_tensor : problem.ops[op_idx].outputs) {
            producer_op[output_tensor] = static_cast<int>(op_idx);
        }
    }
    return producer_op;
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

void PrefuseProducerIntoUnaryPointwiseConsumers(const Problem& problem,
                                                const std::vector<int>& producer_op,
                                                const std::vector<size_t>& topo_rank,
                                                Solution& solution) {
    while (true) {
        bool changed = false;
        for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
            const auto& current_subgraph = solution.subgraphs[sg_idx];
            for (size_t pointwise_op_idx : current_subgraph.ops) {
                const Op& pointwise_op = problem.ops[pointwise_op_idx];
                if (pointwise_op.op_type != "Pointwise" || pointwise_op.inputs.size() != 1) {
                    continue;
                }

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
    auto tiled = context.tiler->tile(context.problem, candidate.solution);
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

auto MaybeUpdateBest(SearchContext& context, const ExactEvaluation& evaluation) -> bool {
    if (!evaluation.valid) {
        return false;
    }
    if (!context.has_best || evaluation.total_latency < context.best_cost) {
        context.has_best = true;
        context.best_cost = evaluation.total_latency;
        context.best_solution = evaluation.solution;
        return true;
    }
    return false;
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
    std::vector<RankedGreedyCandidate> ranked_candidates;
    ranked_candidates.reserve(candidates.size());
    for (const GreedyCandidate& candidate : candidates) {
        double const op_hit_sum = ComputeCandidateOpHitSum(candidate, topk_hits_by_op);
        // Linear penalty: score / (1 + a * hit_sum), where a = topk_failure_penalty.
        double const denom = 1.0 + (topk_failure_penalty * op_hit_sum);
        double const penalized_score =
            static_cast<double>(candidate.potential_score) / std::max(denom, 1.0);
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
                              const Solution& initial_state,
                              const Solution& initial_evaluated_state,
                              bool has_initial_evaluated_state, double initial_cost) {
    std::stack<GreedySearchFrame> fusion_stack;
    fusion_stack.push(GreedySearchFrame{
        .state = initial_state,
        .evaluated_state = initial_evaluated_state,
        .has_evaluated_state = has_initial_evaluated_state,
        .current_cost = initial_cost,
        .base_cost = initial_cost,
        .remaining_lookahead = 0,
        .depth = 0,
    });
    size_t frame_seq = 0;

    while (!fusion_stack.empty()) {
        size_t const current_frame_seq = frame_seq++;
        GreedySearchFrame frame = std::move(fusion_stack.top());
        fusion_stack.pop();

        if (frame.depth >= config.search_depth) {
            continue;
        }
#if MLSYS_ENABLE_FUSER_LOGGING
        LogSearchFrameState(frame);
#endif

        const std::vector<GreedyCandidate>& candidates = TopScoringFusions(context, frame.state);
        std::vector<RankedGreedyCandidate> ranked_candidates = RankCandidatesByOpHitPenalty(
            candidates, context.topk_hits_by_op, config.topk_failure_penalty);
        size_t const beam_limit =
            config.beam_width <= 0 ? 0 : static_cast<size_t>(config.beam_width);
#if MLSYS_ENABLE_FUSER_LOGGING
        LogTopKCandidates(ranked_candidates, frame.depth, config.beam_width);
#endif
        std::vector<GreedySearchFrame> next_frames;
        next_frames.reserve(ranked_candidates.size());
        size_t valid_evaluated = 0;
        std::vector<EvaluatedCandidateResult> evaluated_candidates;
        evaluated_candidates.reserve(ranked_candidates.size());
        std::vector<FrameCandidateEvaluationLog> frame_candidate_logs;
        frame_candidate_logs.reserve(ranked_candidates.size());
        std::optional<BestUpdateEvent> best_update_event;

        for (size_t ranked_idx = 0; ranked_idx < ranked_candidates.size(); ++ranked_idx) {
            const RankedGreedyCandidate& ranked = ranked_candidates[ranked_idx];
            std::string const explore_id =
                BuildExploreId(frame.depth, current_frame_seq, ranked_idx);
            if (valid_evaluated >= beam_limit) {
#if MLSYS_ENABLE_FUSER_LOGGING
                LogCandidateSkippedByBeamLimit(explore_id, frame.depth, ranked_idx, ranked,
                                               beam_limit);
#endif
                continue;
            }
            const GreedyCandidate& candidate = *ranked.candidate;
            ExactEvaluation evaluation = EvaluateCandidateState(context, candidate);
#if MLSYS_ENABLE_FUSER_LOGGING
            LogCandidateEvaluationResult(explore_id, frame.depth, ranked_idx, ranked, evaluation,
                                         frame.current_cost, frame.base_cost);
#endif
            FrameCandidateEvaluationLog frame_log_record;
            frame_log_record.explore_id = explore_id;
            frame_log_record.rank = ranked_idx;
            frame_log_record.move = candidate.type;
            frame_log_record.valid = evaluation.valid;
            frame_log_record.evaluated_cost = evaluation.total_latency;
            frame_log_record.baseline_cost = frame.current_cost;
            frame_log_record.delta_vs_baseline = evaluation.total_latency - frame.current_cost;
#if MLSYS_ENABLE_FUSER_LOGGING
            frame_log_record.producer_sg_idx = candidate.producer_sg_idx;
            frame_log_record.consumer_sg_idx = candidate.consumer_sg_idx;
#endif
            frame_candidate_logs.push_back(frame_log_record);
            bool candidate_improved = false;
            if (!evaluation.valid) {
                evaluated_candidates.push_back(
                    EvaluatedCandidateResult{.candidate = &candidate, .improved = false});
                continue;
            }
            ++valid_evaluated;
            bool const best_updated = MaybeUpdateBest(context, evaluation);
            if (best_updated) {
                best_update_event = BestUpdateEvent{
                    .depth = frame.depth,
                    .frame_seq = current_frame_seq,
                    .explore_id = explore_id,
                    .selected_latency = evaluation.total_latency,
                    .selected_solution = evaluation.solution,
                };
            }
            if (evaluation.total_latency < frame.current_cost) {
                candidate_improved = true;
            }
            evaluated_candidates.push_back(
                EvaluatedCandidateResult{.candidate = &candidate, .improved = candidate_improved});

            GreedySearchFrame next_frame{
                .state = candidate.solution,
                .evaluated_state = evaluation.solution,
                .has_evaluated_state = true,
                .current_cost = evaluation.total_latency,
                .base_cost = frame.base_cost,
                .remaining_lookahead = frame.remaining_lookahead,
                .depth = frame.depth + 1,
            };

            bool selected_for_expansion = false;
            if (frame.remaining_lookahead == 0) {
                if (next_frame.current_cost < frame.current_cost) {
                    next_frame.base_cost = next_frame.current_cost;
                    next_frame.remaining_lookahead = 0;
                } else {
                    next_frame.base_cost = frame.current_cost;
                    next_frame.remaining_lookahead = 2;
                }
                selected_for_expansion = true;
            } else if (next_frame.current_cost < frame.base_cost) {
                next_frame.base_cost = next_frame.current_cost;
                next_frame.remaining_lookahead = 0;
                selected_for_expansion = true;
            } else if (frame.remaining_lookahead > 1) {
                next_frame.base_cost = frame.base_cost;
                next_frame.remaining_lookahead = frame.remaining_lookahead - 1;
                selected_for_expansion = true;
            }

            if (selected_for_expansion) {
#if MLSYS_ENABLE_FUSER_LOGGING
                LogExplorativeFusion(explore_id, frame, ranked_idx, ranked, next_frame,
                                     evaluation.solution);
#endif
                next_frames.push_back(std::move(next_frame));
            } else {
#if MLSYS_ENABLE_FUSER_LOGGING
                LogCandidatePruned(explore_id, frame.depth, ranked_idx, ranked,
                                   "lookahead_exhausted_without_improvement");
#endif
            }
        }

#if MLSYS_ENABLE_FUSER_LOGGING
        if (best_update_event.has_value()) {
            LogSelectedSolution(best_update_event.value(), frame_candidate_logs);
        }
#endif

        for (const EvaluatedCandidateResult& evaluated_candidate : evaluated_candidates) {
            if (evaluated_candidate.improved || evaluated_candidate.candidate == nullptr) {
                continue;
            }
            auto add_subgraph_hit_mass = [&](const std::vector<size_t>& subgraph_ops) {
                if (subgraph_ops.empty()) {
                    return;
                }
                double const per_op_ratio = 1.0 / static_cast<double>(subgraph_ops.size());
                for (size_t op_idx : subgraph_ops) {
                    double& op_hit = context.topk_hits_by_op[op_idx];
                    if (op_hit <= 0.0) {
                        // Bootstrap first touch so multiplicative updates are not stuck at zero.
                        op_hit = per_op_ratio;
                    } else {
                        op_hit += op_hit * per_op_ratio;
                    }
                }
            };
            add_subgraph_hit_mass(evaluated_candidate.candidate->producer_subgraph_ops);
            add_subgraph_hit_mass(evaluated_candidate.candidate->consumer_subgraph_ops);
        }

        for (auto next_it = next_frames.rbegin(); next_it != next_frames.rend(); ++next_it) {
            fusion_stack.push(std::move(*next_it));
        }
    }

#if MLSYS_ENABLE_FUSER_LOGGING
    LogFinalBestSelection(context);
#endif
}
auto BuildInitialGreedySolution(const Problem& problem, const TopologyInfo& topo,
                                const std::vector<int>& producer_op) -> StatusOr<Solution> {
    Solution initial_solution;
    for (size_t op_idx : topo.order) {
        Subgraph subgraph;
        subgraph.ops = {op_idx};
        subgraph.tensors_to_retain = {};
        subgraph.traversal_order = std::nullopt;
        subgraph.subgraph_latency = 0.0;
        initial_solution.subgraphs.push_back(subgraph);
    }

    PrefuseProducerIntoUnaryPointwiseConsumers(problem, producer_op, topo.rank, initial_solution);

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

GreedyFuser::GreedyFuser(GreedyFuserConfig config)
    : GreedyFuser(config, std::make_unique<CostGuidedDivisorTiler>()) {}

GreedyFuser::GreedyFuser(GreedyFuserConfig config, std::unique_ptr<Tiler> tiler)
    : config_(config), tiler_(std::move(tiler)) {}

GreedyFuser::~GreedyFuser() = default;

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
    if (tiler_ == nullptr) {
        return absl::InvalidArgumentError("tiler must not be null");
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
    auto initial_solution_or = BuildInitialGreedySolution(problem, topo, producer_op);
    if (!initial_solution_or.ok()) {
        return initial_solution_or.status();
    }

    Solution initial_solution = initial_solution_or.value();
    SearchContext context{
        .problem = problem,
        .producer_op = producer_op,
        .topo = topo,
        .tiler = tiler_.get(),
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

    Solution const root_evaluated_solution =
        root_evaluation.valid ? root_evaluation.solution : initial_solution;
    double const initial_cost = root_evaluation.valid ? root_evaluation.total_latency
                                                      : std::numeric_limits<double>::infinity();
    RunGreedyLookaheadSearch(config_, context, initial_solution, root_evaluated_solution,
                             root_evaluation.valid, initial_cost);

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
