#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"

#include "cost_model.h"
#include "fuser.h"
#include "mlsys.h"
#include "solver.h"
#include "tiler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

namespace mlsys {

namespace {

struct IntervalPlan {
    bool valid = false;
    Subgraph subgraph;
    double latency = 0.0;
};

auto MaxFusionWidthForProblem(size_t num_ops) -> size_t {
    if (num_ops <= 8) {
        return 6;
    }
    if (num_ops <= 20) {
        return 5;
    }
    return 2;
}

auto GreedyConfigForProblem(size_t num_ops) -> GreedyFuserConfig {
    constexpr double kTopKFailurePenalty = 0.08;
    if (num_ops <= 4) {
        return GreedyFuserConfig{
            .search_depth = 12, .beam_width = 32, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 8) {
        return GreedyFuserConfig{
            .search_depth = 10, .beam_width = 24, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 16) {
        return GreedyFuserConfig{
            .search_depth = 8, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 32) {
        return GreedyFuserConfig{
            .search_depth = 6, .beam_width = 12, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 64) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 8, .topk_failure_penalty = kTopKFailurePenalty};
    }
    return GreedyFuserConfig{
        .search_depth = 3, .beam_width = 6, .topk_failure_penalty = kTopKFailurePenalty};
}

auto BuildSingletonSolution(const Problem& problem) -> absl::StatusOr<Solution> {
    Solution solution;
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        Subgraph sg;
        sg.ops = {i};
        sg.tensors_to_retain = {};
        sg.traversal_order = nullopt;
        solution.subgraphs.push_back(sg);
    }

    CostGuidedDivisorTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    if (!tiled.ok()) {
        return tiled.status();
    }
    return tiled.value();
}

} // namespace

auto BaseSolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    Solution solution;

    // The baseline solver schedules each operation individually in topological order
    // and delegates tile selection to the greedy tiler.
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        Subgraph sg;
        sg.ops.push_back(i);
        sg.tensors_to_retain = {};
        sg.traversal_order = nullopt;

        solution.subgraphs.push_back(sg);
    }

    unique_ptr<Tiler> tiler = make_unique<CostGuidedDivisorTiler>();
    auto tiled_solution = tiler->tile(problem, solution);
    if (!tiled_solution.ok()) {
        return tiled_solution.status();
    }

    unique_ptr<CostModel> cost_model = make_unique<CostModel>(problem);
    auto estimated = cost_model->estimate(tiled_solution.value());
    if (!estimated.ok()) {
        return estimated.status();
    }

    return get<0>(estimated.value());
}

auto BruteForceSolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    unique_ptr<BruteForceFuser> fuser = make_unique<BruteForceFuser>();
    unique_ptr<Tiler> tiler = make_unique<BruteForceTiler>();
    unique_ptr<CostModel> cost_model = make_unique<CostModel>(problem);

    auto fusion_plans = fuser->fuse(problem);
    if (!fusion_plans.ok()) {
        return fusion_plans.status();
    }

    if (fusion_plans.value().empty()) {
        return absl::InternalError("Fuser returned no fusion plans");
    }

    tuple<Solution, SubgraphLatency> optimal_plan;

    for (const auto& plan : fusion_plans.value()) {
        auto tiled_plan = tiler->tile(problem, plan);
        if (!tiled_plan.ok()) {
            // tile() returns an error when no valid tiling is available
            continue;
        }

        auto solu = cost_model->estimate(tiled_plan.value());

        if (optimal_plan == tuple<Solution, SubgraphLatency>() ||
            get<1>(solu.value()) < get<1>(optimal_plan)) {
            optimal_plan = solu.value();
        }
    }

    if (optimal_plan == tuple<Solution, SubgraphLatency>()) {
        return absl::InternalError("No valid plans found by fuser and tiler");
    }

    return get<0>(optimal_plan);
}

auto HeuristicSolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    size_t const num_ops = problem.ops.size();
    if (num_ops == 0) {
        return Solution{};
    }
    size_t const max_fusion_width = MaxFusionWidthForProblem(num_ops);

    auto singleton_solution_status = BuildSingletonSolution(problem);
    if (!singleton_solution_status.ok()) {
        return singleton_solution_status.status();
    }
    Solution singleton_solution = singleton_solution_status.value();

    CostModel cost_model(problem);
    auto singleton_estimated = cost_model.estimate(singleton_solution);
    if (!singleton_estimated.ok()) {
        return singleton_estimated.status();
    }
    singleton_solution = get<0>(singleton_estimated.value());

    std::vector<std::vector<IntervalPlan>> interval_plans(num_ops,
                                                          std::vector<IntervalPlan>(num_ops));
    CostGuidedDivisorTiler tiler;

    for (size_t start = 0; start < num_ops; ++start) {
        for (size_t end = start; end < num_ops && end < start + max_fusion_width; ++end) {
            Subgraph fused_subgraph;
            fused_subgraph.tensors_to_retain = {};
            fused_subgraph.traversal_order = nullopt;
            for (size_t op_idx = start; op_idx <= end; ++op_idx) {
                fused_subgraph.ops.push_back(op_idx);
            }

            Solution candidate_full;
            for (size_t prefix = 0; prefix < start; ++prefix) {
                candidate_full.subgraphs.push_back(singleton_solution.subgraphs[prefix]);
            }
            candidate_full.subgraphs.push_back(fused_subgraph);
            for (size_t suffix = end + 1; suffix < num_ops; ++suffix) {
                candidate_full.subgraphs.push_back(singleton_solution.subgraphs[suffix]);
            }

            auto tiled_candidate = tiler.tile_subgraph(problem, candidate_full, start);
            if (!tiled_candidate.ok()) {
                continue;
            }
            candidate_full = tiled_candidate.value();

            auto estimated = cost_model.estimate(candidate_full);
            if (!estimated.ok()) {
                continue;
            }

            IntervalPlan plan;
            plan.valid = true;
            plan.subgraph = get<0>(estimated.value()).subgraphs[start];
            plan.latency = plan.subgraph.subgraph_latency;
            interval_plans[start][end] = plan;
        }
    }

    std::vector<double> dp(num_ops + 1, std::numeric_limits<double>::infinity());
    std::vector<size_t> next_end(num_ops, num_ops);
    dp[num_ops] = 0.0;

    for (int start = static_cast<int>(num_ops) - 1; start >= 0; --start) {
        for (size_t end = static_cast<size_t>(start);
             end < num_ops && end < static_cast<size_t>(start) + max_fusion_width; ++end) {
            const auto& plan = interval_plans[static_cast<size_t>(start)][end];
            if (!plan.valid) {
                continue;
            }

            double const candidate = plan.latency + dp[end + 1];
            if (candidate < dp[static_cast<size_t>(start)]) {
                dp[static_cast<size_t>(start)] = candidate;
                next_end[static_cast<size_t>(start)] = end;
            }
        }
    }

    if (!std::isfinite(dp[0])) {
        return absl::InternalError("Heuristic solver found no valid interval partition");
    }

    Solution solution;
    for (size_t start = 0; start < num_ops;) {
        size_t const end = next_end[start];
        if (end >= num_ops) {
            return absl::InternalError("Heuristic solver failed to reconstruct solution");
        }
        solution.subgraphs.push_back(interval_plans[start][end].subgraph);
        start = end + 1;
    }

    auto estimated = cost_model.estimate(solution);
    if (!estimated.ok()) {
        return estimated.status();
    }

    return get<0>(estimated.value());
}

GreedySolver::GreedySolver()
    : use_problem_sized_config_(true), config_{.search_depth = 0,
                                               .beam_width = 0,
                                               .topk_failure_penalty = 0.0} {}

GreedySolver::GreedySolver(GreedyFuserConfig config)
    : use_problem_sized_config_(false), config_(config) {}

auto GreedySolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    GreedyFuserConfig const effective_config =
        use_problem_sized_config_ ? GreedyConfigForProblem(problem.ops.size()) : config_;

    GreedyFuser fuser(effective_config);
    return fuser.fuse(problem);
}

} // namespace mlsys
