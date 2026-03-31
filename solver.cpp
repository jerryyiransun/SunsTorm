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
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

using namespace std;

namespace mlsys {

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

    unique_ptr<Tiler> tiler = make_unique<GreedyTiler>();
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

} // namespace mlsys
