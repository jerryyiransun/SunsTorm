#include "solver.h"

#include "cost_model.h"
#include "fuser.h"
#include "tiler.h"

#include <memory>
#include <tuple>

using namespace std;

namespace mlsys {

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
