#include "solver.h"

#include "cost_model.h"
#include "tiler.h"

#include <memory>
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

} // namespace mlsys
