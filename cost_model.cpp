#include "cost_model.h"

namespace mlsys {

// TODO: Implement a cost model that estimates the latency of each subgraph in the solution and
// returns the total latency. Return a tuple of the updated solution with subgraph latencies filled
// in, and the total latency of the solution.
auto CostModel::estimate(const Problem& problem, const Solution& solution)
    -> StatusOr<std::tuple<Solution, SubgraphLatency>> {
    return std::tuple<Solution, SubgraphLatency>{solution, 0};
}

} // namespace mlsys