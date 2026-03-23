#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;

namespace mlsys {

class CostModel {
  public:
    auto estimate(const Problem& problem, const Solution& solution)
        -> StatusOr<std::tuple<Solution, SubgraphLatency>>;
};

} // namespace mlsys
