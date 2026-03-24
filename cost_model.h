#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;

namespace mlsys {

class Tile {
  public:
    size_t tensor_idx;
    int64_t x0;
    int64_t x1;
    int64_t y0;
    int64_t y1;

    auto area() const -> int64_t;

    static auto ComputeNonOverlappingArea(const std::vector<Tile>& tiles) -> int64_t;
};

class CostModel {
  public:
    auto estimate(const Problem& problem, const Solution& solution)
        -> StatusOr<std::tuple<Solution, SubgraphLatency>>;
};

} // namespace mlsys
