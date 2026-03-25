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
    // Tensor id that this rectangular tile belongs to.
    size_t tensor_idx;
    // Half-open horizontal range [x0, x1) in tensor coordinates.
    int64_t x0;
    int64_t x1;
    // Half-open vertical range [y0, y1) in tensor coordinates.
    int64_t y0;
    int64_t y1;

    // Returns tile area in tensor elements. Degenerate ranges return 0.
    [[nodiscard]] auto area() const -> int64_t;

    // Computes exact union area across all tiles using per-tensor sweep-line.
    // Tiles from different tensors are never merged together.
    static auto compute_non_overlapping_area(const std::vector<Tile>& tiles) -> int64_t;
};

class CostModel {
  public:
    CostModel(Problem problem) { problem_ = problem; };
    // Estimates each subgraph latency and returns:
    // 1) updated solution with subgraph_latency filled in
    // 2) total latency across subgraphs
    auto estimate(const Solution& solution) -> StatusOr<std::tuple<Solution, SubgraphLatency>>;

  private:
    Problem problem_;
};

} // namespace mlsys
