/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;

namespace mlsys {

class Tiler {
  public:
    virtual ~Tiler() = default;
    virtual auto tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> = 0;
};

class BruteForceTiler : public Tiler {
  public:
    auto tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> override;

  private:
    std::unordered_map<std::string, std::optional<Granularity>> best_granularity_cache_;
};

class GreedyTiler : public Tiler {
  public:
    auto tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> override;
};

class CostGuidedDivisorTiler : public Tiler {
  public:
    auto tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> override;
    auto tile_subgraph(const Problem& problem, const Solution& solution, size_t sg_idx)
        -> StatusOr<Solution>;
};

} // namespace mlsys
