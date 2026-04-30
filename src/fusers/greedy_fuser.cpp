/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "fuser.h"

#include "fuser_common.h"
#include "tiler.h"

#include <memory>
#include <utility>

namespace mlsys {

GreedyFuser::GreedyFuser(GreedyFuserConfig config)
    : GreedyFuser(config, std::make_unique<GreedyTiler>()) {}

GreedyFuser::GreedyFuser(GreedyFuserConfig config, BestSolutionCallback best_solution_callback)
    : GreedyFuser(config, std::make_unique<GreedyTiler>(), std::move(best_solution_callback)) {}

GreedyFuser::GreedyFuser(GreedyFuserConfig config, std::unique_ptr<Tiler> tiler)
    : GreedyFuser(config, std::move(tiler), nullptr) {}

GreedyFuser::GreedyFuser(GreedyFuserConfig config, std::unique_ptr<Tiler> tiler,
                         BestSolutionCallback best_solution_callback)
    : config_(config), tiler_(std::move(tiler)),
      best_solution_callback_(std::move(best_solution_callback)) {}

GreedyFuser::~GreedyFuser() = default;

auto GreedyFuser::fuse(const Problem& problem) -> StatusOr<Solution> {
    return fuser_internal::RunGreedyFuser(config_, tiler_.get(), best_solution_callback_, problem);
}

} // namespace mlsys
