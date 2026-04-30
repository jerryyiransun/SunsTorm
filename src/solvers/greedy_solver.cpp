/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "solver.h"

#include "fuser.h"
#include "solution_writer.h"
#include "solver_common.h"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

using namespace std;

namespace mlsys {
namespace {

auto GreedyConfigForProblem(size_t num_ops) -> GreedyFuserConfig {
    constexpr double kTopKFailurePenalty = 0.08;
    if (num_ops <= 5) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 24) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 32) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 64) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 128) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    if (num_ops <= 256) {
        return GreedyFuserConfig{
            .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
    }
    return GreedyFuserConfig{
        .search_depth = 4, .beam_width = 16, .topk_failure_penalty = kTopKFailurePenalty};
}

} // namespace

GreedySolver::GreedySolver()
    : use_problem_sized_config_(true), config_{.search_depth = 0,
                                               .beam_width = 0,
                                               .topk_failure_penalty = 0.0} {}

GreedySolver::GreedySolver(std::string anytime_output_path) : GreedySolver() {
    anytime_output_path_ = std::move(anytime_output_path);
}

GreedySolver::GreedySolver(GreedyFuserConfig config)
    : use_problem_sized_config_(false), config_(config) {}

GreedySolver::GreedySolver(GreedyFuserConfig config, std::string anytime_output_path)
    : GreedySolver(config) {
    anytime_output_path_ = std::move(anytime_output_path);
}

auto GreedySolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    GreedyFuserConfig const effective_config =
        use_problem_sized_config_ ? GreedyConfigForProblem(problem.ops.size()) : config_;

    if (!anytime_output_path_.has_value()) {
        GreedyFuser fuser(effective_config);
        return fuser.fuse(problem);
    }

    AnytimeSolutionWriter writer(*anytime_output_path_);
    auto start_status = writer.Start();
    if (!start_status.ok()) {
        return start_status;
    }

    bool first_publish = true;
    GreedyFuser fuser(effective_config,
                      [&](const Solution& solution, TotalLatency cost) -> absl::Status {
                          if (first_publish) {
                              first_publish = false;
                              auto status = writer.PublishAndWaitForWrite(solution, cost);
                              if (status.ok()) {
                                  solver_internal::MaybePauseAfterBaselineForTest();
                              }
                              return status;
                          }
                          return writer.PublishIfBetter(solution, cost);
                      });

    auto solution_status = fuser.fuse(problem);
    auto stop_status = writer.StopAndFlush();
    if (!solution_status.ok()) {
        return solution_status.status();
    }
    if (!stop_status.ok()) {
        return stop_status;
    }

    return solution_status.value();
}

} // namespace mlsys
