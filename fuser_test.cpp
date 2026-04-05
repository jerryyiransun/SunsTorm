#include <vector>

#include <gtest/gtest.h>

#include "fuser.h"
#include "mlsys.h"

namespace {

TEST(FuserTest, EnumeratesRecomputedBranchProducerAcrossSubgraphs) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1000},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 35'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::BruteForceFuser fuser;
    auto solutions = fuser.fuse(problem);
    ASSERT_TRUE(solutions.ok()) << solutions.status().message();

    bool found_recomputed_schedule = false;
    for (const auto& solution : solutions.value()) {
        if (solution.subgraphs.size() != 2) {
            continue;
        }
        if (solution.subgraphs[0].ops == std::vector<size_t>({0, 1}) &&
            solution.subgraphs[1].ops == std::vector<size_t>({0, 2})) {
            found_recomputed_schedule = true;
            break;
        }
    }

    EXPECT_TRUE(found_recomputed_schedule);
}

} // namespace
