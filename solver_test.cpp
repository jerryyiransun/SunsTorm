#include <vector>

#include <gtest/gtest.h>

#include "fuser.h"
#include "mlsys.h"
#include "solver.h"

namespace {

auto MakeProblemForGreedySolverTest() -> mlsys::Problem {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 256, .height = 256},
        {.width = 256, .height = 256},
        {.width = 256, .height = 256},
        {.width = 256, .height = 256},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 35'000;
    problem.slow_memory_bandwidth = 1;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};
    return problem;
}

TEST(SolverTest, GreedySolverMatchesGreedyFuserForSameConfig) {
    mlsys::Problem problem = MakeProblemForGreedySolverTest();
    mlsys::GreedyFuserConfig config{.search_depth = 2, .beam_width = 8};

    mlsys::GreedyFuser fuser(config);
    auto fused = fuser.fuse(problem);
    ASSERT_TRUE(fused.ok()) << fused.status().message();

    mlsys::GreedySolver solver(config);
    auto solved = solver.solve(problem);
    ASSERT_TRUE(solved.ok()) << solved.status().message();

    EXPECT_EQ(*solved, *fused);
}

TEST(SolverTest, GreedySolverDefaultProducesEvaluatablePlan) {
    mlsys::Problem problem = MakeProblemForGreedySolverTest();

    mlsys::GreedySolver solver;
    auto solved = solver.solve(problem);
    ASSERT_TRUE(solved.ok()) << solved.status().message();

    auto eval = mlsys::Evaluate(problem, *solved);
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

} // namespace
