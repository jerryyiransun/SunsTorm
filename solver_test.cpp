#include <gtest/gtest.h>

#include "mlsys.h"
#include "solver.h"

namespace {

auto MakeDivisorSensitivePointwiseProblem() -> mlsys::Problem {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 384, .height = 128},
        {.width = 384, .height = 128},
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100}};
    problem.fast_memory_capacity = 13'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};
    return problem;
}

TEST(SolverTest, BaseSolverUsesCostGuidedDivisorTiler) {
    auto problem = MakeDivisorSensitivePointwiseProblem();

    mlsys::BaseSolver solver;
    auto solution = solver.solve(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution.value().subgraphs.size(), 1);
    EXPECT_EQ(solution.value().subgraphs[0].granularity.width, 96);
    EXPECT_EQ(solution.value().subgraphs[0].granularity.height, 128);

    auto eval = mlsys::Evaluate(problem, solution.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(SolverTest, HeuristicSolverUsesCostGuidedDivisorTilerForIntervals) {
    auto problem = MakeDivisorSensitivePointwiseProblem();

    mlsys::HeuristicSolver solver;
    auto solution = solver.solve(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution.value().subgraphs.size(), 1);
    EXPECT_EQ(solution.value().subgraphs[0].granularity.width, 96);
    EXPECT_EQ(solution.value().subgraphs[0].granularity.height, 128);

    auto eval = mlsys::Evaluate(problem, solution.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

} // namespace
