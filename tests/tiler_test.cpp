/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include <gtest/gtest.h>

#include "mlsys.h"
#include "test_utils.h"
#include "tiler.h"

namespace {

using mlsys::test::ExpectTraversalOrderForBothTilers;
using mlsys::test::MakeSinglePointwiseProblem;

auto MakeMatMulPointwiseEpilogueProblem() -> mlsys::Problem {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 384, .height = 128},
        {.width = 128, .height = 384},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000},
        {.op_type = "Pointwise", .inputs = {2}, .outputs = {3}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 60'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};
    return problem;
}

auto MakeFusedMatMulPointwiseSolution() -> mlsys::Solution {
    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.traversal_order = std::nullopt;

    mlsys::Solution solution;
    solution.subgraphs = {sg};
    return solution;
}

TEST(TilerTest, SingleMatMulCanBeTiledToFitFastMemory) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 512, .height = 512},
        {.width = 512, .height = 512},
        {.width = 512, .height = 512},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000}};
    problem.fast_memory_capacity = 60'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    solution.subgraphs = {sg};

    mlsys::BruteForceTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, Benchmark1BaselinePartitionHasValidTiling) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 512, .height = 512}, {.width = 512, .height = 512}, {.width = 512, .height = 512},
        {.width = 512, .height = 512}, {.width = 512, .height = 512}, {.width = 512, .height = 512},
        {.width = 512, .height = 512}, {.width = 512, .height = 512}, {.width = 512, .height = 512},
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {4}, .base_cost = 2000},
        {.op_type = "Pointwise", .inputs = {4}, .outputs = {5}, .base_cost = 500},
        {.op_type = "MatMul", .inputs = {5, 2}, .outputs = {6}, .base_cost = 2000},
        {.op_type = "MatMul", .inputs = {6, 3}, .outputs = {7}, .base_cost = 2000},
        {.op_type = "Pointwise", .inputs = {7, 0}, .outputs = {8}, .base_cost = 500},
    };
    problem.fast_memory_capacity = 60'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Solution solution;
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        mlsys::Subgraph sg;
        sg.ops = {op_idx};
        sg.tensors_to_retain = {};
        solution.subgraphs.push_back(sg);
    }

    mlsys::BruteForceTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, GreedyTilerStillFindsValidTiling) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 512, .height = 512},
        {.width = 512, .height = 512},
        {.width = 512, .height = 512},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000}};
    problem.fast_memory_capacity = 60'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    solution.subgraphs = {sg};

    mlsys::GreedyTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, GreedyTilerUsesCeilCandidatesBetweenHalvingSteps) {
    auto problem = MakeSinglePointwiseProblem(384, 128);
    problem.fast_memory_capacity = 13'000;

    mlsys::GreedyTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 96);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 64);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, GreedyTilerStartsDepthAtNativeForFinalMatMul) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 384, .height = 128},
        {.width = 128, .height = 384},
        {.width = 128, .height = 128},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.depth, 128);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, BruteForceTilerStartsDepthAtNativeForFinalMatMul) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 384, .height = 128},
        {.width = 128, .height = 384},
        {.width = 128, .height = 128},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::BruteForceTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.depth, 128);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, GreedyTilerUsesSplitKForPointwiseEpilogueFinalOutput) {
    mlsys::Problem problem = MakeMatMulPointwiseEpilogueProblem();
    mlsys::Solution solution = MakeFusedMatMulPointwiseSolution();

    mlsys::GreedyTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.depth, 128);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, BruteForceTilerUsesSplitKForPointwiseEpilogueFinalOutput) {
    mlsys::Problem problem = MakeMatMulPointwiseEpilogueProblem();
    mlsys::Solution solution = MakeFusedMatMulPointwiseSolution();

    mlsys::BruteForceTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.depth, 128);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, CostGuidedDivisorTilerUsesSplitKForPointwiseEpilogueFinalOutput) {
    mlsys::Problem problem = MakeMatMulPointwiseEpilogueProblem();
    mlsys::Solution solution = MakeFusedMatMulPointwiseSolution();

    mlsys::CostGuidedDivisorTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.depth, 128);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, CostGuidedDivisorTilerFindsValidTiling) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 512, .height = 512},
        {.width = 512, .height = 512},
        {.width = 512, .height = 512},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000}};
    problem.fast_memory_capacity = 60'000;
    problem.slow_memory_bandwidth = 20;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::CostGuidedDivisorTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, CostGuidedDivisorTilerUsesGreedyCeilCandidateSet) {
    auto problem = MakeSinglePointwiseProblem(500, 128);
    problem.fast_memory_capacity = 12'000;

    mlsys::CostGuidedDivisorTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 84);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 64);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, CostGuidedDivisorTilerChoosesLowerLatencySplitKMove) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 256, .height = 128},
        {.width = 128, .height = 256},
        {.width = 128, .height = 128},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 2000}};
    problem.fast_memory_capacity = 50'000;
    problem.slow_memory_bandwidth = 1'000'000'000;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::CostGuidedDivisorTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();

    EXPECT_EQ(tiled.value().subgraphs[0].granularity.width, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.height, 128);
    EXPECT_EQ(tiled.value().subgraphs[0].granularity.depth, 128);

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, CostGuidedDivisorTilerPreservesSnakeTraversalOrder) {
    auto problem = MakeSinglePointwiseProblem(384, 256);

    mlsys::CostGuidedDivisorTiler tiler;
    auto tiled = tiler.tile(problem, mlsys::test::MakeSingleOpSolution());
    ASSERT_TRUE(tiled.ok()) << tiled.status().message();
    ASSERT_TRUE(tiled.value().subgraphs[0].traversal_order.has_value());
    EXPECT_EQ(tiled.value().subgraphs[0].traversal_order.value(),
              (mlsys::TraversalOrder{0, 1, 2, 5, 4, 3}));

    auto eval = mlsys::Evaluate(problem, tiled.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

TEST(TilerTest, WidthDominantGridUsesHorizontalSnakeOrder) {
    auto problem = MakeSinglePointwiseProblem(384, 256);
    ExpectTraversalOrderForBothTilers(problem, mlsys::TraversalOrder{0, 1, 2, 5, 4, 3});
}

TEST(TilerTest, HeightDominantGridUsesVerticalSnakeOrder) {
    auto problem = MakeSinglePointwiseProblem(256, 384);
    ExpectTraversalOrderForBothTilers(problem, mlsys::TraversalOrder{0, 2, 4, 5, 3, 1});
}

TEST(TilerTest, SquareGridUsesWidthFirstSnakeOrder) {
    auto problem = MakeSinglePointwiseProblem(256, 256);
    ExpectTraversalOrderForBothTilers(problem, mlsys::TraversalOrder{0, 1, 3, 2});
}

TEST(TilerTest, SingleTileProducesSingletonTraversalOrder) {
    auto problem = MakeSinglePointwiseProblem(128, 128);
    ExpectTraversalOrderForBothTilers(problem, mlsys::TraversalOrder{0});
}

} // namespace
