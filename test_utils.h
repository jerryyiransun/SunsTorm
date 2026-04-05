#pragma once

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "cost_model.h"
#include "mlsys.h"
#include "tiler.h"

namespace mlsys::test {

inline auto TestDataPath(const std::string& filename) -> std::string {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

inline void ExpectPass(const std::string& input_file, const std::string& output_file) {
    auto problem = mlsys::ReadProblem(TestDataPath(input_file));
    ASSERT_TRUE(problem.ok()) << problem.status().message();

    auto solution = mlsys::ReadSolution(TestDataPath(output_file));
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    auto result = mlsys::Evaluate(problem.value(), solution.value());
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_GE(result.value(), 0);
}

inline void ExpectFail(const std::string& input_file, const std::string& output_file,
                       const std::string& expected_substr) {
    auto problem = mlsys::ReadProblem(TestDataPath(input_file));
    ASSERT_TRUE(problem.ok()) << problem.status().message();

    auto solution = mlsys::ReadSolution(TestDataPath(output_file));
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    auto result = mlsys::Evaluate(problem.value(), solution.value());
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find(expected_substr), std::string::npos)
        << "Expected error containing \"" << expected_substr
        << "\", got: " << result.status().message();
}

inline auto MakeSinglePointwiseProblem(int64_t width, int64_t height) -> mlsys::Problem {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = width, .height = height},
        {.width = width, .height = height},
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};
    return problem;
}

inline auto MakeSingleOpSolution() -> mlsys::Solution {
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.traversal_order = std::nullopt;
    mlsys::Solution solution;
    solution.subgraphs = {sg};
    return solution;
}

inline void ExpectTraversalOrderForBothTilers(const mlsys::Problem& problem,
                                              const mlsys::TraversalOrder& expected) {
    {
        mlsys::BruteForceTiler tiler;
        auto tiled = tiler.tile(problem, MakeSingleOpSolution());
        ASSERT_TRUE(tiled.ok()) << tiled.status().message();
        ASSERT_TRUE(tiled.value().subgraphs[0].traversal_order.has_value());
        EXPECT_EQ(tiled.value().subgraphs[0].traversal_order.value(), expected);

        auto eval = mlsys::Evaluate(problem, tiled.value());
        ASSERT_TRUE(eval.ok()) << eval.status().message();
    }

    {
        mlsys::GreedyTiler tiler;
        auto tiled = tiler.tile(problem, MakeSingleOpSolution());
        ASSERT_TRUE(tiled.ok()) << tiled.status().message();
        ASSERT_TRUE(tiled.value().subgraphs[0].traversal_order.has_value());
        EXPECT_EQ(tiled.value().subgraphs[0].traversal_order.value(), expected);

        auto eval = mlsys::Evaluate(problem, tiled.value());
        ASSERT_TRUE(eval.ok()) << eval.status().message();
    }
}

inline void ExpectExampleMatches(const std::string& input_file, const std::string& output_file) {
    auto problem = mlsys::ReadProblem(TestDataPath(input_file));
    ASSERT_TRUE(problem.ok()) << problem.status().message();

    auto solution = mlsys::ReadSolution(TestDataPath(output_file));
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    std::unique_ptr<mlsys::CostModel> cost_model_ptr =
        std::make_unique<mlsys::CostModel>(problem.value());
    auto estimated = cost_model_ptr->estimate(solution.value());
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    const auto& estimated_solution = std::get<0>(estimated.value());
    const auto& expected_solution = solution.value();

    ASSERT_EQ(estimated_solution.subgraphs.size(), expected_solution.subgraphs.size());

    double expected_total = 0.0;
    for (size_t i = 0; i < expected_solution.subgraphs.size(); ++i) {
        EXPECT_NEAR(estimated_solution.subgraphs[i].subgraph_latency,
                    expected_solution.subgraphs[i].subgraph_latency, 1e-6);
        expected_total += expected_solution.subgraphs[i].subgraph_latency;
    }

    EXPECT_NEAR(std::get<1>(estimated.value()), expected_total, 1e-6);
}

} // namespace mlsys::test
