#include <string>

#include <gtest/gtest.h>

#include "mlsys.h"
#include "test_utils.h"

namespace {

using mlsys::test::ExpectFail;
using mlsys::test::ExpectPass;

TEST(EvaluateTest, Example1_OutputA_Pass) {
    ExpectPass("example-1-input.json", "example-1-output-A.json");
}

TEST(EvaluateTest, Example1_OutputB_Pass) {
    ExpectPass("example-1-input.json", "example-1-output-B.json");
}

TEST(EvaluateTest, Example1_OutputC_Pass) {
    ExpectPass("example-1-input.json", "example-1-output-C.json");
}

TEST(EvaluateTest, Example1_Fail_Dependency) {
    ExpectFail("example-1-input.json", "example-1-output-F-dependency.json", "[Unmet Dependency]");
}

TEST(EvaluateTest, Example1_Fail_InvalidOp) {
    ExpectFail("example-1-input.json", "example-1-output-F-invalid-op.json", "[Invalid Op Index]");
}

TEST(EvaluateTest, Example1_Fail_MissedOutput) {
    ExpectFail("example-1-input.json", "example-1-output-F-missed-output.json", "[Missed Output]");
}

TEST(EvaluateTest, Example2_OutputA_Pass) {
    ExpectPass("example-2-input.json", "example-2-output-A.json");
}

TEST(EvaluateTest, Example2_OutputB_Pass) {
    ExpectPass("example-2-input.json", "example-2-output-B.json");
}

TEST(EvaluateTest, Example3_OutputA_Pass) {
    ExpectPass("example-3-input.json", "example-3-output-A.json");
}

TEST(EvaluateTest, Example3_OutputB_Pass) {
    ExpectPass("example-3-input.json", "example-3-output-B.json");
}

TEST(EvaluateTest, Example3_OutputC_Pass) {
    ExpectPass("example-3-input.json", "example-3-output-C.json");
}

TEST(EvaluateTest, Example4_OutputA_Pass) {
    ExpectPass("example-4-input.json", "example-4-output-A.json");
}

TEST(EvaluateTest, Example4_OutputB_Pass) {
    ExpectPass("example-4-input.json", "example-4-output-B.json");
}

TEST(EvaluateTest, Example5_Fail_Capacity) {
    ExpectFail("example-5-input.json", "example-5-output-F-capacity.json",
               "[Fast Memory Capacity Exceeded]");
}

TEST(EvaluateTest, Example5_OutputB_Pass) {
    ExpectPass("example-5-input.json", "example-5-output-B.json");
}

TEST(EvaluateTest, RetainedLoadedInput_CountsTowardCapacity) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64},
        {.width = 64, .height = 64},
        {.width = 64, .height = 64},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1000}};
    problem.fast_memory_capacity = 10'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {0};
    sg.granularity = {.width = 64, .height = 64, .depth = 64};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Fast Memory Capacity Exceeded]"),
              std::string::npos)
        << result.status().message();
}

TEST(EvaluateTest, RetainedTensor_OutOfRange_Fail) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64},
        {.width = 64, .height = 64},
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {99};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Invalid Retained Tensor]"),
              std::string::npos)
        << result.status().message();
}

TEST(EvaluateTest, TraversalOrder_LengthMismatch_Fail) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = mlsys::TraversalOrder{0, 1, 2};
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Invalid Traversal Order]"),
              std::string::npos)
        << result.status().message();
}

} // namespace
