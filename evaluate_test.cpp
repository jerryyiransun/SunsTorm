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

TEST(EvaluateTest, RetainedEphemeralTensor_CountsTowardNextSubgraphCapacity) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input to op0
        {.width = 128, .height = 128}, // t1 ephemeral in sg0, retained for sg1
        {.width = 128, .height = 128}, // t2 output of op1 in sg0
        {.width = 128, .height = 128}, // t3 rhs input for sg1 matmul
        {.width = 128, .height = 128}, // t4 output of sg1 matmul
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // op0
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1}, // op1
        {.op_type = "MatMul", .inputs = {1, 3}, .outputs = {4}, .base_cost = 1}, // op2
    };
    // sg0 working set (64x64 tile + retained full t1) = 4096 + 16384 = 20480 (fits).
    // sg1 working set with retained full t1:
    //   retained t1 full 16384 + rhs strip 4096 + output tile 4096 = 24576 (exceeds).
    problem.fast_memory_capacity = 22'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;

    mlsys::Subgraph sg0;
    sg0.ops = {0, 1};
    sg0.tensors_to_retain = {1}; // retain ephemeral t1 into next subgraph
    sg0.granularity = {.width = 64, .height = 64, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 1.0;

    mlsys::Subgraph sg1;
    sg1.ops = {2};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 64, .height = 64, .depth = 64};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 1.0;

    solution.subgraphs = {sg0, sg1};

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
