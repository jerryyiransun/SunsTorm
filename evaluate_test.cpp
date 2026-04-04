#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mlsys.h"
#include "tiler.h"

namespace {

std::string TestDataPath(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// asserts that evaluating a (problem, solution) pair succeeds.
void ExpectPass(const std::string& input_file, const std::string& output_file) {
    auto problem = mlsys::ReadProblem(TestDataPath(input_file));
    ASSERT_TRUE(problem.ok()) << problem.status().message();

    auto solution = mlsys::ReadSolution(TestDataPath(output_file));
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    auto result = mlsys::Evaluate(problem.value(), solution.value());
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_GE(result.value(), 0);
}

// asserts that evaluating a (problem, solution) pair fails with a message containing
// `expected_substr`.
void ExpectFail(const std::string& input_file, const std::string& output_file,
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

// ---- Example 1 ----

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

// ---- Example 2 ----

TEST(EvaluateTest, Example2_OutputA_Pass) {
    ExpectPass("example-2-input.json", "example-2-output-A.json");
}

TEST(EvaluateTest, Example2_OutputB_Pass) {
    ExpectPass("example-2-input.json", "example-2-output-B.json");
}

// ---- Example 3 ----

TEST(EvaluateTest, Example3_OutputA_Pass) {
    ExpectPass("example-3-input.json", "example-3-output-A.json");
}

TEST(EvaluateTest, Example3_OutputB_Pass) {
    ExpectPass("example-3-input.json", "example-3-output-B.json");
}

TEST(EvaluateTest, Example3_OutputC_Pass) {
    ExpectPass("example-3-input.json", "example-3-output-C.json");
}

// ---- Example 4 ----

TEST(EvaluateTest, Example4_OutputA_Pass) {
    ExpectPass("example-4-input.json", "example-4-output-A.json");
}

TEST(EvaluateTest, Example4_OutputB_Pass) {
    ExpectPass("example-4-input.json", "example-4-output-B.json");
}

// ---- Example 5 ----

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
        {.width = 64, .height = 64}, // lhs
        {.width = 64, .height = 64}, // rhs
        {.width = 64, .height = 64}  // out
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1000}};
    // Output(64x64) + rhs(64x64) + lhs(64x64 retained) = 12,288 > 10,000
    problem.fast_memory_capacity = 10'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {0}; // retain loaded input tensor
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
        {.width = 64, .height = 64}, // in
        {.width = 64, .height = 64}  // out
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {99}; // invalid tensor index
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
        {.width = 128, .height = 128}, // in
        {.width = 128, .height = 128}  // out
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1}; // 4 tiles expected
    sg.traversal_order = mlsys::TraversalOrder{0, 1, 2};      // invalid length
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Invalid Traversal Order]"),
              std::string::npos)
        << result.status().message();
}

TEST(EvaluateTest, SameSubgraphChainPassesWithEphemeralInputs) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // intermediate
        {.width = 64, .height = 64}  // output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_TRUE(result.ok()) << result.status().message();
}

TEST(EvaluateTest, SameSubgraphFanOutPassesWithEphemeralInputs) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // shared intermediate
        {.width = 64, .height = 64}, // branch output
        {.width = 64, .height = 64}  // branch output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0, 1, 2};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_TRUE(result.ok()) << result.status().message();
}

TEST(EvaluateTest, CrossSubgraphCannotUseInternalIntermediate) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // internal intermediate
        {.width = 64, .height = 64}, // final output of subgraph 0
        {.width = 64, .height = 64}  // output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg0;
    sg0.ops = {0, 1};
    sg0.tensors_to_retain = {};
    sg0.granularity = {.width = 64, .height = 64, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 1.0;

    mlsys::Subgraph sg1;
    sg1.ops = {2};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 64, .height = 64, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 1.0;
    solution.subgraphs = {sg0, sg1};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Unmet Dependency]"),
              std::string::npos)
        << result.status().message();
}

TEST(EvaluateTest, RetainedTensorExpiresAfterImmediateNextSubgraph) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // retained internal tensor
        {.width = 64, .height = 64}, // subgraph 0 final output
        {.width = 64, .height = 64}, // subgraph 1 output
        {.width = 64, .height = 64}  // output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1, 2}, .outputs = {3}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1, 3}, .outputs = {4}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg0;
    sg0.ops = {0, 1};
    sg0.tensors_to_retain = {1};
    sg0.granularity = {.width = 64, .height = 64, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 1.0;

    mlsys::Subgraph sg1;
    sg1.ops = {2};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 64, .height = 64, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 1.0;

    mlsys::Subgraph sg2;
    sg2.ops = {3};
    sg2.tensors_to_retain = {};
    sg2.granularity = {.width = 64, .height = 64, .depth = 1};
    sg2.traversal_order = std::nullopt;
    sg2.subgraph_latency = 1.0;
    solution.subgraphs = {sg0, sg1, sg2};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Unmet Dependency]"),
              std::string::npos)
        << result.status().message();
}

TEST(EvaluateTest, RetainedTensorCanBeReRetainedForLaterSubgraph) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // retained internal tensor
        {.width = 64, .height = 64}, // subgraph 0 final output
        {.width = 64, .height = 64}, // subgraph 1 output
        {.width = 64, .height = 64}  // output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1, 2}, .outputs = {3}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1, 3}, .outputs = {4}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg0;
    sg0.ops = {0, 1};
    sg0.tensors_to_retain = {1};
    sg0.granularity = {.width = 64, .height = 64, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 1.0;

    mlsys::Subgraph sg1;
    sg1.ops = {2};
    sg1.tensors_to_retain = {1};
    sg1.granularity = {.width = 64, .height = 64, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 1.0;

    mlsys::Subgraph sg2;
    sg2.ops = {3};
    sg2.tensors_to_retain = {};
    sg2.granularity = {.width = 64, .height = 64, .depth = 1};
    sg2.traversal_order = std::nullopt;
    sg2.subgraph_latency = 1.0;
    solution.subgraphs = {sg0, sg1, sg2};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_TRUE(result.ok()) << result.status().message();
}

TEST(EvaluateTest, MissedOutputCheckIgnoresInternalIntermediates) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // internal intermediate
        {.width = 64, .height = 64}  // output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_TRUE(result.ok()) << result.status().message();
}

TEST(EvaluateTest, OmittedOpStillFailsWithMissedOutput) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 64, .height = 64}, // input
        {.width = 64, .height = 64}, // intermediate
        {.width = 64, .height = 64}  // output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::Solution solution;
    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 1.0;
    solution.subgraphs = {sg};

    auto result = mlsys::Evaluate(problem, solution);
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find("[Missed Output]"),
              std::string::npos)
        << result.status().message();
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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

} // namespace
