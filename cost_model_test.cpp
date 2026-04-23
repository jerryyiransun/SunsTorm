#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "cost_model.h"
#include "mlsys.h"
#include "test_utils.h"

namespace {

using mlsys::test::ExpectExampleMatches;

TEST(TileTest, AreaUsesRawRectangleSize) {
    mlsys::Tile tile{.tensor_idx = 0, .x0 = 1, .x1 = 11, .y0 = 2, .y1 = 7};
    EXPECT_EQ(tile.area(), 50);
}

TEST(TileTest, AreaIsZeroForDegenerateRanges) {
    mlsys::Tile bad_x{.tensor_idx = 0, .x0 = 10, .x1 = 5, .y0 = 0, .y1 = 10};
    mlsys::Tile bad_y{.tensor_idx = 0, .x0 = 0, .x1 = 10, .y0 = 8, .y1 = 4};
    EXPECT_EQ(bad_x.area(), 0);
    EXPECT_EQ(bad_y.area(), 0);
}

TEST(CostModelTest, DuplicateFullBoundaryInputTilesAreNotDoubleCounted) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 shared input
        {.width = 128, .height = 128}, // t1 final output
        {.width = 128, .height = 128}, // t2 final output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {2}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // Duplicate full-tensor boundary reads are deduplicated:
    // read t0 once + write t1 and t2.
    // (16384 + 16384 + 16384) / 100 = 491.52
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 491.52, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 491.52, 1e-6);
}

TEST(CostModelTest, FullSharedInputSuppressesPartialRead) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 shared input
        {.width = 128, .height = 128}, // t1 full output of op0 (pointwise)
        {.width = 128, .height = 128}, // t2 RHS for op1 (matmul)
        {.width = 128, .height = 64},  // t3 half-height output of op1
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "MatMul", .inputs = {0, 2}, .outputs = {3}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 128};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // Shared boundary input t0 is required as full and partial in the same step.
    // Full fetch takes canonical residency, so the partial request is a hit.
    //
    // Reads: t0(full)=16384, t2=16384
    // Writes: t1=16384, t3=8192
    // Total memory time: (16384 + 16384 + 16384 + 8192) / 100 = 573.44
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 573.44, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 573.44, 1e-6);
}

TEST(CostModelTest, PartialThenFullSharedInputStillChargesSingleFullRead) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 shared boundary input
        {.width = 128, .height = 128}, // t1 rhs of first matmul
        {.width = 128, .height = 128}, // t2 intermediate output (linked matmuls)
        {.width = 128, .height = 128}, // t3 final output
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1}, // op0
        {.op_type = "MatMul", .inputs = {2, 0}, .outputs = {3}, .base_cost = 1}, // op1
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 64};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // depth=64 with K=128 => split-k runs 2 k-steps.
    // In step1 backprop, t0 is discovered first as partial (op1 rhs), then as full (op0 lhs).
    // Expected charging is one full t0 for the step.
    // Step1 reads: t0(full)=16384, t1(partial)=8192. (no write, intermediate split-k step)
    // Step2 reads: t1(partial next k-slice)=8192; writes t3(full)=16384.
    // Total memory time: (24576 + 24576) / 100 = 491.52.
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 491.52, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 491.52, 1e-6);
}

TEST(CostModelTest, DistinctPartialSpecsOnSameTensorAreDuplicated) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 shared tensor with two different partial specs
        {.width = 128, .height = 128}, // t1 rhs of op0
        {.width = 128, .height = 128}, // t2 lhs of op1
        {.width = 128, .height = 128}, // t3 output of op0
        {.width = 128, .height = 128}, // t4 output of op1
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {3}, .base_cost = 1}, // t0 lhs partial
        {.op_type = "MatMul", .inputs = {2, 0}, .outputs = {4}, .base_cost = 1}, // t0 rhs partial
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 64};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // depth=64 with K=128 => 2 split-k steps.
    // Each step reads four partials (t0 lhs + t0 rhs + t1 + t2) = 32768.
    // Distinct t0 partial specs are both charged in each step (no partial-partial reuse).
    // Final step writes t3+t4 full = 32768.
    // Total memory time: (32768 + 32768 + 32768) / 100 = 983.04
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 983.04, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 983.04, 1e-6);
}

TEST(CostModelTest, RetainedFullTensorSuppressesLaterPartialRead) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input to sg0
        {.width = 128, .height = 128}, // t1 produced in sg0, retained full
        {.width = 128, .height = 128}, // t2 rhs input to sg1 matmul
        {.width = 128, .height = 128}, // t3 output of sg1
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // sg0 op
        {.op_type = "MatMul", .inputs = {1, 2}, .outputs = {3}, .base_cost = 1}, // sg1 op
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg0;
    sg0.ops = {0};
    sg0.tensors_to_retain = {1};
    sg0.granularity = {.width = 128, .height = 128, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 0.0;

    mlsys::Subgraph sg1;
    sg1.ops = {1};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 128, .height = 128, .depth = 64};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg0, sg1}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // sg0: read t0 full only => 16384 / 100 = 163.84
    // sg1 split-k:
    // - step1 read t2 partial 8192 (t1 partial covered by retained full t1), no write
    // - step2 read t2 partial 8192, write t3 full 16384
    // sg1 total = 32768 / 100 = 327.68
    // total = 163.84 + 327.68 = 491.52
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 163.84, 1e-6);
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[1].subgraph_latency, 327.68, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 491.52, 1e-6);
}

TEST(CostModelTest, SingleMatMulSameTensorInputsCountStripsIndependently) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 shared input used as both lhs and rhs
        {.width = 128, .height = 128}, // t1 output
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 0}, .outputs = {1}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    // Single op, but smaller granularity forces 4 spatial tiles.
    sg.granularity = {.width = 64, .height = 64, .depth = 128};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // For each 64x64 output tile, the matmul needs two strips from same tensor t0:
    // - horizontal LHS strip: 128x64 = 8192
    // - vertical RHS strip:   64x128 = 8192
    // They are counted independently even though they overlap and share tensor id.
    //
    // Raster traversal over 4 tiles yields read sequence:
    // step1: 16384, step2: 8192 (reuse one strip), step3: 16384, step4: 8192
    // => total reads 49152. Writes are 4 * 4096 = 16384.
    // Total memory time: (49152 + 16384) / 100 = 655.36
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 655.36, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 655.36, 1e-6);
}

TEST(CostModelTest, DuplicateInternalProducerTilesDoNotDoubleCountCompute) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input
        {.width = 128, .height = 128}, // t1 shared internal
        {.width = 128, .height = 128}, // t2 final output
        {.width = 128, .height = 128}, // t3 final output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // op0
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1}, // op1
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 1}, // op2
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1, 2};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // Unit base costs keep this memory-bound:
    // read t0 once (full-fetch dedupe across duplicated internal paths), write t2+t3.
    // (16384 + 16384 + 16384) / 100 = 491.52
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 491.52, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 491.52, 1e-6);
}

TEST(CostModelTest, DuplicateInternalProducerTilesDoNotDoubleCountCompute_WhenComputeBound) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input
        {.width = 128, .height = 128}, // t1 shared internal
        {.width = 128, .height = 128}, // t2 final output
        {.width = 128, .height = 128}, // t3 final output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1000}, // op0
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1000}, // op1
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 1000}, // op2
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 1'000'000'000'000;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1, 2};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // op0 output tile is reached by two backward paths, but compute is area-deduped:
    // op0 + op1 + op2 = 3000.
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 3000.0, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 3000.0, 1e-6);
}

TEST(CostModelTest, Example1A_Golden) {
    ExpectExampleMatches("example-1-input.json", "example-1-output-A.json");
}

TEST(CostModelTest, Example1B_Golden) {
    ExpectExampleMatches("example-1-input.json", "example-1-output-B.json");
}

TEST(CostModelTest, Example1C_Golden) {
    ExpectExampleMatches("example-1-input.json", "example-1-output-C.json");
}

TEST(CostModelTest, Example2A_Golden) {
    ExpectExampleMatches("example-2-input.json", "example-2-output-A.json");
}

TEST(CostModelTest, Example2B_Golden) {
    ExpectExampleMatches("example-2-input.json", "example-2-output-B.json");
}

TEST(CostModelTest, Example3A_Golden) {
    ExpectExampleMatches("example-3-input.json", "example-3-output-A.json");
}

TEST(CostModelTest, Example3B_Golden) {
    ExpectExampleMatches("example-3-input.json", "example-3-output-B.json");
}

TEST(CostModelTest, Example3C_Golden) {
    ExpectExampleMatches("example-3-input.json", "example-3-output-C.json");
}

TEST(CostModelTest, Example4A_Golden) {
    ExpectExampleMatches("example-4-input.json", "example-4-output-A.json");
}

TEST(CostModelTest, Example4B_Golden) {
    ExpectExampleMatches("example-4-input.json", "example-4-output-B.json");
}

TEST(CostModelTest, Example5B_Golden) {
    ExpectExampleMatches("example-5-input.json", "example-5-output-B.json");
}

TEST(CostModelTest, SmallTileBelowNativeKeepsFullComputePerStep) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {{.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1000}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 1'000'000'000;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated = cost_model_ptr->estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 4000.0, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 4000.0, 1e-6);
}

TEST(CostModelTest, SplitKNonDivisibleScalesFinalChunkCompute) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 100, .height = 128},
        {.width = 128, .height = 100},
        {.width = 128, .height = 128},
    };
    problem.ops = {{.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1000}};
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 1'000'000'000;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {2};
    sg.granularity = {.width = 128, .height = 128, .depth = 32};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated = cost_model_ptr->estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 1000.0, 1e-3);
}

TEST(CostModelTest, SplitKRetainedGraphOutputStillWritesOnlyFinalKStep) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 lhs input
        {.width = 128, .height = 128}, // t1 rhs input
        {.width = 128, .height = 128}, // t2 graph-final output
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg_no_retain;
    sg_no_retain.ops = {0};
    sg_no_retain.tensors_to_retain = {};
    sg_no_retain.granularity = {.width = 128, .height = 128, .depth = 32};
    sg_no_retain.traversal_order = std::nullopt;
    sg_no_retain.subgraph_latency = 0.0;

    mlsys::Subgraph sg_retain = sg_no_retain;
    sg_retain.tensors_to_retain = {2};

    mlsys::Solution solution_no_retain{.subgraphs = {sg_no_retain}};
    mlsys::Solution solution_retain{.subgraphs = {sg_retain}};

    mlsys::CostModel cost_model(problem);
    auto estimated_no_retain = cost_model.estimate(solution_no_retain);
    auto estimated_retain = cost_model.estimate(solution_retain);
    ASSERT_TRUE(estimated_no_retain.ok()) << estimated_no_retain.status().message();
    ASSERT_TRUE(estimated_retain.ok()) << estimated_retain.status().message();

    // split-k=4 (depth 32 over K=128):
    // reads = 4 * (128*32 + 32*128) = 32768
    // write graph output once at final k-step = 16384
    // total memory time = (32768 + 16384) / 100 = 491.52
    EXPECT_NEAR(std::get<1>(estimated_no_retain.value()), 491.52, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated_retain.value()), 491.52, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated_retain.value()), std::get<1>(estimated_no_retain.value()),
                1e-6);
}

TEST(CostModelTest, EphemeralIntermediateHasNoSlowMemoryTransferCost) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0, 1};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated = cost_model_ptr->estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 3276.8, 1e-6);
}

TEST(CostModelTest, RetainingEphemeralTensorCarriesToNextSubgraph) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input
        {.width = 128, .height = 128}, // t1 ephemeral in sg0, consumed by sg1
        {.width = 128, .height = 128}, // t2 sg0 final output
        {.width = 128, .height = 128}, // t3 sg1 final output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // op0
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1}, // op1 in sg0
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 1}, // op2 in sg1
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg0_no_retain;
    sg0_no_retain.ops = {0, 1};
    sg0_no_retain.tensors_to_retain = {};
    sg0_no_retain.granularity = {.width = 128, .height = 128, .depth = 1};
    sg0_no_retain.traversal_order = std::nullopt;
    sg0_no_retain.subgraph_latency = 0.0;

    mlsys::Subgraph sg0_retain_ephemeral = sg0_no_retain;
    sg0_retain_ephemeral.tensors_to_retain = {1};

    mlsys::Subgraph sg1;
    sg1.ops = {2};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 128, .height = 128, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 0.0;

    mlsys::Solution no_retain{.subgraphs = {sg0_no_retain, sg1}};
    mlsys::Solution retain_ephemeral{.subgraphs = {sg0_retain_ephemeral, sg1}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated_no_retain = cost_model_ptr->estimate(no_retain);
    auto estimated_retain = cost_model_ptr->estimate(retain_ephemeral);

    ASSERT_TRUE(estimated_no_retain.ok()) << estimated_no_retain.status().message();
    ASSERT_TRUE(estimated_retain.ok()) << estimated_retain.status().message();

    // Without retaining t1:
    // sg0: read t0 + write t2 = 327.68
    // sg1: read t1 + write t3 = 327.68
    EXPECT_NEAR(std::get<1>(estimated_no_retain.value()), 655.36, 1e-6);

    // With retaining t1:
    // sg0 unchanged (t1 is ephemeral inside sg0): 327.68
    // sg1 reuses retained full t1, so only write t3: 163.84
    EXPECT_NEAR(std::get<1>(estimated_retain.value()), 491.52, 1e-6);
    EXPECT_LT(std::get<1>(estimated_retain.value()), std::get<1>(estimated_no_retain.value()));
}

TEST(CostModelTest, RetainedBoundaryTensorRemovesReload) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg0;
    sg0.ops = {0};
    sg0.tensors_to_retain = {};
    sg0.granularity = {.width = 128, .height = 128, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 0.0;

    mlsys::Subgraph sg1;
    sg1.ops = {1};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 128, .height = 128, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 0.0;

    mlsys::Solution no_retain{.subgraphs = {sg0, sg1}};

    mlsys::Subgraph sg0_retain = sg0;
    sg0_retain.tensors_to_retain = {1};
    mlsys::Solution retain_boundary{.subgraphs = {sg0_retain, sg1}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated_no_retain = cost_model_ptr->estimate(no_retain);
    auto estimated_retain = cost_model_ptr->estimate(retain_boundary);

    ASSERT_TRUE(estimated_no_retain.ok()) << estimated_no_retain.status().message();
    ASSERT_TRUE(estimated_retain.ok()) << estimated_retain.status().message();

    EXPECT_NEAR(std::get<1>(estimated_no_retain.value()), 6553.6, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated_retain.value()), 3276.8, 1e-6);
}

TEST(CostModelTest, TensorConsumedInsideSameSubgraphDoesNotForceWriteback) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input
        {.width = 128, .height = 128}, // t1 shared intermediate
        {.width = 128, .height = 128}, // t2 final output of sg0
        {.width = 128, .height = 128}, // t3 final output of sg1
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // op0
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1}, // op1 (inside sg0)
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 1}, // op2 (in sg1)
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100; // Memory-bound to expose writeback accounting.
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg0;
    sg0.ops = {0, 1};
    sg0.tensors_to_retain = {};
    sg0.granularity = {.width = 128, .height = 128, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 0.0;

    mlsys::Subgraph sg1;
    sg1.ops = {2};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 128, .height = 128, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg0, sg1}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated = cost_model_ptr->estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // t1 is consumed inside sg0, so it remains ephemeral and does not force writeback.
    // sg0: (read t0 + write t2) / 100 = (16384 * 2) / 100 = 327.68
    // sg1: (read t1 + write t3) / 100 = (16384 * 2) / 100 = 327.68
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 327.68, 1e-6);
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[1].subgraph_latency, 327.68, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 655.36, 1e-6);
}

TEST(CostModelTest, RetainedPureOutputInFinalSubgraphStillWritesBack) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input
        {.width = 128, .height = 128}, // t1 pure output
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // op0
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100; // Memory-bound to expose writeback accounting.
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {1};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // Final subgraph must still flush required boundary outputs even if retained:
    // (read t0 + write t1) / 100 = (16384 * 2) / 100 = 327.68
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 327.68, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 327.68, 1e-6);
}

TEST(CostModelTest, RetainedPureOutputInNonFinalSubgraphStillWritesBack) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, // t0 input for sg0
        {.width = 128, .height = 128}, // t1 pure graph output from sg0
        {.width = 128, .height = 128}, // t2 input for sg1
        {.width = 128, .height = 128}, // t3 pure graph output from sg1
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1}, // op0
        {.op_type = "Pointwise", .inputs = {2}, .outputs = {3}, .base_cost = 1}, // op1
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg0;
    sg0.ops = {0};
    sg0.tensors_to_retain = {1}; // retained but must still be flushed (graph output)
    sg0.granularity = {.width = 128, .height = 128, .depth = 1};
    sg0.traversal_order = std::nullopt;
    sg0.subgraph_latency = 0.0;

    mlsys::Subgraph sg1;
    sg1.ops = {1};
    sg1.tensors_to_retain = {};
    sg1.granularity = {.width = 128, .height = 128, .depth = 1};
    sg1.traversal_order = std::nullopt;
    sg1.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg0, sg1}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);
    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    // sg0: read t0 + write t1 = 327.68
    // sg1: read t2 + write t3 = 327.68
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[0].subgraph_latency, 327.68, 1e-6);
    EXPECT_NEAR(std::get<0>(estimated.value()).subgraphs[1].subgraph_latency, 327.68, 1e-6);
    EXPECT_NEAR(std::get<1>(estimated.value()), 655.36, 1e-6);
}

TEST(CostModelTest, CacheHitIsFetched_StatsIncrease) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem, true);

    auto first = cost_model.estimate(solution);
    ASSERT_TRUE(first.ok()) << first.status().message();
    auto stats_after_first = cost_model.cache_stats();
    EXPECT_EQ(stats_after_first.hits, 0);
    EXPECT_EQ(stats_after_first.misses, 1);
    EXPECT_EQ(stats_after_first.estimate_subgraph_calls, 1);

    auto second = cost_model.estimate(solution);
    ASSERT_TRUE(second.ok()) << second.status().message();
    auto stats_after_second = cost_model.cache_stats();

    EXPECT_EQ(stats_after_second.hits, 1);
    EXPECT_EQ(stats_after_second.misses, 1);
    EXPECT_EQ(stats_after_second.estimate_subgraph_calls, 1);

    EXPECT_NEAR(std::get<1>(first.value()), std::get<1>(second.value()), 1e-6);
}

TEST(CostModelTest, CacheStatsDisabled_DefaultNoTracking) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 100;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 1};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto first = cost_model.estimate(solution);
    ASSERT_TRUE(first.ok()) << first.status().message();
    auto second = cost_model.estimate(solution);
    ASSERT_TRUE(second.ok()) << second.status().message();

    auto stats = cost_model.cache_stats();
    EXPECT_EQ(stats.hits, 0);
    EXPECT_EQ(stats.misses, 0);
    EXPECT_EQ(stats.estimate_subgraph_calls, 0);
}

TEST(CostModelTest, RasterTraversalReuseIsEnabledByDefault) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1500},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 64, .height = 64, .depth = 128};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    std::unique_ptr<mlsys::CostModel> cost_model_ptr = std::make_unique<mlsys::CostModel>(problem);
    auto estimated = cost_model_ptr->estimate(solution);

    ASSERT_TRUE(estimated.ok()) << estimated.status().message();

    EXPECT_NEAR(std::get<1>(estimated.value()), 7096.0, 1e-6);
    EXPECT_LT(std::get<1>(estimated.value()), 8192.0);
}

TEST(CostModelTest, RejectsGranularityDepthAboveNative) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 256, .height = 128},
        {.width = 128, .height = 256},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 1500},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::Subgraph sg;
    sg.ops = {0};
    sg.tensors_to_retain = {};
    sg.granularity = {.width = 128, .height = 128, .depth = 256};
    sg.traversal_order = std::nullopt;
    sg.subgraph_latency = 0.0;

    mlsys::Solution solution{.subgraphs = {sg}};

    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(solution);

    ASSERT_FALSE(estimated.ok());
    EXPECT_NE(std::string(estimated.status().message()).find("native granularity"),
              std::string::npos);
}

} // namespace
