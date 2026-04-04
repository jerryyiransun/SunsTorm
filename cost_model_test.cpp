#include <memory>

#include <gtest/gtest.h>

#include "cost_model.h"
#include "mlsys.h"
#include "test_utils.h"

namespace {

using mlsys::test::ExpectExampleMatches;

TEST(TileTest, ComputeNonOverlappingArea_Empty) {
    EXPECT_EQ(mlsys::Tile::compute_non_overlapping_area({}), 0);
}

TEST(TileTest, ComputeNonOverlappingArea_FullOverlapSingleTensor) {
    std::vector<mlsys::Tile> tiles = {
        {.tensor_idx = 0, .x0 = 0, .x1 = 10, .y0 = 0, .y1 = 10},
        {.tensor_idx = 0, .x0 = 0, .x1 = 10, .y0 = 0, .y1 = 10},
    };
    EXPECT_EQ(mlsys::Tile::compute_non_overlapping_area(tiles), 100);
}

TEST(TileTest, ComputeNonOverlappingArea_PartialOverlapSingleTensor) {
    std::vector<mlsys::Tile> tiles = {
        {.tensor_idx = 0, .x0 = 0, .x1 = 10, .y0 = 0, .y1 = 10},
        {.tensor_idx = 0, .x0 = 5, .x1 = 15, .y0 = 0, .y1 = 10},
    };
    EXPECT_EQ(mlsys::Tile::compute_non_overlapping_area(tiles), 150);
}

TEST(TileTest, ComputeNonOverlappingArea_EdgeTouchNoOverlap) {
    std::vector<mlsys::Tile> tiles = {
        {.tensor_idx = 0, .x0 = 0, .x1 = 10, .y0 = 0, .y1 = 10},
        {.tensor_idx = 0, .x0 = 10, .x1 = 20, .y0 = 0, .y1 = 10},
    };
    EXPECT_EQ(mlsys::Tile::compute_non_overlapping_area(tiles), 200);
}

TEST(TileTest, ComputeNonOverlappingArea_DifferentTensorsDoNotOverlap) {
    std::vector<mlsys::Tile> tiles = {
        {.tensor_idx = 0, .x0 = 0, .x1 = 10, .y0 = 0, .y1 = 10},
        {.tensor_idx = 1, .x0 = 0, .x1 = 10, .y0 = 0, .y1 = 10},
    };
    EXPECT_EQ(mlsys::Tile::compute_non_overlapping_area(tiles), 200);
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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

TEST(CostModelTest, RetainingEphemeralTensorHasNoCrossSubgraphEffect) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {2}, .outputs = {3}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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

    EXPECT_NEAR(std::get<1>(estimated_no_retain.value()), std::get<1>(estimated_retain.value()),
                1e-6);
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

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

} // namespace
