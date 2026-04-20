#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "fuser.h"
#include "mlsys.h"

namespace {

auto MakeGreedyConfig(int search_depth, int beam_width) -> mlsys::GreedyFuserConfig {
    return mlsys::GreedyFuserConfig{
        .search_depth = search_depth,
        .beam_width = beam_width,
        .topk_failure_penalty = 0.8,
    };
}

auto FormatSolution(const mlsys::Solution& solution) -> std::string {
    std::string out;
    for (const auto& subgraph : solution.subgraphs) {
        out += "[ops:";
        for (size_t op_idx : subgraph.ops) {
            out += std::to_string(op_idx);
            out += ",";
        }
        out += " retain:";
        for (size_t tensor_idx : subgraph.tensors_to_retain) {
            out += std::to_string(tensor_idx);
            out += ",";
        }
        out += "]";
    }
    return out;
}

void ExpectGreedySolutionValid(const mlsys::Problem& problem, const mlsys::Solution& solution) {
    auto eval = mlsys::Evaluate(problem, solution);
    ASSERT_TRUE(eval.ok()) << eval.status().message();
}

auto SubgraphProducesTensor(const mlsys::Problem& problem, const mlsys::Subgraph& subgraph,
                            size_t tensor_idx) -> bool {
    for (size_t op_idx : subgraph.ops) {
        if (problem.ops[op_idx].outputs[0] == tensor_idx) {
            return true;
        }
    }
    return false;
}

auto SubgraphConsumesTensor(const mlsys::Problem& problem, const mlsys::Subgraph& subgraph,
                            size_t tensor_idx) -> bool {
    for (size_t op_idx : subgraph.ops) {
        for (size_t input_tensor : problem.ops[op_idx].inputs) {
            if (input_tensor == tensor_idx) {
                return true;
            }
        }
    }
    return false;
}

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

TEST(FuserTest, GreedyFindsDiamondRecomputationSchedule) {
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

    mlsys::GreedyFuser fuser(MakeGreedyConfig(2, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    bool found_left_branch = false;
    bool found_right_branch = false;
    for (const auto& subgraph : solution->subgraphs) {
        if (subgraph.ops == std::vector<size_t>({0, 1})) {
            found_left_branch = true;
        }
        if (subgraph.ops == std::vector<size_t>({0, 2})) {
            found_right_branch = true;
        }
    }

    EXPECT_TRUE(found_left_branch) << FormatSolution(*solution);
    EXPECT_TRUE(found_right_branch) << FormatSolution(*solution);
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, PrefuseFusesThroughUnaryChainThenStopsAtMultiInputConsumer) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, {.width = 128, .height = 128}, {.width = 128, .height = 128},
        {.width = 128, .height = 128}, {.width = 128, .height = 128}, {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {2}, .outputs = {3}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {3, 4}, .outputs = {5}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 2u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({0, 1, 2}));
    EXPECT_EQ(solution->subgraphs[1].ops, std::vector<size_t>({3}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, PrefuseDoesNotTreatMultiInputPointwiseConsumerAsFree) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {1, 2}, .outputs = {3}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 2u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({0}));
    EXPECT_EQ(solution->subgraphs[1].ops, std::vector<size_t>({1}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, PrefuseSkipsDirectMergeWhenProducerOutputHasAnotherConsumer) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {3}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 3u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({0}));
    EXPECT_EQ(solution->subgraphs[1].ops, std::vector<size_t>({1}));
    EXPECT_EQ(solution->subgraphs[2].ops, std::vector<size_t>({2}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, PrefuseDoesNotHardFuseIntoMatMulConsumer) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 32, .height = 32},
        {.width = 32, .height = 32},
        {.width = 16, .height = 32},
        {.width = 16, .height = 32},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
        {.op_type = "MatMul", .inputs = {1, 2}, .outputs = {3}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 32, .height = 32, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 2u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({0}));
    EXPECT_EQ(solution->subgraphs[1].ops, std::vector<size_t>({1}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, PrefuseFusesMatMulProducerIntoUnaryPointwiseConsumer) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 32, .height = 32},
        {.width = 32, .height = 32},
        {.width = 32, .height = 32},
        {.width = 32, .height = 32},
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {0, 1}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {2}, .outputs = {3}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 32, .height = 32, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 1u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({0, 1}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, DirectMergeRemovesProducerSubgraphWhenSafe) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {1, 2}, .outputs = {3}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 1u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({0, 1}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, AvoidsInvalidDestructiveMergeWhenProducerIsStillNeeded) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, {.width = 128, .height = 128}, {.width = 32, .height = 128},
        {.width = 32, .height = 128},  {.width = 32, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "MatMul", .inputs = {1, 2}, .outputs = {3}, .base_cost = 1000},
        {.op_type = "MatMul", .inputs = {1, 3}, .outputs = {4}, .base_cost = 1000},
    };
    problem.fast_memory_capacity = 8'000;
    problem.slow_memory_bandwidth = 1;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    std::optional<size_t> producer_subgraph_idx;
    std::optional<size_t> intermediate_consumer_idx;
    std::optional<size_t> later_consumer_idx;
    for (size_t sg_idx = 0; sg_idx < solution->subgraphs.size(); ++sg_idx) {
        const auto& subgraph = solution->subgraphs[sg_idx];
        if (!producer_subgraph_idx.has_value() &&
            std::find(subgraph.ops.begin(), subgraph.ops.end(), 0) != subgraph.ops.end()) {
            producer_subgraph_idx = sg_idx;
        }
        if (!intermediate_consumer_idx.has_value() &&
            std::find(subgraph.ops.begin(), subgraph.ops.end(), 1) != subgraph.ops.end()) {
            intermediate_consumer_idx = sg_idx;
        }
        if (!later_consumer_idx.has_value() &&
            std::find(subgraph.ops.begin(), subgraph.ops.end(), 2) != subgraph.ops.end()) {
            later_consumer_idx = sg_idx;
        }
    }

    ASSERT_TRUE(producer_subgraph_idx.has_value()) << FormatSolution(*solution);
    ASSERT_TRUE(intermediate_consumer_idx.has_value()) << FormatSolution(*solution);
    ASSERT_TRUE(later_consumer_idx.has_value()) << FormatSolution(*solution);
    EXPECT_LT(producer_subgraph_idx.value(), intermediate_consumer_idx.value())
        << FormatSolution(*solution);
    EXPECT_LT(intermediate_consumer_idx.value(), later_consumer_idx.value())
        << FormatSolution(*solution);
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, RetainCarriesTensorAcrossUnrelatedIntermediateSubgraph) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, {.width = 128, .height = 128}, {.width = 32, .height = 128},
        {.width = 32, .height = 128},  {.width = 128, .height = 128}, {.width = 128, .height = 128},
        {.width = 32, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1'000'000},
        {.op_type = "MatMul", .inputs = {1, 2}, .outputs = {3}, .base_cost = 1000},
        {.op_type = "Pointwise", .inputs = {4}, .outputs = {5}, .base_cost = 100},
        {.op_type = "MatMul", .inputs = {1, 3}, .outputs = {6}, .base_cost = 1000},
    };
    problem.fast_memory_capacity = 100'000;
    problem.slow_memory_bandwidth = 1;
    problem.native_granularity = {.width = 64, .height = 64, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();
    ExpectGreedySolutionValid(problem, *solution);

    bool retained_through_unrelated_subgraph = false;
    for (const auto& subgraph : solution->subgraphs) {
        bool const retains_tensor =
            std::find(subgraph.tensors_to_retain.begin(), subgraph.tensors_to_retain.end(), 1) !=
            subgraph.tensors_to_retain.end();
        if (!retains_tensor) {
            continue;
        }

        bool const touches_tensor = SubgraphProducesTensor(problem, subgraph, 1) ||
                                    SubgraphConsumesTensor(problem, subgraph, 1);
        if (!touches_tensor) {
            retained_through_unrelated_subgraph = true;
            break;
        }
    }

    EXPECT_TRUE(retained_through_unrelated_subgraph) << FormatSolution(*solution);
}

TEST(FuserTest, UsesExplicitTopologicalOrderForGeneratedSolution) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
        {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 1u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({1, 0}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, BeamWidthOneStillFindsRecomputationSchedule) {
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

    mlsys::GreedyFuser fuser(MakeGreedyConfig(2, 1));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    bool found_left_branch = false;
    bool found_right_branch = false;
    for (const auto& subgraph : solution->subgraphs) {
        if (subgraph.ops == std::vector<size_t>({0, 1})) {
            found_left_branch = true;
        }
        if (subgraph.ops == std::vector<size_t>({0, 2})) {
            found_right_branch = true;
        }
    }

    EXPECT_TRUE(found_left_branch) << FormatSolution(*solution);
    EXPECT_TRUE(found_right_branch) << FormatSolution(*solution);
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, BeamWidthOnePrioritizesHigherScoredFuseCandidate) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 256, .height = 256}, // 0
        {.width = 256, .height = 256}, // 1
        {.width = 256, .height = 256}, // 2
        {.width = 64, .height = 64},   // 3
        {.width = 64, .height = 64},   // 4
        {.width = 64, .height = 64},   // 5
        {.width = 256, .height = 256}, // 6
        {.width = 64, .height = 64},   // 7
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {1, 6}, .outputs = {2}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {3}, .outputs = {4}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {4, 7}, .outputs = {5}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 1};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 1));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();
    ExpectGreedySolutionValid(problem, *solution);

    bool fused_large_pair = false;
    for (const auto& subgraph : solution->subgraphs) {
        if (subgraph.ops == std::vector<size_t>({0, 1})) {
            fused_large_pair = true;
            break;
        }
    }

    EXPECT_TRUE(fused_large_pair) << FormatSolution(*solution);
}

} // namespace
