#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "cost_model.h"
#include "fuser.h"
#include "mlsys.h"
#include "tiler.h"

namespace mlsys {
auto ScoreGreedyPotentialForTesting(GreedyPotentialScorer potential_scorer, int64_t saved_elements,
                                    size_t shared_tensor_count, bool is_retain, size_t retain_span,
                                    int64_t retained_elements, int64_t fast_memory_capacity,
                                    int64_t direct_boundary_elements = 0) -> int64_t;
}

namespace {

auto MakeGreedyConfig(int search_depth, int beam_width,
                      mlsys::GreedyPotentialScorer potential_scorer =
                          mlsys::GreedyPotentialScorer::kMemoryTrafficDensity)
    -> mlsys::GreedyFuserConfig {
    return mlsys::GreedyFuserConfig{
        .search_depth = search_depth,
        .beam_width = beam_width,
        .topk_failure_penalty = 0.8,
        .potential_scorer = potential_scorer,
    };
}

struct GreedyRunResult {
    mlsys::Solution solution;
    mlsys::TotalLatency latency = 0.0;
};

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

auto RunGreedyWithScorer(const mlsys::Problem& problem,
                         mlsys::GreedyPotentialScorer potential_scorer, int search_depth,
                         int beam_width) -> GreedyRunResult {
    mlsys::GreedyFuser fuser(MakeGreedyConfig(search_depth, beam_width, potential_scorer));
    auto solution = fuser.fuse(problem);
    EXPECT_TRUE(solution.ok()) << solution.status().message();
    if (!solution.ok()) {
        return {};
    }

    auto latency = mlsys::Evaluate(problem, solution.value());
    EXPECT_TRUE(latency.ok()) << latency.status().message();
    if (!latency.ok()) {
        return GreedyRunResult{.solution = solution.value(), .latency = 0.0};
    }
    return GreedyRunResult{.solution = solution.value(), .latency = latency.value()};
}

auto MakeSolution(const std::vector<std::vector<size_t>>& subgraph_ops) -> mlsys::Solution {
    mlsys::Solution solution;
    for (const auto& ops : subgraph_ops) {
        mlsys::Subgraph subgraph;
        subgraph.ops = ops;
        subgraph.tensors_to_retain = {};
        subgraph.traversal_order = std::nullopt;
        subgraph.subgraph_latency = 0.0;
        solution.subgraphs.push_back(subgraph);
    }
    return solution;
}

auto EvaluateTiledLatency(const mlsys::Problem& problem, const mlsys::Solution& solution)
    -> mlsys::TotalLatency {
    mlsys::GreedyTiler tiler;
    auto tiled = tiler.tile(problem, solution);
    EXPECT_TRUE(tiled.ok()) << tiled.status().message();
    if (!tiled.ok()) {
        return 0.0;
    }
    mlsys::CostModel cost_model(problem);
    auto estimated = cost_model.estimate(tiled.value());
    EXPECT_TRUE(estimated.ok()) << estimated.status().message();
    return estimated.ok() ? std::get<1>(estimated.value()) : 0.0;
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

auto CountOpOccurrences(const mlsys::Solution& solution, size_t op_idx) -> size_t {
    size_t count = 0;
    for (const auto& subgraph : solution.subgraphs) {
        for (size_t subgraph_op_idx : subgraph.ops) {
            if (subgraph_op_idx == op_idx) {
                ++count;
            }
        }
    }
    return count;
}

auto CountRetainedTensorEntries(const mlsys::Solution& solution) -> size_t {
    size_t count = 0;
    for (const auto& subgraph : solution.subgraphs) {
        count += subgraph.tensors_to_retain.size();
    }
    return count;
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

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

TEST(FuserTest, GreedyAvoidsDiamondRecomputationSchedule) {
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(2, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 1), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 2), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 3), 1u) << FormatSolution(*solution);
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 1), 1u) << FormatSolution(*solution);
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 1), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 2), 1u) << FormatSolution(*solution);
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
    problem.native_granularity = {.width = 32, .height = 32, .depth = 32};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 1), 1u) << FormatSolution(*solution);
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, InitialPrefuseDoesNotFuseMatMulProducerIntoUnaryPointwiseConsumer) {
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
    problem.native_granularity = {.width = 32, .height = 32, .depth = 32};

    std::vector<mlsys::Solution> best_updates;
    mlsys::GreedyFuser fuser(
        MakeGreedyConfig(0, 8),
        [&](const mlsys::Solution& solution, mlsys::TotalLatency) -> absl::Status {
            best_updates.push_back(solution);
            return absl::OkStatus();
        });
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_FALSE(best_updates.empty());
    ASSERT_EQ(best_updates[0].subgraphs.size(), 2u);
    EXPECT_EQ(best_updates[0].subgraphs[0].ops, std::vector<size_t>({0}));
    EXPECT_EQ(best_updates[0].subgraphs[1].ops, std::vector<size_t>({1}));
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

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
    problem.native_granularity = {.width = 64, .height = 64, .depth = 64};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 1), 1u) << FormatSolution(*solution);
    EXPECT_EQ(CountOpOccurrences(*solution, 2), 1u) << FormatSolution(*solution);
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, RetainDoesNotPassThroughUnrelatedIntermediateSubgraph) {
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
    problem.native_granularity = {.width = 64, .height = 64, .depth = 64};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();
    ExpectGreedySolutionValid(problem, *solution);

    bool retained_through_unrelated_subgraph = false;
    bool retained_by_producer = false;
    for (const auto& subgraph : solution->subgraphs) {
        bool const retains_tensor =
            std::find(subgraph.tensors_to_retain.begin(), subgraph.tensors_to_retain.end(), 1) !=
            subgraph.tensors_to_retain.end();
        if (!retains_tensor) {
            continue;
        }

        if (SubgraphProducesTensor(problem, subgraph, 1)) {
            retained_by_producer = true;
        }

        bool const touches_tensor = SubgraphProducesTensor(problem, subgraph, 1) ||
                                    SubgraphConsumesTensor(problem, subgraph, 1);
        if (!touches_tensor) {
            retained_through_unrelated_subgraph = true;
            break;
        }
    }

    EXPECT_TRUE(retained_by_producer) << FormatSolution(*solution);
    EXPECT_FALSE(retained_through_unrelated_subgraph) << FormatSolution(*solution);
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 8));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    ASSERT_EQ(solution->subgraphs.size(), 1u);
    EXPECT_EQ(solution->subgraphs[0].ops, std::vector<size_t>({1, 0}));
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, BeamWidthOneAvoidsRecomputationSchedule) {
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
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(2, 1));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    EXPECT_EQ(CountOpOccurrences(*solution, 0), 1u) << FormatSolution(*solution);
    ExpectGreedySolutionValid(problem, *solution);
}

TEST(FuserTest, LookaheadPatienceResetsAfterEachImprovement) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 4096, .height = 4096}, // 0: shared first weight
        {.width = 4096, .height = 4096}, // 1: shared second weight
        {.width = 4096, .height = 128},  // 2: chain A input
        {.width = 4096, .height = 128},  // 3: chain B input
        {.width = 128, .height = 4096},  // 4: chain A final weight
        {.width = 128, .height = 4096},  // 5: chain B final weight
        {.width = 4096, .height = 128},  // 6: chain A first output
        {.width = 4096, .height = 128},  // 7: chain A second output
        {.width = 128, .height = 128},   // 8: chain A final output
        {.width = 4096, .height = 128},  // 9: chain B first output
        {.width = 4096, .height = 128},  // 10: chain B second output
        {.width = 128, .height = 128},   // 11: chain B final output
    };
    problem.ops = {
        {.op_type = "MatMul", .inputs = {2, 0}, .outputs = {6}, .base_cost = 5000},
        {.op_type = "MatMul", .inputs = {6, 1}, .outputs = {7}, .base_cost = 5000},
        {.op_type = "MatMul", .inputs = {7, 4}, .outputs = {8}, .base_cost = 5000},
        {.op_type = "MatMul", .inputs = {3, 0}, .outputs = {9}, .base_cost = 5000},
        {.op_type = "MatMul", .inputs = {9, 1}, .outputs = {10}, .base_cost = 5000},
        {.op_type = "MatMul", .inputs = {10, 5}, .outputs = {11}, .base_cost = 5000},
    };
    problem.fast_memory_capacity = 600'000;
    problem.slow_memory_bandwidth = 50;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(1, 1));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();
    ExpectGreedySolutionValid(problem, *solution);

    EXPECT_GT(CountRetainedTensorEntries(*solution), 1u) << FormatSolution(*solution);
}

TEST(FuserTest, ZeroLookaheadPrunesNonImprovingBranchesButKeepsImmediateImprovements) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 128, .height = 128}, {.width = 128, .height = 128}, {.width = 128, .height = 128},
        {.width = 128, .height = 128}, {.width = 128, .height = 128}, {.width = 128, .height = 128},
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {1, 2}, .outputs = {3}, .base_cost = 10},
        {.op_type = "Pointwise", .inputs = {3, 4}, .outputs = {5}, .base_cost = 10},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    mlsys::GreedyFuser fuser(MakeGreedyConfig(0, 4));
    auto solution = fuser.fuse(problem);
    ASSERT_TRUE(solution.ok()) << solution.status().message();
    ExpectGreedySolutionValid(problem, *solution);

    EXPECT_LT(solution->subgraphs.size(), 3u) << FormatSolution(*solution);
}

TEST(FuserTest, MemoryTrafficDensityRanksLowerLatencyMultiTensorFuseAboveLegacyAverage) {
    mlsys::Problem problem;
    problem.tensors = {
        {.width = 90, .height = 100},  // 0: multi-tensor chain input
        {.width = 90, .height = 100},  // 1: first shared tensor
        {.width = 90, .height = 100},  // 2: second shared tensor
        {.width = 90, .height = 100},  // 3: multi-tensor consumer output
        {.width = 100, .height = 100}, // 4: single-tensor producer input
        {.width = 100, .height = 100}, // 5: single larger shared tensor
        {.width = 100, .height = 100}, // 6: single-tensor consumer output
        {.width = 90, .height = 100},  // 7: external multi-tensor consumer input
        {.width = 100, .height = 100}, // 8: external single-tensor consumer input
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {1}, .outputs = {2}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {1, 2, 7}, .outputs = {3}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {4}, .outputs = {5}, .base_cost = 1},
        {.op_type = "Pointwise", .inputs = {5, 8}, .outputs = {6}, .base_cost = 1},
    };
    problem.fast_memory_capacity = 1'000'000;
    problem.slow_memory_bandwidth = 1;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    int64_t const single_tensor_saved = 2 * 100 * 100;
    int64_t const multi_tensor_saved = 2 * 2 * 90 * 100;
    EXPECT_GT(mlsys::ScoreGreedyPotentialForTesting(mlsys::GreedyPotentialScorer::kLegacyAverage,
                                                    single_tensor_saved, 1, false, 0, 0, 0),
              mlsys::ScoreGreedyPotentialForTesting(mlsys::GreedyPotentialScorer::kLegacyAverage,
                                                    multi_tensor_saved, 2, false, 0, 0, 0));
    EXPECT_LT(
        mlsys::ScoreGreedyPotentialForTesting(mlsys::GreedyPotentialScorer::kMemoryTrafficDensity,
                                              single_tensor_saved, 1, false, 0, 0, 0),
        mlsys::ScoreGreedyPotentialForTesting(mlsys::GreedyPotentialScorer::kMemoryTrafficDensity,
                                              multi_tensor_saved, 2, false, 0, 0, 0));

    mlsys::Solution legacy_ranked_solution = MakeSolution({{0}, {1}, {2}, {3, 4}});
    mlsys::Solution improved_ranked_solution = MakeSolution({{0, 1, 2}, {3}, {4}});
    EXPECT_GT(EvaluateTiledLatency(problem, legacy_ranked_solution),
              EvaluateTiledLatency(problem, improved_ranked_solution));
}

TEST(FuserTest, DirectFusionBoundaryPressureRanksSmallerWorkingSetHigher) {
    int64_t const saved_elements = 20'000;
    size_t const shared_tensor_count = 1;
    int64_t const fast_memory_capacity = 10'000;

    int64_t const small_boundary_score = mlsys::ScoreGreedyPotentialForTesting(
        mlsys::GreedyPotentialScorer::kMemoryTrafficDensity, saved_elements, shared_tensor_count,
        false, 0, 0, fast_memory_capacity, 10'000);
    int64_t const large_boundary_score = mlsys::ScoreGreedyPotentialForTesting(
        mlsys::GreedyPotentialScorer::kMemoryTrafficDensity, saved_elements, shared_tensor_count,
        false, 0, 0, fast_memory_capacity, 90'000);

    EXPECT_GT(small_boundary_score, large_boundary_score);
}

TEST(FuserTest, RetainPressureRanksShorterRetainAboveLegacyAverage) {
    int64_t const long_retain_elements = 11'000;
    int64_t const short_retain_elements = 10'000;
    int64_t const fast_memory_capacity = 35'000;

    EXPECT_GT(mlsys::ScoreGreedyPotentialForTesting(mlsys::GreedyPotentialScorer::kLegacyAverage,
                                                    long_retain_elements, 1, true, 8,
                                                    long_retain_elements, fast_memory_capacity),
              mlsys::ScoreGreedyPotentialForTesting(mlsys::GreedyPotentialScorer::kLegacyAverage,
                                                    short_retain_elements, 1, true, 1,
                                                    short_retain_elements, fast_memory_capacity));

    EXPECT_LT(mlsys::ScoreGreedyPotentialForTesting(
                  mlsys::GreedyPotentialScorer::kMemoryTrafficDensity, long_retain_elements, 1,
                  true, 8, long_retain_elements, fast_memory_capacity),
              mlsys::ScoreGreedyPotentialForTesting(
                  mlsys::GreedyPotentialScorer::kMemoryTrafficDensity, short_retain_elements, 1,
                  true, 1, short_retain_elements, fast_memory_capacity));
}

TEST(FuserTest, BeamWidthOneDirectPressurePrioritizesSmallerBoundaryFuse) {
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
        {.width = 256, .height = 256}, // 8
        {.width = 256, .height = 256}, // 9
        {.width = 256, .height = 256}, // 10
        {.width = 256, .height = 256}, // 11
        {.width = 256, .height = 256}, // 12
        {.width = 256, .height = 256}, // 13
        {.width = 256, .height = 256}, // 14
        {.width = 256, .height = 256}, // 15
        {.width = 256, .height = 256}, // 16
        {.width = 256, .height = 256}, // 17
        {.width = 256, .height = 256}, // 18
    };
    problem.ops = {
        {.op_type = "Pointwise", .inputs = {0}, .outputs = {1}, .base_cost = 100},
        {.op_type = "Pointwise",
         .inputs = {1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
         .outputs = {2},
         .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {3}, .outputs = {4}, .base_cost = 100},
        {.op_type = "Pointwise", .inputs = {4, 7}, .outputs = {5}, .base_cost = 100},
    };
    problem.fast_memory_capacity = 35'000;
    problem.slow_memory_bandwidth = 10;
    problem.native_granularity = {.width = 128, .height = 128, .depth = 128};

    GreedyRunResult legacy_result =
        RunGreedyWithScorer(problem, mlsys::GreedyPotentialScorer::kLegacyAverage, 1, 1);
    GreedyRunResult improved_result =
        RunGreedyWithScorer(problem, mlsys::GreedyPotentialScorer::kMemoryTrafficDensity, 1, 1);

    bool legacy_fused_large_pair = false;
    for (const auto& subgraph : legacy_result.solution.subgraphs) {
        if (subgraph.ops == std::vector<size_t>({0, 1})) {
            legacy_fused_large_pair = true;
            break;
        }
    }

    bool improved_fused_small_pair = false;
    for (const auto& subgraph : improved_result.solution.subgraphs) {
        if (subgraph.ops == std::vector<size_t>({2, 3})) {
            improved_fused_small_pair = true;
            break;
        }
    }

    EXPECT_TRUE(legacy_fused_large_pair) << FormatSolution(legacy_result.solution);
    EXPECT_TRUE(improved_fused_small_pair) << FormatSolution(improved_result.solution);
}

} // namespace
