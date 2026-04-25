#include <gtest/gtest.h>

#include "mlsys.h"
#include "solution_writer.h"
#include "solver.h"
#include "test_utils.h"

#include "absl/status/status.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

auto UniqueTempDir() -> std::filesystem::path {
    auto const now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir =
        std::filesystem::path("/tmp") /
        ("mlsys_solution_writer_test_" + std::to_string(now) + "_" +
         std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::filesystem::create_directories(dir);
    return dir;
}

auto MakeSolution(double latency, size_t op_idx = 0) -> mlsys::Solution {
    mlsys::Subgraph subgraph;
    subgraph.ops = {op_idx};
    subgraph.tensors_to_retain = {};
    subgraph.granularity = {.width = 1, .height = 1, .depth = 1};
    subgraph.traversal_order = std::nullopt;
    subgraph.subgraph_latency = latency;

    mlsys::Solution solution;
    solution.subgraphs = {subgraph};
    return solution;
}

auto ReadSingleLatency(const std::filesystem::path& path) -> absl::StatusOr<double> {
    auto solution = mlsys::ReadSolution(path.string());
    if (!solution.ok()) {
        return solution.status();
    }
    if (solution.value().subgraphs.empty()) {
        return absl::InternalError("Expected at least one subgraph");
    }
    return solution.value().subgraphs[0].subgraph_latency;
}

} // namespace

TEST(SolutionWriterTest, AtomicWriteProducesReadableSolutionJson) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";

    auto status = mlsys::WriteSolutionAtomically(MakeSolution(7.0), output_path.string());
    ASSERT_TRUE(status.ok()) << status.message();

    auto latency = ReadSingleLatency(output_path);
    ASSERT_TRUE(latency.ok()) << latency.status().message();
    EXPECT_DOUBLE_EQ(latency.value(), 7.0);

    std::filesystem::remove_all(dir);
}

TEST(SolutionWriterTest, WorseCostDoesNotReplaceBufferedBest) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";

    mlsys::AnytimeSolutionWriter writer(output_path.string());
    ASSERT_TRUE(writer.Start().ok());
    ASSERT_TRUE(writer.PublishAndWaitForWrite(MakeSolution(10.0), 10.0).ok());
    ASSERT_TRUE(writer.PublishIfBetter(MakeSolution(20.0), 20.0).ok());
    ASSERT_TRUE(writer.StopAndFlush().ok());

    auto latency = ReadSingleLatency(output_path);
    ASSERT_TRUE(latency.ok()) << latency.status().message();
    EXPECT_DOUBLE_EQ(latency.value(), 10.0);

    std::filesystem::remove_all(dir);
}

TEST(SolutionWriterTest, BetterCostReplacesBufferedBestOnFlush) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";

    mlsys::AnytimeSolutionWriter writer(output_path.string());
    ASSERT_TRUE(writer.Start().ok());
    ASSERT_TRUE(writer.PublishAndWaitForWrite(MakeSolution(10.0), 10.0).ok());
    ASSERT_TRUE(writer.PublishIfBetter(MakeSolution(5.0), 5.0).ok());
    ASSERT_TRUE(writer.StopAndFlush().ok());

    auto latency = ReadSingleLatency(output_path);
    ASSERT_TRUE(latency.ok()) << latency.status().message();
    EXPECT_DOUBLE_EQ(latency.value(), 5.0);

    std::filesystem::remove_all(dir);
}

TEST(SolutionWriterTest, DelayedWritePublishesLatestBestSnapshot) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";

    mlsys::AnytimeSolutionWriter writer(output_path.string());
    ASSERT_TRUE(writer.Start().ok());
    ASSERT_TRUE(writer.PublishAndWaitForWrite(MakeSolution(10.0), 10.0).ok());
    ASSERT_TRUE(writer.PublishIfBetter(MakeSolution(8.0), 8.0).ok());

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto latency_before_delay = ReadSingleLatency(output_path);
    ASSERT_TRUE(latency_before_delay.ok()) << latency_before_delay.status().message();
    EXPECT_DOUBLE_EQ(latency_before_delay.value(), 10.0);

    ASSERT_TRUE(writer.PublishIfBetter(MakeSolution(4.0), 4.0).ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(350));

    auto latency_after_delay = ReadSingleLatency(output_path);
    ASSERT_TRUE(latency_after_delay.ok()) << latency_after_delay.status().message();
    EXPECT_DOUBLE_EQ(latency_after_delay.value(), 4.0);
    ASSERT_TRUE(writer.StopAndFlush().ok());

    std::filesystem::remove_all(dir);
}

TEST(SolutionWriterTest, StopFlushesDirtySolutionBeforeDelayExpires) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";

    mlsys::AnytimeSolutionWriter writer(output_path.string());
    ASSERT_TRUE(writer.Start().ok());
    ASSERT_TRUE(writer.PublishAndWaitForWrite(MakeSolution(10.0), 10.0).ok());
    ASSERT_TRUE(writer.PublishIfBetter(MakeSolution(3.0), 3.0).ok());
    ASSERT_TRUE(writer.StopAndFlush().ok());

    auto latency = ReadSingleLatency(output_path);
    ASSERT_TRUE(latency.ok()) << latency.status().message();
    EXPECT_DOUBLE_EQ(latency.value(), 3.0);

    std::filesystem::remove_all(dir);
}

#if defined(__unix__) || defined(__APPLE__)
TEST(SolutionWriterTest, GreedyBaselineSurvivesSigtermAfterBaselineWrite) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";
    std::filesystem::path const marker_path = dir / "baseline.marker";
    mlsys::Problem problem = mlsys::test::MakeSinglePointwiseProblem(64, 64);

    pid_t const child = fork();
    ASSERT_GE(child, 0);

    if (child == 0) {
        setenv("MLSYS_GREEDY_TEST_PAUSE_AFTER_BASELINE", marker_path.string().c_str(), 1);
        mlsys::GreedySolver solver(output_path.string());
        auto solution = solver.solve(problem);
        _exit(solution.ok() ? 0 : 2);
    }

    bool baseline_ready = false;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        auto solution = mlsys::ReadSolution(output_path.string());
        if (std::filesystem::exists(marker_path) && solution.ok()) {
            baseline_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (!baseline_ready) {
        kill(child, SIGTERM);
        int ignored_status = 0;
        waitpid(child, &ignored_status, 0);
    }
    ASSERT_TRUE(baseline_ready) << "Timed out waiting for baseline solution file";

    ASSERT_EQ(kill(child, SIGTERM), 0);
    int child_status = 0;
    ASSERT_EQ(waitpid(child, &child_status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(child_status));
    EXPECT_EQ(WTERMSIG(child_status), SIGTERM);

    auto solution = mlsys::ReadSolution(output_path.string());
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    auto eval = mlsys::Evaluate(problem, solution.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();

    std::filesystem::remove_all(dir);
}

TEST(SolutionWriterTest, HeuristicBaselineSurvivesSigtermAfterBaselineWrite) {
    std::filesystem::path const dir = UniqueTempDir();
    std::filesystem::path const output_path = dir / "solution.json";
    std::filesystem::path const marker_path = dir / "baseline.marker";
    mlsys::Problem problem = mlsys::test::MakeSinglePointwiseProblem(64, 64);

    pid_t const child = fork();
    ASSERT_GE(child, 0);

    if (child == 0) {
        setenv("MLSYS_TEST_PAUSE_AFTER_BASELINE", marker_path.string().c_str(), 1);
        mlsys::HeuristicSolver solver(output_path.string());
        auto solution = solver.solve(problem);
        _exit(solution.ok() ? 0 : 2);
    }

    bool baseline_ready = false;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        auto solution = mlsys::ReadSolution(output_path.string());
        if (std::filesystem::exists(marker_path) && solution.ok()) {
            baseline_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (!baseline_ready) {
        kill(child, SIGTERM);
        int ignored_status = 0;
        waitpid(child, &ignored_status, 0);
    }
    ASSERT_TRUE(baseline_ready) << "Timed out waiting for baseline solution file";

    ASSERT_EQ(kill(child, SIGTERM), 0);
    int child_status = 0;
    ASSERT_EQ(waitpid(child, &child_status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(child_status));
    EXPECT_EQ(WTERMSIG(child_status), SIGTERM);

    auto solution = mlsys::ReadSolution(output_path.string());
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    auto eval = mlsys::Evaluate(problem, solution.value());
    ASSERT_TRUE(eval.ok()) << eval.status().message();

    std::filesystem::remove_all(dir);
}
#endif
