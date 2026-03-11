#include <string>

#include <gtest/gtest.h>

#include "mlsys.h"

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
    EXPECT_GT(result.value(), 0);
}

// asserts that evaluating a (problem, solution) pair fails with a message containing `expected_substr`.
void ExpectFail(const std::string& input_file, const std::string& output_file,
                const std::string& expected_substr) {
    auto problem = mlsys::ReadProblem(TestDataPath(input_file));
    ASSERT_TRUE(problem.ok()) << problem.status().message();

    auto solution = mlsys::ReadSolution(TestDataPath(output_file));
    ASSERT_TRUE(solution.ok()) << solution.status().message();

    auto result = mlsys::Evaluate(problem.value(), solution.value());
    ASSERT_FALSE(result.ok());
    EXPECT_NE(std::string(result.status().message()).find(expected_substr),
              std::string::npos)
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
    ExpectFail("example-1-input.json", "example-1-output-F-dependency.json",
               "[Unmet Dependency]");
}

TEST(EvaluateTest, Example1_Fail_InvalidOp) {
    ExpectFail("example-1-input.json", "example-1-output-F-invalid-op.json",
               "[Invalid Op Index]");
}

TEST(EvaluateTest, Example1_Fail_MissedOutput) {
    ExpectFail("example-1-input.json", "example-1-output-F-missed-output.json",
               "[Missed Output]");
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

}  // namespace
