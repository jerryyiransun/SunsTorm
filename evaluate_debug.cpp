#include "mlsys.h"
#include "solver.h"

#include <iostream>
#include <string>

namespace {

constexpr char kDefaultTestName[] = "Benchmark1_WithTestJson_DebugOutput";

void PrintUsage() {
    std::cerr << "Usage: ./evaluate_debug <path_to_problem.json> [test_name]\n";
    std::cerr << "  test_name controls output file name: <test_name>.json\n";
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    if (argc < 2 || argc > 3) {
        PrintUsage();
        return 1;
    }

    const std::string problem_path = argv[1];
    const std::string test_name = (argc == 3) ? argv[2] : kDefaultTestName;
    if (test_name.empty()) {
        std::cerr << "Error: test_name must not be empty.\n";
        return 1;
    }
    const std::string output_path = test_name + ".json";

    std::cout << "Problem JSON: " << problem_path << "\n";
    std::cout << "Solution JSON: " << output_path << "\n";

    auto problem_status = mlsys::ReadProblem(problem_path);
    if (problem_status.ok() == false) {
        std::cerr << "Error reading problem: " << problem_status.status().message() << "\n";
        return 1;
    }
    const mlsys::Problem problem = problem_status.value();

    mlsys::GreedySolver solver;
    auto solution_status = solver.solve(problem);
    if (solution_status.ok() == false) {
        std::cerr << "Error solving problem: " << solution_status.status().message() << "\n";
        return 1;
    }
    const mlsys::Solution& solution = solution_status.value();

    auto write_status = mlsys::WriteSolution(solution, output_path);
    if (write_status.ok() == false) {
        std::cerr << "Error writing solution: " << write_status.message() << "\n";
        return 1;
    }

    auto eval_status = mlsys::Evaluate(problem, solution);
    if (eval_status.ok()) {
        std::cout << "Evaluation successful.\n";
        std::cout << "Total Latency: " << eval_status.value() << "\n";
        return 0;
    }

    std::cerr << "Error evaluating solution: " << eval_status.status().message() << "\n";
    return 1;
}
