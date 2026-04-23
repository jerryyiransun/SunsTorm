#include "mlsys.h"

#include <iostream>
#include <string>

namespace {

void PrintUsage() {
    std::cerr << "Usage: ./evaluate_debug <path_to_problem.json> <path_to_solution.json>\n";
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    if (argc != 3) {
        PrintUsage();
        return 1;
    }

    const std::string problem_path = argv[1];
    const std::string solution_path = argv[2];

    std::cout << "Problem JSON: " << problem_path << "\n";
    std::cout << "Solution JSON: " << solution_path << "\n";

    auto problem_status = mlsys::ReadProblem(problem_path);
    if (problem_status.ok() == false) {
        std::cerr << "Error reading problem: " << problem_status.status().message() << "\n";
        return 1;
    }
    const mlsys::Problem problem = problem_status.value();

    auto solution_status = mlsys::ReadSolution(solution_path);
    if (solution_status.ok() == false) {
        std::cerr << "Error reading solution: " << solution_status.status().message() << "\n";
        return 1;
    }
    const mlsys::Solution& solution = solution_status.value();

    auto eval_status = mlsys::Evaluate(problem, solution);
    if (eval_status.ok()) {
        std::cout << "Evaluation successful.\n";
        std::cout << "Total Latency: " << eval_status.value() << "\n";
        return 0;
    }

    std::cerr << "Error evaluating solution: " << eval_status.status().message() << "\n";
    return 1;
}
