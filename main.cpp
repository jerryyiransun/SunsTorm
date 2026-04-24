#include "mlsys.h"
#include "solver.h"

#include <iostream>
#include <memory>
#include <string>

using namespace mlsys;

auto main(int argc, char* argv[]) -> int {
    if (argc != 3) {
        std::cerr << "Usage: ./mlsys <path_to_input.json> <path_to_output.json>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    auto problem_status = ReadProblem(input_path);
    if (!problem_status.ok()) {
        std::cerr << "Error reading input: " << problem_status.status().message() << "\n";
        return 1;
    }
    const Problem& problem = problem_status.value();

    std::unique_ptr<Solver> solver = std::make_unique<HeuristicSolver>();
    auto solution_status = solver->solve(problem);
    if (!solution_status.ok()) {
        std::cerr << "Error solving problem: " << solution_status.status().message() << "\n";
        return 1;
    }
    const Solution& solution = solution_status.value();

    // #ifdef DEBUG
    auto eval_status = Evaluate(problem, solution);
    if (eval_status.ok()) {
        std::cout << "Evaluation successful.\n";
        std::cout << "Total Latency: " << eval_status.value() << "\n";
    } else {
        std::cerr << "Error evaluating solution: " << eval_status.status().message() << "\n";
    }
    // #endif // DEBUG

    auto write_status = WriteSolutionAtomically(solution, output_path);
    if (!write_status.ok()) {
        std::cerr << "Error writing output: " << write_status.message() << "\n";
        return 1;
    }

    return 0;
}
