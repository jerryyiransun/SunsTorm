#include "mlsys.h"
#include "nlohmann/json.hpp"
#include "solver.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace mlsys;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./mlsys <path_to_input.json> <path_to_output.json>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::cout << "Starting MLSys run...\n";
    std::cout << "Input file: " << input_path << "\n";
    std::cout << "Output file: " << output_path << "\n";

    // Read the problem
    auto problem_status = ReadProblem(input_path);
    if (!problem_status.ok()) {
        std::cerr << "Error reading input: " << problem_status.status().message() << "\n";
        return 1;
    }
    Problem problem = problem_status.value();

#ifdef DEBUG
    std::cout << "***DEBUG*** " << problem.tensors.size() << " tensors and " << problem.ops.size()
              << " operations loaded.\n";
#endif

    // Scheduling logic
    std::unique_ptr<Solver> solver = std::make_unique<BaseSolver>();
    auto solution_status = solver->solve(problem);
    if (!solution_status.ok()) {
        std::cerr << "Error solving problem: " << solution_status.status().message() << "\n";
        return 1;
    }
    Solution solution = solution_status.value();

#ifdef DEBUG
    auto eval_status = Evaluate(problem, solution);
    if (eval_status.ok()) {
        std::cout << "Evaluation successful.\n";
        std::cout << "Total Latency: " << eval_status.value() << "\n";
    } else {
        std::cerr << "Error evaluating solution: " << eval_status.status().message() << "\n";
    }
#endif

    // Write the output
    auto write_status = WriteSolution(solution, output_path);
    if (!write_status.ok()) {
        std::cerr << "Error writing output: " << write_status.message() << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}