#include <iostream>
#include <string>
#include "mlsys.h"

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
    auto problem_status = mlsys::ReadProblem(input_path);
    if (!problem_status.ok()) {
        std::cerr << "Error reading input: " << problem_status.status().message() << "\n";
        return 1;
    }
    mlsys::Problem problem = problem_status.value();
    // Debug print the problem
    std::cout << problem.tensors.size() << " tensors and " << problem.ops.size() << " operations loaded.\n";

    // Scheduling logic
    // mlsys::Solution solution = Solve(problem);

    // Write the output

    std::cout << "Done.\n";
    return 0;
}