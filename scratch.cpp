#include <iostream>
#include "mlsys.h"
#include "solver.h"

#include <vector>

void RunTest(const std::string& input_path, const std::string& output_path, const std::string& expected_result) {
    std::cout << "Testing " << input_path << " with " << output_path << "\n";
    
    auto problem_status = mlsys::ReadProblem(input_path);
    if (!problem_status.ok()) {
        std::cout << "  [FAIL] Failed to read problem: " << problem_status.status().message() << "\n";
        return;
    }

    auto solution_status = mlsys::ReadSolution(output_path);
    if (!solution_status.ok()) {
        std::cout << "  [FAIL] Failed to read solution: " << solution_status.status().message() << "\n";
        return;
    }

    auto eval_status = mlsys::Evaluate(problem_status.value(), solution_status.value());
    if (eval_status.ok() && expected_result == "Pass") {
        std::cout << "  [PASS] Total Latency: " << eval_status.value() << "\n";
    } else if (!eval_status.ok() && eval_status.status().message().find(expected_result) != std::string::npos){
        std::cout << "  [PASS] Expected " << expected_result << ", Got " << eval_status.status().message() << "\n";
    } else {
        std::cout << "  [FAIL] Expected " << expected_result << ", Got " << eval_status.status().message() << "\n";
    }
}

int main() {
    std::vector<std::vector<std::string>> tests = {
        {"examples/example-1-input.json", "examples/example-1-output-A.json", "Pass"},
        {"examples/example-1-input.json", "examples/example-1-output-B.json", "Pass"},
        {"examples/example-1-input.json", "examples/example-1-output-C.json", "Pass"},
        {"examples/example-1-input.json", "examples/example-1-output-F-dependency.json", "[Unmet Dependency]"},
        {"examples/example-1-input.json", "examples/example-1-output-F-invalid-op.json", "[Invalid Op Index]"},
        {"examples/example-5-input.json", "examples/example-5-output-F-capacity.json", "[Fast Memory Capacity Exceeded]"},
        {"examples/example-1-input.json", "examples/example-1-output-F-missed-output.json", "[Missed Output]"},
        
        // I don't see proper output JSON files for examples 2-5 in the directory, 
        // presumably I should only test what exists in the examples dir. Wait, I'll check what files exist.
    };

    for (const auto& test : tests) {
        RunTest(test[0], test[1], test[2]);
    }

    return 0;
}
