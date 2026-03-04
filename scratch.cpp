#include <iostream>
#include "mlsys.h"
#include "solver.h"

int main() {
    std::string solution_path = "examples/example-1-output-C.json";
    std::cout << "Reading solution from: " << solution_path << "\n";

    auto solution_status = mlsys::ReadSolution(solution_path);
    if (!solution_status.ok()) {
        std::cerr << "Error reading solution: " << solution_status.status().message() << "\n";
        return 1;
    }

    mlsys::Solution solution = solution_status.value();
    std::cout << "Successfully loaded solution with " << solution.subgraphs.size() << " subgraphs.\n";

    for (size_t i = 0; i < solution.subgraphs.size(); ++i) {
        const auto& sg = solution.subgraphs[i];
        std::cout << "Subgraph " << i << ":\n";
        std::cout << "  Ops: [";
        for (size_t j = 0; j < sg.ops.size(); ++j) {
            std::cout << sg.ops[j] << (j == sg.ops.size() - 1 ? "" : ", ");
        }
        std::cout << "]\n";
        
        std::cout << "  Tensors to Retain: [";
        for (size_t j = 0; j < sg.tensors_to_retain.size(); ++j) {
            std::cout << sg.tensors_to_retain[j] << (j == sg.tensors_to_retain.size() - 1 ? "" : ", ");
        }
        std::cout << "]\n";

        if (sg.traversal_order.has_value()) {
            std::cout << "  Traversal Order: [";
            const auto& order = sg.traversal_order.value();
            for (size_t j = 0; j < order.size(); ++j) {
                std::cout << order[j] << (j == order.size() - 1 ? "" : ", ");
            }
            std::cout << "]\n";
        } else {
            std::cout << "  Traversal Order: [None]\n";
        }

        std::cout << "  Granularity: " << sg.granularity.width << "x" << sg.granularity.height << "x" << sg.granularity.depth << "\n";
        std::cout << "  Latency: " << sg.subgraph_latency << "\n";
    }

    return 0;
}
