#include "tiler.h"
#include <string>
#include <algorithm>

namespace mlsys {

auto Tiler::tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    
    for (auto& sg : tiled_solution.subgraphs) {
        sg.granularity.width = problem.native_granularity.width;
        sg.granularity.height = problem.native_granularity.height;
        sg.granularity.depth = 1;
        for (size_t op_idx : sg.ops) {
            if (problem.ops[op_idx].op_type == "MatMul") {
                size_t lhs_idx = problem.ops[op_idx].inputs[0];
                sg.granularity.depth = std::max(sg.granularity.depth, problem.tensors[lhs_idx].width);
            }
        }
    }
    
    while (true) {
        auto eval_status = Evaluate(problem, tiled_solution);
        if (eval_status.ok()) {
            break;
        }
        
        std::string err_msg = std::string(eval_status.status().message());
        if (err_msg.find("[Fast Memory Capacity Exceeded]") != std::string::npos) {
            size_t pos = err_msg.find_last_of(' ');
            if (pos != std::string::npos) {
                int sg_idx = std::stoi(err_msg.substr(pos + 1));
                auto& sg = tiled_solution.subgraphs[sg_idx];
                
                int64_t w = sg.granularity.width;
                int64_t h = sg.granularity.height;
                int64_t k = sg.granularity.depth;
                
                int64_t max_dim = std::max({w, h, k});
                if (max_dim <= 1) {
                    return absl::ResourceExhaustedError("Cannot fit working set even at minimum tile size");
                }
                
                // Prioritize shrinking the largest dimension
                if (k == max_dim) {
                    sg.granularity.depth = (k + 1) / 2;
                } else if (w == max_dim) {
                    sg.granularity.width = (w + 1) / 2;
                } else {
                    sg.granularity.height = (h + 1) / 2;
                }
            } else {
                return eval_status.status();
            }
        } else {
            return eval_status.status();
        }
    }
    
    return tiled_solution;
}

} // namespace mlsys