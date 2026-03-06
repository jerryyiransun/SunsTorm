#include "solver.h"

namespace mlsys {

absl::StatusOr<Solution> BaseSolver::Solve(const Problem& problem) {
    Solution solution;

    // The baseline solver evaluates each operation individually in topological order.
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        const Op& op = problem.ops[i];

        Width w_out = problem.tensors[op.outputs[0]].width;
        Height h_out = problem.tensors[op.outputs[0]].height;

        Width compute_w = problem.native_granularity.width;
        Height compute_h = problem.native_granularity.height;

        Width w = std::min(w_out, compute_w);
        Height h = std::min(h_out, compute_h);

        Depth k = 1;
        Depth k_full = 1;
        
        if (op.op_type == "MatMul") {
            k = std::max(problem.tensors[op.inputs[0]].width, problem.tensors[op.inputs[1]].height);
            k_full = k;
        }

        // Iteratively reduce tile size until the working set fits in fast memory
        while (true) {
            FastMemoryCapacity current_working_set_size = 0;
            if (op.op_type == "MatMul") {
                // MatMul: Working set = LHS slice + RHS slice + Output slice
                current_working_set_size = w * h + w * k + k * h;
            } else {
                // Pointwise: In-place buffer aliasing. The output overwrites an input buffer.
                current_working_set_size = std::max(op.inputs.size(), op.outputs.size()) * (w * h);
            }

            #ifdef DEBUG
            std::cout << "***DEBUG*** Op " << i << ": Trying tile size (w=" << w << ", h=" << h << ", k=" << k << ") with working set size " << current_working_set_size << "\n";
            #endif

            if (current_working_set_size <= problem.fast_memory_capacity) {
                break;  // Found a working set size that fits in fast memory
            }

            int64_t max_dim = std::max({w, h, k});
            if (max_dim <= 1) {
                // Cannot reduce further, hardware constraints are mathematically impossible
                return absl::InvalidArgumentError(
                    "Baseline solver cannot fit working set of op " + std::to_string(i) +
                    " into fast memory even at minimum tile size. Working set size: " + std::to_string(current_working_set_size) +
                    ", Fast memory capacity: " + std::to_string(problem.fast_memory_capacity)
                );
            }

            if (op.op_type == "MatMul" && k == max_dim) {
                k = (k + 1) / 2; // Split-K tiling
            } else if (w == max_dim) {
                w = (w + 1) / 2; // Spatial Width tiling
            } else {
                h = (h + 1) / 2; // Spatial Height tiling
            }
        }

        auto spatial_tiles_w = (w_out + w - 1) / w;
        auto spatial_tiles_h = (h_out + h - 1) / h;
        auto k_steps = (op.op_type == "MatMul") ? (k_full + k - 1) / k : 1;
        
        auto total_steps = spatial_tiles_w * spatial_tiles_h * k_steps;

        SubgraphLatency compute_time = op.base_cost;

        SubgraphLatency memory_time = 0;

        if (op.op_type == "MatMul") {
            memory_time += (w * h + w * k + k * h) / static_cast<double>(problem.slow_memory_bandwidth);
        } else {
            memory_time += (op.inputs.size() * w * h + op.outputs.size() * w * h) / static_cast<double>(problem.slow_memory_bandwidth);
        }

        Subgraph sg;
        sg.granularity = {w, h, k};
        sg.ops.push_back(i);
        sg.tensors_to_retain = {};
        sg.subgraph_latency = total_steps * std::max(compute_time, memory_time);
        sg.traversal_order = std::nullopt;

        solution.subgraphs.push_back(sg);
    }

    return solution;
}



}  // namespace mlsys
