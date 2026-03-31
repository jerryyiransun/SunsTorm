#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"

#include "cost_model.h"
#include "fuser.h"
#include "mlsys.h"
#include "solver.h"
#include "tiler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

using namespace std;

namespace mlsys {

auto BaseSolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    Solution solution;

    // The baseline solver evaluates each operation individually in topological order.
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        const Op& op = problem.ops[i];

        Width const w_out = problem.tensors[op.outputs[0]].width;
        Height const h_out = problem.tensors[op.outputs[0]].height;

        Width const compute_w = problem.native_granularity.width;
        Height const compute_h = problem.native_granularity.height;

        Width w = min(w_out, compute_w);
        Height h = min(h_out, compute_h);

        Depth k = 1;
        Depth k_full = 1;

        if (op.op_type == "MatMul") {
            k = max(problem.tensors[op.inputs[0]].width, problem.tensors[op.inputs[1]].height);
            k_full = k;
        }

        // Iteratively reduce tile size until the working set fits in fast memory
        while (true) {
            FastMemoryCapacity current_working_set_size = 0;
            if (op.op_type == "MatMul") {
                // MatMul: Working set = LHS slice + RHS slice + Output slice
                current_working_set_size = (w * h) + (w * k) + (k * h);
            } else {
                // Pointwise: In-place buffer aliasing. The output overwrites an input buffer.
                current_working_set_size = max(op.inputs.size(), op.outputs.size()) * (w * h);
            }

#ifdef DEBUG
            cout << "***DEBUG*** Op " << i << ": Trying tile size (w=" << w << ", h=" << h
                 << ", k=" << k << ") with working set size " << current_working_set_size << "\n";
#endif

            if (current_working_set_size <= problem.fast_memory_capacity) {
                break; // Found a working set size that fits in fast memory
            }

            int64_t const max_dim = max({w, h, k});
            if (max_dim <= 1) {
                // Cannot reduce further, hardware constraints are mathematically impossible
                return absl::InvalidArgumentError(
                    "Baseline solver cannot fit working set of op " + to_string(i) +
                    " into fast memory even at minimum tile size. Working set size: " +
                    to_string(current_working_set_size) +
                    ", Fast memory capacity: " + to_string(problem.fast_memory_capacity));
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

        SubgraphLatency const compute_time = op.base_cost;

        SubgraphLatency memory_time = 0;

        if (op.op_type == "MatMul") {
            memory_time +=
                ((w * h) + (w * k) + (k * h)) / static_cast<double>(problem.slow_memory_bandwidth);
        } else {
            memory_time += ((op.inputs.size() * w * h) + (op.outputs.size() * w * h)) /
                           static_cast<double>(problem.slow_memory_bandwidth);
        }

        Subgraph sg;
        sg.granularity = {.width = w, .height = h, .depth = k};
        sg.ops.push_back(i);
        sg.tensors_to_retain = {};
        sg.subgraph_latency = total_steps * max(compute_time, memory_time);
        sg.traversal_order = nullopt;

        solution.subgraphs.push_back(sg);
    }

    return solution;
}

auto BruteForceSolver::solve(const Problem& problem) -> absl::StatusOr<Solution> {
    unique_ptr<BruteForceFuser> fuser = make_unique<BruteForceFuser>();
    unique_ptr<Tiler> tiler = make_unique<Tiler>();
    unique_ptr<CostModel> cost_model = make_unique<CostModel>(problem);

    auto fusion_plans = fuser->fuse(problem);
    if (!fusion_plans.ok()) {
        return fusion_plans.status();
    }

    if (fusion_plans.value().empty()) {
        return absl::InternalError("Fuser returned no fusion plans");
    }

    tuple<Solution, SubgraphLatency> optimal_plan;

    for (const auto& plan : fusion_plans.value()) {
        auto tiled_plan = tiler->tile(problem, plan);
        if (!tiled_plan.ok()) {
            // tile() returns an error when no valid tiling is available
            continue;
        }

        auto solu = cost_model->estimate(tiled_plan.value());

        if (optimal_plan == tuple<Solution, SubgraphLatency>() ||
            get<1>(solu.value()) < get<1>(optimal_plan)) {
            optimal_plan = solu.value();
        }
    }

    if (optimal_plan == tuple<Solution, SubgraphLatency>()) {
        return absl::InternalError("No valid plans found by fuser and tiler");
    }

    return get<0>(optimal_plan);
}

} // namespace mlsys
