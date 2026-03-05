#include "mlsys.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>
#include <map>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

using namespace absl;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

namespace mlsys {

StatusOr<Problem> ReadProblem(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return absl::NotFoundError("File not found: " + filename);
    }

    json data;
    try {
        data = json::parse(file);
    } catch (const json::parse_error& e) {
        return absl::InvalidArgumentError(std::string("JSON parse error: ") + e.what());
    }

    Problem problem;

    try {
        // Parse Tensors
        const auto& widths = data.at("widths");
        const auto& heights = data.at("heights");
        
        if (widths.size() != heights.size()) {
            return absl::InvalidArgumentError("widths and heights arrays must be the same size");
        }
        
        for (size_t i = 0; i < widths.size(); ++i) {
            problem.tensors.push_back(Tensor{
                widths[i].get<Width>(), 
                heights[i].get<Height>()
            });
        }

        // Parse Operations
        const auto& inputs = data.at("inputs");
        const auto& outputs = data.at("outputs");
        const auto& base_costs = data.at("base_costs");
        const auto& op_types = data.at("op_types");

        if (inputs.size() != outputs.size() || 
            inputs.size() != base_costs.size() || 
            inputs.size() != op_types.size()) {
            return absl::InvalidArgumentError("Op arrays must all be the same size");
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            Op op;
            op.op_type = op_types[i].get<OpType>();
            op.inputs = inputs[i].get<Inputs>();
            op.outputs = outputs[i].get<Outputs>();
            op.base_cost = base_costs[i].get<BaseCost>();
            problem.ops.push_back(op);
        }

        // Parse System Constraints
        problem.fast_memory_capacity = data.at("fast_memory_capacity").get<FastMemoryCapacity>();
        problem.slow_memory_bandwidth = data.at("slow_memory_bandwidth").get<SlowMemoryBandwidth>();

        // Parse Granularity 
        const auto& gran = data.at("native_granularity");
        if (gran.size() >= 2) {
            problem.native_granularity.width = gran[0].get<Width>();
            problem.native_granularity.height = gran[1].get<Height>();
            problem.native_granularity.depth = 1;
        } else {
            return absl::InvalidArgumentError("native_granularity must have at least 2 elements");
        }

    } catch (const json::type_error& e) {
        return absl::InvalidArgumentError(std::string("JSON type error: ") + e.what());
    } catch (const json::out_of_range& e) {
        return absl::InvalidArgumentError(std::string("JSON missing expected field: ") + e.what());
    }

    return problem;
}

StatusOr<Solution> ReadSolution(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return absl::NotFoundError("File not found: " + filename);
    }

    json data;
    try {
        data = json::parse(file);
    } catch (const json::parse_error& e) {
        return absl::InvalidArgumentError(std::string("JSON parse error: ") + e.what());
    }

    Solution solution;

    try {
        // Parse Subgraphs
        const auto& subgraphs = data.at("subgraphs");
        const auto& granularities = data.at("granularities");
        const auto& tensors_to_retain = data.at("tensors_to_retain");
        const auto& traversal_orders = data.at("traversal_orders");
        const auto& subgraph_latencies = data.at("subgraph_latencies");
        
        // Check all same size
        if (subgraphs.size() != granularities.size() || 
            subgraphs.size() != tensors_to_retain.size() || 
            subgraphs.size() != traversal_orders.size() || 
            subgraphs.size() != subgraph_latencies.size()) {
            return absl::InvalidArgumentError("Subgraph arrays must all be the same size");
        }

        // Check all the subgraph latency are positive
        for (size_t i = 0; i < subgraph_latencies.size(); i++) {
            if (subgraph_latencies[i].get<SubgraphLatency>() < 0) {
                return absl::InvalidArgumentError("Subgraph latency must be positive");
            }
        }

        for (size_t i = 0; i < subgraphs.size(); i++) {
            Subgraph subgraph;
            subgraph.ops = subgraphs[i].get<Outputs>();
            subgraph.tensors_to_retain = tensors_to_retain[i].get<Outputs>();

            // Parse granularity
            const auto& gran = granularities[i];
            if (gran.size() >= 2) {
                subgraph.granularity.width = gran[0].get<Width>();
                subgraph.granularity.height = gran[1].get<Height>();
                subgraph.granularity.depth = 1;
            } else {
                return absl::InvalidArgumentError("native_granularity must have at least 2 elements");
            }

            const auto& traversal_order = traversal_orders[i];
            if (traversal_order != nullptr) {
                subgraph.traversal_order = traversal_order.get<TraversalOrder>();
            }

            subgraph.subgraph_latency = subgraph_latencies[i].get<SubgraphLatency>();
            solution.subgraphs.push_back(subgraph);
        }

    } catch (const json::type_error& e) {
        return absl::InvalidArgumentError(std::string("JSON type error: ") + e.what());
    } catch (const json::out_of_range& e) {
        return absl::InvalidArgumentError(std::string("JSON missing expected field: ") + e.what());
    }

    return solution;
}

StatusOr<TotalLatency> Evaluate(const Problem& problem, const Solution& solution) {
    // 1. Setup the inputs_satisfied map
    // Initially all inputs are satisfied, then we traverse the operations in order and mark the outputs as NOT satisfied
    std::vector<bool> inputs_satisfied(problem.tensors.size(), true);
    for (const auto& op : problem.ops) {
        for (size_t out : op.outputs) {
            inputs_satisfied[out] = false;
        }
    }

    TotalLatency total_latency = 0.0;
    std::set<size_t> currently_retained;

    // 2. Check each subgraph for dependencies
    for (size_t i = 0; i < solution.subgraphs.size(); ++i) {
        const auto& sg = solution.subgraphs[i];

        // Ensure ops inputs are satisfied
        for (size_t op_idx : sg.ops) {
            if (op_idx >= problem.ops.size()) {
                return absl::InvalidArgumentError("[Invalid Op Index] Invalid op index in subgraph");
            }
            const auto& op = problem.ops[op_idx];
            for (size_t in : op.inputs) {
                if (!inputs_satisfied[in]) {
                    return absl::FailedPreconditionError("[Unmet Dependency] Dependency not met for tensor " + std::to_string(in));
                }
            }
            // Mark outputs of operation as satisfied
            // Technically there is only ever 1 output per operation, but we use a loop for consistency
            for (size_t out : op.outputs) {
                inputs_satisfied[out] = true;
            }
        }

        // 1. Check that the granularity doesn't overflow the fast memory.
        int64_t memory_for_retained = 0;
        for (size_t t_idx : currently_retained) {
            if (t_idx < problem.tensors.size()) {
                const auto& t = problem.tensors[t_idx];
                memory_for_retained += t.width * t.height;
            }
        }

        // Identify tiled tensors:
        // "If there are multiple matmul ops within the same subgraph only tile the last 2 matmul inputs"
        // Interpreted as: the 2 inputs of the LAST MatMul in the subgraph are tiled.
        std::set<size_t> tiled_tensors;
        int last_matmul_idx = -1;
        int num_matmuls = 0;
        for (int op_i = (int)sg.ops.size() - 1; op_i >= 0; --op_i) {
            const auto& op = problem.ops[sg.ops[op_i]];
            if (op.op_type == "MatMul") {
                if (last_matmul_idx == -1) {
                    last_matmul_idx = sg.ops[op_i];
                }
                num_matmuls++;
            }
        }

        if (last_matmul_idx != -1) {
            // Tile the inputs of the last matmul
            const auto& matmul_op = problem.ops[last_matmul_idx];
            for (size_t in_idx : matmul_op.inputs) {
                tiled_tensors.insert(in_idx);
            }
        } else {
            // For pointwise only subgraphs, tile everything? 
            // The examples suggest pointwise is tiled by w x h
            for (size_t op_idx : sg.ops) {
                const auto& op = problem.ops[op_idx];
                for (size_t in : op.inputs) tiled_tensors.insert(in);
                for (size_t out : op.outputs) tiled_tensors.insert(out);
            }
        }

        // Collect all tensors used in this subgraph
        std::set<size_t> subgraph_tensors;
        for (size_t op_idx : sg.ops) {
            const auto& op = problem.ops[op_idx];
            for (size_t in : op.inputs) subgraph_tensors.insert(in);
            for (size_t out : op.outputs) subgraph_tensors.insert(out);
        }

        // Calculate max memory footprint required
        int64_t max_memory_required = memory_for_retained;
        
        for (size_t t_idx : subgraph_tensors) {
            // Skip currently retained tensors (already counted)
            if (currently_retained.count(t_idx)) continue;

            // Check if it's the output of the last operation
            bool is_last_op_output = false;
            if (!sg.ops.empty()) {
                for (size_t out : problem.ops[sg.ops.back()].outputs) {
                    if (out == t_idx) is_last_op_output = true;
                }
            }

            // Check if it was produced in this subgraph (intermediate / ephemeral)
            bool produced_in_sg = false;
            for (size_t op_idx : sg.ops) {
                for (size_t out : problem.ops[op_idx].outputs) {
                    if (out == t_idx) produced_in_sg = true;
                }
            }

            if (is_last_op_output) {
                // "Specifically the output of the last operation in a subgraph
                // is output stationary so it must also always stay in fast memory within a substep"
                // For pure pointwise subgraphs, the input sharing means the output can reuse input space.
                if (num_matmuls == 0) {
                    bool has_external_input = false;
                    for (size_t in_idx : subgraph_tensors) {
                        if (!currently_retained.count(in_idx)) {
                            bool prod = false;
                            for (size_t sub_op : sg.ops) {
                                for (size_t o : problem.ops[sub_op].outputs) if (o == in_idx) prod = true;
                            }
                            if (!prod) has_external_input = true;
                        }
                    }
                    if (!has_external_input) {
                        max_memory_required += sg.granularity.width * sg.granularity.height;
                    }
                } else {
                    max_memory_required += sg.granularity.width * sg.granularity.height;
                }
            } else if (produced_in_sg) {
                // Intermediate/ephemeral tensor will just be written and accumulated within the output fast memory space
                // No need to add extra memory
            } else if (tiled_tensors.count(t_idx)) {    // current tensor is a tiled input
                // Tiled input
                // For a MatMul: LHS is (h x k), RHS is (k x w). Pointwise is (h x w).
                if (num_matmuls > 0) {    // current subgraph has a matmul op
                    const auto& matmul_op = problem.ops[last_matmul_idx];
                    if (matmul_op.inputs.size() == 2) {
                        if (t_idx == matmul_op.inputs[0]) { // LHS
                            max_memory_required += sg.granularity.height * sg.granularity.depth;
                        } else if (t_idx == matmul_op.inputs[1]) { // RHS
                            max_memory_required += sg.granularity.width * sg.granularity.depth;
                        } else {        // should not happen
                            assert(false);
                        }
                    } else {        // should not happen
                        assert(false);
                    }
                } else {
                    // should have tiling even for pointwise ops
                    max_memory_required += sg.granularity.width * sg.granularity.height;
                }
            } else {
                // Fully loaded size
                const auto& t = problem.tensors[t_idx];
                max_memory_required += t.width * t.height;
            }
        }

        if (max_memory_required > problem.fast_memory_capacity) {
            return absl::ResourceExhaustedError("[Fast Memory Capacity Exceeded] Fast memory capacity exceeded in subgraph " + std::to_string(i));
        }

        // 3. Return totalLatency
        total_latency += sg.subgraph_latency;

        // Update currently_retained for next subgraph
        // Note: Technically redundant since we never append to currently_retained
        // but good for clarity
        currently_retained.clear();
        for (size_t t_idx : sg.tensors_to_retain) {
            currently_retained.insert(t_idx);
        }
    }

    // 4. Check all outputs were produced using inputs_satisfied
    std::set<size_t> all_outputs;
    for (const auto& op : problem.ops) {
        for (size_t out : op.outputs) {
            all_outputs.insert(out);
        }
    }
    for (size_t out : all_outputs) {
        if (!inputs_satisfied[out]) {
            return absl::FailedPreconditionError("[Missed Output] Output " + std::to_string(out) + " was not produced");
        }
    }

    return total_latency;
}

Status WriteSolution(const Solution& solution, const std::string& filename) {
    ordered_json j;
    
    j["subgraphs"] = ordered_json::array();
    j["granularities"] = ordered_json::array();
    j["tensors_to_retain"] = ordered_json::array();
    j["traversal_orders"] = ordered_json::array();
    j["subgraph_latencies"] = ordered_json::array();

    for (const auto& sg : solution.subgraphs) {
        j["subgraphs"].push_back(sg.ops);
        j["granularities"].push_back({
            sg.granularity.width, 
            sg.granularity.height, 
            sg.granularity.depth
        });
        j["tensors_to_retain"].push_back(sg.tensors_to_retain);
        
        if (sg.traversal_order.has_value()) {
            j["traversal_orders"].push_back(sg.traversal_order.value());
        } else {
            j["traversal_orders"].push_back(nullptr);
        }
        
        j["subgraph_latencies"].push_back(sg.subgraph_latency);
    }

    // Write to file with a 2-space indentation for readability
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        return absl::NotFoundError("Failed to open output file: " + filename);
    }
    
    out_file << j.dump(2) << std::endl;
    out_file.close();

    return absl::OkStatus();
}

}  // namespace mlsys
