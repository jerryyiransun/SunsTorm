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

            const auto& gran = granularities[i];
            if (gran.size() >= 2) {
                subgraph.granularity.width = gran[0].get<Width>();
                subgraph.granularity.height = gran[1].get<Height>();
                if (gran.size() >= 3) {
                    subgraph.granularity.depth = gran[2].get<Depth>();
                } else {
                    subgraph.granularity.depth = 1;
                }
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
    std::vector<bool> inputs_satisfied(problem.tensors.size(), false);
    
    // 1. Identify "Problem Inputs" (tensors not produced by any op)
    std::vector<bool> is_produced(problem.tensors.size(), false);
    std::vector<int> producer_op(problem.tensors.size(), -1);
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        for (size_t out : problem.ops[i].outputs) {
            is_produced[out] = true;
            producer_op[out] = i;
        }
    }
    
    for (size_t i = 0; i < problem.tensors.size(); ++i) {
        if (!is_produced[i]) {
            inputs_satisfied[i] = true;
        }
    }

    TotalLatency total_latency = 0.0;
    std::set<size_t> prev_retained_tensors;
    std::vector<bool> op_executed(problem.ops.size(), false);

    for (size_t i = 0; i < solution.subgraphs.size(); ++i) {
        const auto& subgraph = solution.subgraphs[i];
        
        // --- Dependency Check ---
        for (size_t op_idx : subgraph.ops) {
            if (op_idx >= problem.ops.size()) {
                return absl::InvalidArgumentError("[Invalid Op Index] Invalid op index in subgraph");
            }
            op_executed[op_idx] = true;
            
            for (size_t in : problem.ops[op_idx].inputs) {
                if (!inputs_satisfied[in]) {
                    return absl::FailedPreconditionError("[Unmet Dependency] Dependency not met for tensor " + std::to_string(in));
                }
            }
            
            for (size_t out : problem.ops[op_idx].outputs) {
                inputs_satisfied[out] = true;
            }
        }
        
        // --- Fast Memory Capacity Check ---
        int64_t fast_memory_usage = 0;
        
        // Tensors retained from previous subgraph
        for (size_t t : prev_retained_tensors) {
            fast_memory_usage += problem.tensors[t].width * problem.tensors[t].height;
        }
        
        std::set<size_t> visited_tensors;
        std::vector<size_t> q;
        std::set<size_t> subgraph_produced;
        std::set<size_t> subgraph_consumed;
        
        for (size_t op_idx : subgraph.ops) {
            for (size_t out : problem.ops[op_idx].outputs) {
                subgraph_produced.insert(out);
            }
            for (size_t in : problem.ops[op_idx].inputs) {
                subgraph_consumed.insert(in);
            }
        }
        
        // Final outputs of this subgraph take granularity space
        for (size_t t : subgraph_produced) {
            bool is_final_output = (subgraph_consumed.find(t) == subgraph_consumed.end());
            // if we need to retain it for the next subgraph
            bool is_retained = (std::find(subgraph.tensors_to_retain.begin(), subgraph.tensors_to_retain.end(), t) != subgraph.tensors_to_retain.end());
            
            if ((is_final_output || is_retained) && visited_tensors.find(t) == visited_tensors.end()) {
                fast_memory_usage += subgraph.granularity.width * subgraph.granularity.height;
                visited_tensors.insert(t);
                q.push_back(t);
            }
        }

        // BFS traversal backwards
        size_t head = 0;
        std::set<size_t> ops_in_subgraph(subgraph.ops.begin(), subgraph.ops.end());

        while (head < q.size()) {
            size_t curr_t = q[head++];
            int prod_idx = producer_op[curr_t];
            
            // If tensor is not produced in this subgraph, stop crawling it
            if (prod_idx == -1 || ops_in_subgraph.find(prod_idx) == ops_in_subgraph.end()) {
                continue; 
            }
            
            const Op& op = problem.ops[prod_idx];
            
            if (op.op_type == "MatMul") {
                size_t out_t = op.outputs[0];
                
                // Process LHS
                size_t lhs_t = op.inputs[0];
                if (visited_tensors.find(lhs_t) == visited_tensors.end()) {
                    visited_tensors.insert(lhs_t);
                    if (prev_retained_tensors.find(lhs_t) == prev_retained_tensors.end()) {
                        fast_memory_usage += subgraph.granularity.depth * subgraph.granularity.height; // k * h
                    }
                    q.push_back(lhs_t);
                }
                
                // Process RHS
                size_t rhs_t = op.inputs[1];
                if (visited_tensors.find(rhs_t) == visited_tensors.end()) {
                    visited_tensors.insert(rhs_t);
                    if (prev_retained_tensors.find(rhs_t) == prev_retained_tensors.end()) {
                        fast_memory_usage += subgraph.granularity.width * subgraph.granularity.depth; // w * k
                    }
                    q.push_back(rhs_t);
                }
            } else if (op.op_type == "Pointwise") {
                for (size_t in_t : op.inputs) {
                    if (visited_tensors.find(in_t) == visited_tensors.end()) {
                        visited_tensors.insert(in_t);
                        // Pointwise inputs do not take extra fast memory space in the tile calculation
                        q.push_back(in_t);
                    }
                }
            }
        }
        
        if (fast_memory_usage > problem.fast_memory_capacity) {
            return absl::ResourceExhaustedError("[Fast Memory Capacity Exceeded] Fast memory capacity exceeded in subgraph " + std::to_string(i));
        }
        
        // --- Total Latency ---
        total_latency += subgraph.subgraph_latency;
        
        // Update retained tensors for next subgraph
        prev_retained_tensors.clear();
        for (size_t t : subgraph.tensors_to_retain) {
            prev_retained_tensors.insert(t);
        }
    }
    
    // --- All Operations Done Check ---
    for (size_t i = 0; i < problem.tensors.size(); ++i) {
        if (!inputs_satisfied[i]) {
            return absl::FailedPreconditionError("[Missed Output] Output " + std::to_string(i) + " was not produced");
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
