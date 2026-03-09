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
    std::vector<bool> inputs_satisfied(problem.tensors.size(), true);
    
    // Identify tensors that are not produced by any op
    std::vector<int> producer_op(problem.tensors.size(), -1);   // producer_op[i] is the index of the op that produces tensor i
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        assert(problem.ops[i].outputs.size() == 1);
        size_t out = problem.ops[i].outputs[0];
        inputs_satisfied[out] = false;  // if the tensor is an output then it does not have its inputs satisfied before any operations are complete
        producer_op[out] = i;
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
            assert(subgraph.granularity.width <= problem.tensors[op_idx].width);
            assert(subgraph.granularity.height <= problem.tensors[op_idx].height);

            op_executed[op_idx] = true;
            
            for (size_t in : problem.ops[op_idx].inputs) {
                if (!inputs_satisfied[in]) {
                    return absl::FailedPreconditionError("[Unmet Dependency] Dependency not met for tensor " + std::to_string(in));
                }
            }
            
            size_t out = problem.ops[op_idx].outputs[0];
            inputs_satisfied[out] = true;
        }
        
        // --- Fast Memory Capacity Check ---
        FastMemoryCapacity fast_memory_usage = 0;
        
        #ifdef DEBUG
        std::cout << "\n[DEBUG] Subgraph " << i << " Fast Memory Capacity Check ---\n";
        #endif

        // Tensors retained from previous subgraph
        for (size_t t : prev_retained_tensors) {
            fast_memory_usage += problem.tensors[t].width * problem.tensors[t].height;
            #ifdef DEBUG
            std::cout << "[DEBUG] Tensor " << t << " (retained) takes " << problem.tensors[t].width * problem.tensors[t].height 
                      << " | total_mem=" << fast_memory_usage << "\n";
            #endif
        }
        
        std::set<size_t> visited_tensors; 
        // Queue of tensors to process in the form of (tensor_id, required_width, required_height, is_final_output_or_retained)
        std::vector<std::tuple<size_t, Width, Height, bool>> q;
        std::set<size_t> subgraph_produced;
        std::set<size_t> subgraph_consumed;
        
        for (size_t op_idx : subgraph.ops) {
            size_t out = problem.ops[op_idx].outputs[0];
            subgraph_produced.insert(out);

            for (size_t in : problem.ops[op_idx].inputs) {
                subgraph_consumed.insert(in);
            }
        }
        
        // Account the final tensors and to be retained tensors
        for (size_t t : subgraph_produced) {
            bool is_final_output = (subgraph_consumed.find(t) == subgraph_consumed.end());
            bool is_to_be_retained = (std::find(subgraph.tensors_to_retain.begin(), subgraph.tensors_to_retain.end(), t) != subgraph.tensors_to_retain.end());
            bool not_visited = (visited_tensors.find(t) == visited_tensors.end());
            
            if ((is_final_output || is_to_be_retained) && not_visited) {
                fast_memory_usage += subgraph.granularity.width * subgraph.granularity.height;
                visited_tensors.insert(t);
                q.push_back({t, subgraph.granularity.width, subgraph.granularity.height, true});

                #ifdef DEBUG
                std::cout << "[DEBUG] Tensor " << t 
                          << (is_final_output ? " (subgraph output)" : " (to be retained)")
                          << " takes " << subgraph.granularity.width * subgraph.granularity.height 
                          << " (w=" << subgraph.granularity.width << " h=" << subgraph.granularity.height << ")"
                          << " | total_mem=" << fast_memory_usage << "\n";
                #endif
            }
        }

        // BFS traversal backwards
        size_t head = 0;
        std::set<size_t> ops_in_subgraph(subgraph.ops.begin(), subgraph.ops.end());

        // Assumption: We do not need to be as granular as verifying overlapping spatial
        // Each tile loop we will have to load the necessary amount of data into fast memory whether or not it is retained(what we calculate for)
        while (head < q.size()) {
            auto [curr_t, req_w, req_h, is_final] = q[head++];
            int prod_idx = producer_op[curr_t];
            
            #ifdef DEBUG
            std::cout << "[DEBUG] Popped Tensor " << curr_t << " req_w=" << req_w << " req_h=" << req_h 
                      << " is_final=" << is_final << " from OP " << prod_idx << "\n";
            #endif
                      
            // If tensor is not produced in this subgraph, stop crawling it
            if (prod_idx == -1 || ops_in_subgraph.find(prod_idx) == ops_in_subgraph.end()) {
                continue; 
            }
            
            const Op& op = problem.ops[prod_idx];
            
            if (op.op_type == "MatMul") {
                assert(op.inputs.size() == 2);

                size_t lhs_t = op.inputs[0];
                size_t rhs_t = op.inputs[1];

                    int inner_k;
                if (is_final) {
                    inner_k = subgraph.granularity.depth;
                } else {
                    assert(problem.tensors[lhs_t].width == problem.tensors[rhs_t].height);
                    inner_k = std::max(problem.tensors[lhs_t].width, problem.tensors[rhs_t].height); // k is the full depth
                }
                
                #ifdef DEBUG
                std::cout << "[DEBUG] MatMul OP " << prod_idx << " uses inner_k=" << inner_k << "\n";
                #endif

                // Process LHS
                Width lhs_req_w = inner_k;
                Height lhs_req_h = req_h;
                
                if (visited_tensors.find(lhs_t) == visited_tensors.end()) {
                    visited_tensors.insert(lhs_t);
                    bool lhs_is_ephemeral = (subgraph_produced.find(lhs_t) != subgraph_produced.end());
                    bool lhs_is_retained = (std::find(subgraph.tensors_to_retain.begin(), subgraph.tensors_to_retain.end(), lhs_t) != subgraph.tensors_to_retain.end());
                    
                    if (!lhs_is_ephemeral && !lhs_is_retained) {
                        fast_memory_usage += lhs_req_w * lhs_req_h;
                    }
                    #ifdef DEBUG
                    std::cout << "[DEBUG] MatMul LHS Tensor " << lhs_t
                            << (lhs_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes" 
                                : lhs_is_retained ? " is RETAINED! Takes 0 bytes"
                                : " takes " + std::to_string(lhs_req_w * lhs_req_h) 
                                    + " (req_w=" + std::to_string(lhs_req_w) + " req_h=" + std::to_string(lhs_req_h) + ")")
                            << " | total_mem=" << fast_memory_usage << "\n";
                    #endif
                    q.push_back({lhs_t, lhs_req_w, lhs_req_h, false});
                }
                
                // Process RHS
                Width rhs_req_w = req_w;
                Height rhs_req_h = inner_k;
                
                if (visited_tensors.find(rhs_t) == visited_tensors.end()) {
                    visited_tensors.insert(rhs_t);
                    bool rhs_is_ephemeral = (subgraph_produced.find(rhs_t) != subgraph_produced.end());
                    bool rhs_is_retained = (std::find(subgraph.tensors_to_retain.begin(), subgraph.tensors_to_retain.end(), rhs_t) != subgraph.tensors_to_retain.end());
                    
                    if (!rhs_is_ephemeral && !rhs_is_retained) {
                        fast_memory_usage += rhs_req_w * rhs_req_h;
                    }
                    #ifdef DEBUG
                    std::cout << "[DEBUG] MatMul RHS Tensor " << rhs_t
                            << (rhs_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes"
                                : rhs_is_retained ? " is RETAINED! Takes 0 bytes"
                                : " takes " + std::to_string(rhs_req_w * rhs_req_h)
                                    + " (req_w=" + std::to_string(rhs_req_w) + " req_h=" + std::to_string(rhs_req_h) + ")")
                            << " | total_mem=" << fast_memory_usage << "\n";
                    #endif
                    q.push_back({rhs_t, rhs_req_w, rhs_req_h, false});
                }
            } else if (op.op_type == "Pointwise") {
                // When there is only 1 input the input tensor shares the same space as the output tensor
                // so we do not need to add any extra space for the input tensor
                if (op.inputs.size() == 1) {
                    size_t in_t = op.inputs[0];
                    if (visited_tensors.find(in_t) == visited_tensors.end()) {
                        visited_tensors.insert(in_t);
                        // Pointwise inputs do not take extra fast memory space in the tile calculation
                        q.push_back({in_t, req_w, req_h, false});
                    }
                } else {
                    // Otherwise the each of the input needs to be the same w,h size as the output
                    // so we need to add the space for each of the inputs
                    for (size_t in_t : op.inputs) {
                        if (visited_tensors.find(in_t) == visited_tensors.end()) {
                            visited_tensors.insert(in_t);
                            bool in_is_ephemeral = (subgraph_produced.find(in_t) != subgraph_produced.end());
                            bool in_is_retained = (prev_retained_tensors.find(in_t) != prev_retained_tensors.end());
                            
                            if (!in_is_ephemeral && !in_is_retained) {
                                fast_memory_usage += req_w * req_h;
                            }
                            #ifdef DEBUG
                            std::cout << "[DEBUG] Pointwise Input Tensor " << in_t
                                      << (in_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes"
                                          : in_is_retained ? " is RETAINED! Takes 0 bytes"
                                          : " takes " + std::to_string(req_w * req_h)
                                            + " (req_w=" + std::to_string(req_w) + " req_h=" + std::to_string(req_h) + ")")
                                      << " | total_mem=" << fast_memory_usage << "\n";
                            #endif
                            q.push_back({in_t, req_w, req_h, false});
                        }
                    }
                }
            }
        }
        
        #ifdef DEBUG
        std::cout << "[DEBUG] Subgraph " << i << " Fast Memory Total: " << fast_memory_usage << " / " << problem.fast_memory_capacity << "\n";
        #endif
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
