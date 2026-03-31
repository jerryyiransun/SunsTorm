#include "fuser.h"
#include <iostream>
#include <set>
#include <vector>

namespace mlsys {

auto BruteForceFuser::fuse(const Problem& problem) -> StatusOr<std::vector<Solution>> {
    std::vector<Solution> results;
    size_t num_ops = problem.ops.size();
    if (num_ops == 0) {
        return results;
    }

    // Step 1: Iterate through all contiguous partitions of the ops.
    // We can imagine placing an optional "divider" between each adjacent operation.
    // For N operations, there are N-1 potential dividers. 
    // We represent these choices as a bitmask ranging from 0 to (2^(N-1) - 1).
    // A bit value of 1 means "split here" (end current subgraph, start a new one).
    // A bit value of 0 means "fuse" (keep the operation in the current subgraph).
    size_t num_partitions = 1ULL << (num_ops - 1);
    
    for (size_t mask = 0; mask < num_partitions; ++mask) {
        // Collect operations grouped into separate subgraphs based on the mask
        std::vector<std::vector<size_t>> subgraphs_ops;
        std::vector<size_t> current_ops;
        
        for (size_t i = 0; i < num_ops; ++i) {
            current_ops.push_back(i);
            // If this is the last op, OR if the mask bit at this position says to split:
            if (i == num_ops - 1 || (((mask >> i) & 1) != 0)) {
                subgraphs_ops.push_back(current_ops);
                current_ops.clear();
            }
        }
        
        // Step 2: For each generated subgraph, find all candidate tensors to retain.
        // A tensor can only be retained by a subgraph if it's either an input to or 
        // an output from that subgraph's operations.
        std::vector<std::vector<size_t>> all_cands;
        for (const auto& ops_list : subgraphs_ops) {
            std::set<size_t> cands;
            for (size_t op_idx : ops_list) {
                // Collect unique inputs
                for (size_t in_idx : problem.ops[op_idx].inputs) {
                    cands.insert(in_idx);
                }
                // Collect unique outputs
                for (size_t out_idx : problem.ops[op_idx].outputs) {
                    cands.insert(out_idx);
                }
            }
            all_cands.push_back(std::vector<size_t>(cands.begin(), cands.end()));
        }
        
        // Step 3: Determine the total number of tensor retention permutations for this partition structure.
        // If subgraph 0 has 3 candidate tensors, it has 2^3 = 8 retention possibilities.
        // We multiply the subsets across all subgraphs to get the total combinations.
        size_t num_subgraphs = subgraphs_ops.size();
        std::vector<size_t> num_subsets;
        size_t total_combinations = 1;
        for (const auto& cands : all_cands) {
            size_t subsets = 1ULL << cands.size();
            num_subsets.push_back(subsets);
            total_combinations *= subsets;
        }
        
        // Step 4: Iterate through every valid combination of retained tensors across all subgraphs
        for (size_t combo = 0; combo < total_combinations; ++combo) {
            Solution sol;
            size_t current_combo = combo;
            
            // Unpack the large 'combo' integer into individual bitmasks for each subgraph
            for (size_t sg_idx = 0; sg_idx < num_subgraphs; ++sg_idx) {
                size_t subset_mask = current_combo % num_subsets[sg_idx];
                current_combo /= num_subsets[sg_idx];
                
                Subgraph subgraph;
                subgraph.ops = subgraphs_ops[sg_idx];
                
                // Add the tensors instructed to be retained by this specific subset mask
                for (size_t i = 0; i < all_cands[sg_idx].size(); ++i) {
                    if (((subset_mask >> i) & 1) != 0) {
                        subgraph.tensors_to_retain.push_back(all_cands[sg_idx][i]);
                    }
                }
                sol.subgraphs.push_back(subgraph);
            }
            results.push_back(sol);

#ifdef DEBUG
            std::cout << "[DEBUG] Solution " << results.size() << " generated:\n";
            for (size_t dbg_idx = 0; dbg_idx < sol.subgraphs.size(); ++dbg_idx) {
                std::cout << "[DEBUG]  Subgraph " << dbg_idx << " ops: [";
                for (size_t o : sol.subgraphs[dbg_idx].ops) std::cout << o << " ";
                std::cout << "], retains: [";
                for (size_t t : sol.subgraphs[dbg_idx].tensors_to_retain) std::cout << t << " ";
                std::cout << "]\n";
            }
#endif
        }
    }
    
    return results;
}

} // namespace mlsys