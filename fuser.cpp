#include "fuser.h"
#include <iostream>
#include <set>
#include <vector>

namespace mlsys {

namespace {

auto BuildProducerOpIndex(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer_op(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        for (size_t output_tensor : problem.ops[op_idx].outputs) {
            producer_op[output_tensor] = static_cast<int>(op_idx);
        }
    }
    return producer_op;
}

auto CanIncludeOp(const Problem& problem, const std::vector<int>& producer_op, size_t op_idx,
                  const std::vector<char>& covered_ops,
                  const std::vector<char>& selected_ops) -> bool {
    for (size_t input_tensor : problem.ops[op_idx].inputs) {
        int const producer = producer_op[input_tensor];
        if (producer >= 0 && !covered_ops[producer] && !selected_ops[producer]) {
            return false;
        }
    }
    return true;
}

void EnumerateSubgraphCandidates(const Problem& problem, const std::vector<int>& producer_op,
                                 const std::vector<char>& covered_ops, size_t op_idx,
                                 std::vector<char>& selected_ops,
                                 std::vector<size_t>& current_subgraph, bool introduces_new_op,
                                 std::vector<std::vector<size_t>>& candidates) {
    if (op_idx == problem.ops.size()) {
        if (!current_subgraph.empty() && introduces_new_op) {
            candidates.push_back(current_subgraph);
        }
        return;
    }

    EnumerateSubgraphCandidates(problem, producer_op, covered_ops, op_idx + 1, selected_ops,
                                current_subgraph, introduces_new_op, candidates);

    if (!CanIncludeOp(problem, producer_op, op_idx, covered_ops, selected_ops)) {
        return;
    }

    current_subgraph.push_back(op_idx);
    selected_ops[op_idx] = true;
    EnumerateSubgraphCandidates(problem, producer_op, covered_ops, op_idx + 1, selected_ops,
                                current_subgraph, introduces_new_op || !covered_ops[op_idx],
                                candidates);
    selected_ops[op_idx] = false;
    current_subgraph.pop_back();
}

auto BuildSchedulableSubgraphCandidates(const Problem& problem,
                                        const std::vector<int>& producer_op,
                                        const std::vector<char>& covered_ops)
    -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> candidates;
    std::vector<char> selected_ops(problem.ops.size(), false);
    std::vector<size_t> current_subgraph;
    EnumerateSubgraphCandidates(problem, producer_op, covered_ops, 0, selected_ops,
                                current_subgraph, false, candidates);
    return candidates;
}

void EnumerateSchedules(const Problem& problem, const std::vector<int>& producer_op,
                        std::vector<char>& covered_ops, size_t covered_count,
                        std::vector<std::vector<size_t>>& current_schedule,
                        std::vector<std::vector<std::vector<size_t>>>& schedules) {
    if (covered_count == problem.ops.size()) {
        schedules.push_back(current_schedule);
        return;
    }

    for (const auto& candidate :
         BuildSchedulableSubgraphCandidates(problem, producer_op, covered_ops)) {
        current_schedule.push_back(candidate);

        std::vector<size_t> newly_covered;
        for (size_t op_idx : candidate) {
            if (!covered_ops[op_idx]) {
                covered_ops[op_idx] = true;
                newly_covered.push_back(op_idx);
            }
        }

        EnumerateSchedules(problem, producer_op, covered_ops, covered_count + newly_covered.size(),
                           current_schedule, schedules);

        for (size_t op_idx : newly_covered) {
            covered_ops[op_idx] = false;
        }
        current_schedule.pop_back();
    }
}

} // namespace

auto BruteForceFuser::fuse(const Problem& problem) -> StatusOr<std::vector<Solution>> {
    std::vector<Solution> results;
    size_t num_ops = problem.ops.size();
    if (num_ops == 0) {
        return results;
    }

    std::vector<int> producer_op = BuildProducerOpIndex(problem);
    std::vector<char> covered_ops(num_ops, false);
    std::vector<std::vector<size_t>> current_schedule;
    std::vector<std::vector<std::vector<size_t>>> schedules;
    EnumerateSchedules(problem, producer_op, covered_ops, 0, current_schedule, schedules);

    for (const auto& subgraphs_ops : schedules) {

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

        // Step 3: Determine the total number of tensor retention permutations for this partition
        // structure. If subgraph 0 has 3 candidate tensors, it has 2^3 = 8 retention possibilities.
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
                for (size_t o : sol.subgraphs[dbg_idx].ops)
                    std::cout << o << " ";
                std::cout << "], retains: [";
                for (size_t t : sol.subgraphs[dbg_idx].tensors_to_retain)
                    std::cout << t << " ";
                std::cout << "]\n";
            }
#endif
        }
    }

    return results;
}

} // namespace mlsys
