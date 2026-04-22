#include "mlsys.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace absl;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

#if defined(M_Assert)
#define ASSERT_WITH_CONTEXT(condition, context_stream)                                             \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << context_stream << std::endl;                                              \
        }                                                                                          \
        M_Assert(condition);                                                                       \
    } while (0)
#else
#define ASSERT_WITH_CONTEXT(condition, context_stream)                                             \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::cerr << "Assertion failed: " #condition << "\n" << context_stream << std::endl;   \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)
#endif

namespace mlsys {

auto ReadProblem(const std::string& filename) -> StatusOr<Problem> {
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
            problem.tensors.push_back(
                Tensor{.width = widths[i].get<Width>(), .height = heights[i].get<Height>()});
        }

        // Parse Operations
        const auto& inputs = data.at("inputs");
        const auto& outputs = data.at("outputs");
        const auto& base_costs = data.at("base_costs");
        const auto& op_types = data.at("op_types");

        if (inputs.size() != outputs.size() || inputs.size() != base_costs.size() ||
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

auto ReadSolution(const std::string& filename) -> StatusOr<Solution> {
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
        for (const auto& subgraph_latencie : subgraph_latencies) {
            if (subgraph_latencie.get<SubgraphLatency>() < 0) {
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
                return absl::InvalidArgumentError("granularity must have at least 2 elements");
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

namespace {

auto SubgraphFitsFastMemoryImpl(const Problem& problem, const Solution& solution,
                                size_t subgraph_idx, const std::set<size_t>& prev_retained_tensors,
                                const std::vector<int>& producer_op, bool emit_debug) -> bool {
    if (subgraph_idx >= solution.subgraphs.size()) {
        return false;
    }

    const auto& subgraph = solution.subgraphs[subgraph_idx];
    std::set<size_t> subgraph_ops(subgraph.ops.begin(), subgraph.ops.end());
    std::set<size_t> subgraph_produced;
    std::set<size_t> subgraph_consumed;
    std::set<size_t> final_output_tensors;

    for (size_t const op_idx : subgraph.ops) {
        if (op_idx >= problem.ops.size()) {
            return false;
        }
        size_t const out = problem.ops[op_idx].outputs[0];
        subgraph_produced.insert(out);

        for (size_t const in : problem.ops[op_idx].inputs) {
            subgraph_consumed.insert(in);
        }
    }

    FastMemoryCapacity fast_memory_usage = 0;
    std::set<size_t> queued_tensors;
    std::set<size_t> full_counted_tensors;
    std::vector<size_t> partial_counted_usage(problem.tensors.size(), 0);

#ifdef DEBUG
    if (emit_debug) {
        std::cout << "\n[DEBUG] Subgraph " << subgraph_idx << " Fast Memory Capacity Check ---\n";
    }
#endif

    for (size_t const t : prev_retained_tensors) {
        if (t >= problem.tensors.size()) {
            return false;
        }
        fast_memory_usage += problem.tensors[t].width * problem.tensors[t].height;
        queued_tensors.insert(t);
        full_counted_tensors.insert(t);
#ifdef DEBUG
        if (emit_debug) {
            std::cout << "[DEBUG] Tensor " << t << " (retained) takes "
                      << problem.tensors[t].width * problem.tensors[t].height
                      << " | total_mem=" << fast_memory_usage << "\n";
        }
#endif
    }

    std::vector<std::tuple<size_t, Tensor, bool>> q;
    std::set<size_t> to_be_retained_tensors(subgraph.tensors_to_retain.begin(),
                                            subgraph.tensors_to_retain.end());

    for (size_t const t_idx : to_be_retained_tensors) {
        if (t_idx >= problem.tensors.size()) {
            return false;
        }
        if (!full_counted_tensors.contains(t_idx)) {
            size_t const required_size =
                problem.tensors[t_idx].width * problem.tensors[t_idx].height;
            fast_memory_usage += required_size;
            full_counted_tensors.insert(t_idx);
        }
        queued_tensors.insert(t_idx);
#ifdef DEBUG
        if (emit_debug) {
            std::cout << "[DEBUG] Tensor " << t_idx << " (to be retained) takes "
                      << problem.tensors[t_idx].width * problem.tensors[t_idx].height
                      << " | total_mem=" << fast_memory_usage << "\n";
        }
#endif
    }

    for (size_t const t_idx : subgraph_produced) {
        bool const is_final_output = (!subgraph_consumed.contains(t_idx));
        if (!is_final_output) {
            continue;
        }

        final_output_tensors.insert(t_idx);
        if (!queued_tensors.contains(t_idx)) {
            size_t const required_size = subgraph.granularity.width * subgraph.granularity.height;
            fast_memory_usage += required_size;
            queued_tensors.insert(t_idx);
#ifdef DEBUG
            if (emit_debug) {
                std::cout << "[DEBUG] Tensor " << t_idx << " (subgraph output) takes total space "
                          << required_size << " | total_mem=" << fast_memory_usage << "\n";
            }
#endif
        }

        Tensor tensor{
            .width = subgraph.granularity.width,
            .height = subgraph.granularity.height,
        };
        q.emplace_back(t_idx, tensor, true);
    }

    size_t head = 0;
    auto is_ignored_tensor_in_backward = [&](size_t tensor_idx) {
        return final_output_tensors.contains(tensor_idx) ||
               to_be_retained_tensors.contains(tensor_idx) ||
               prev_retained_tensors.contains(tensor_idx);
    };
    auto is_full_requirement = [&](size_t tensor_idx, const Tensor& req_tensor) {
        return req_tensor.width == problem.tensors[tensor_idx].width &&
               req_tensor.height == problem.tensors[tensor_idx].height;
    };
    auto add_requirement_usage = [&](size_t tensor_idx, const Tensor& req_tensor, bool is_ephemeral,
                                     bool is_retained) {
        if (is_ephemeral || is_retained) {
            return size_t{0};
        }

        size_t const required_size = req_tensor.width * req_tensor.height;
        bool const is_full = is_full_requirement(tensor_idx, req_tensor);
        if (full_counted_tensors.contains(tensor_idx)) {
            return size_t{0};
        }

        // Naming asymmetry:
        // - full move creates a canonical resident name that covers all later partial accesses
        // - partial moves do not supersede full coverage
        if (is_full && partial_counted_usage[tensor_idx] > 0) {
            fast_memory_usage -= partial_counted_usage[tensor_idx];
            partial_counted_usage[tensor_idx] = 0;
        }

        fast_memory_usage += required_size;
        if (is_full) {
            full_counted_tensors.insert(tensor_idx);
        } else {
            partial_counted_usage[tensor_idx] += required_size;
        }
        return required_size;
    };
    auto ignored_tensor_reason = [&](size_t tensor_idx) {
        std::string reason;
        if (final_output_tensors.contains(tensor_idx)) {
            reason += "final_output";
        }
        if (to_be_retained_tensors.contains(tensor_idx)) {
            if (!reason.empty()) {
                reason += ",";
            }
            reason += "retain_next";
        }
        if (prev_retained_tensors.contains(tensor_idx)) {
            if (!reason.empty()) {
                reason += ",";
            }
            reason += "retain_prev";
        }
        return reason;
    };
    auto emit_ignored_debug = [&](const std::string& site, size_t tensor_idx) {
#ifdef DEBUG
        if (emit_debug) {
            bool const is_ephemeral_candidate = subgraph_produced.contains(tensor_idx);
            std::cout
                << "[DEBUG] " << site << " Tensor " << tensor_idx << " skipped by ignore-set ("
                << ignored_tensor_reason(tensor_idx) << ")"
                << (is_ephemeral_candidate
                        ? "; ephemeral candidate, but memory path was pre-accounted by ignore-set"
                        : "; not produced in this subgraph")
                << " | total_mem=" << fast_memory_usage << "\n";
        }
#endif
    };

    while (head < q.size()) {
        auto [curr_tensor_idx, curr_tensor_dim, is_final] = q[head++];

        if (curr_tensor_idx >= producer_op.size()) {
            return false;
        }
        int const op_idx = producer_op[curr_tensor_idx];

        if (op_idx == -1 || !subgraph_ops.contains(static_cast<size_t>(op_idx))) {
#ifdef DEBUG
            if (emit_debug) {
                std::cout << "[DEBUG] Visit Tensor " << curr_tensor_idx
                          << " req_w=" << curr_tensor_dim.width
                          << " req_h=" << curr_tensor_dim.height << " is_final=" << is_final
                          << (op_idx == -1 ? " from OP -1 (Global Input)\n"
                                           : " reached subgraph boundary\n");
            }
#endif
            continue;
        }
        if (op_idx < 0 || static_cast<size_t>(op_idx) >= problem.ops.size()) {
            return false;
        }

        const Op& op = problem.ops[static_cast<size_t>(op_idx)];

#ifdef DEBUG
        if (emit_debug) {
            std::cout << "[DEBUG] Visit Tensor " << curr_tensor_idx
                      << " req_w=" << curr_tensor_dim.width << " req_h=" << curr_tensor_dim.height
                      << " is_final=" << is_final << " from OP " << op_idx << "\n";
        }
#endif

        if (op.op_type == "MatMul") {
            size_t const lhs_tensor_idx = op.inputs[0];
            size_t const rhs_tensor_idx = op.inputs[1];

            int64_t const inner_k =
                is_final ? subgraph.granularity.depth : problem.tensors[lhs_tensor_idx].width;

#ifdef DEBUG
            if (emit_debug) {
                std::cout << "[DEBUG] MatMul OP " << op_idx << " uses inner_k=" << inner_k << "\n";
            }
#endif

            Tensor lhs_tensor{
                .width = inner_k,
                .height = curr_tensor_dim.height,
            };
            bool const lhs_is_ignored = is_ignored_tensor_in_backward(lhs_tensor_idx);
            if (!lhs_is_ignored) {
                bool const lhs_is_ephemeral = (subgraph_produced.contains(lhs_tensor_idx));
                bool const lhs_is_retained = (prev_retained_tensors.contains(lhs_tensor_idx));
                size_t const lhs_added_size = add_requirement_usage(
                    lhs_tensor_idx, lhs_tensor, lhs_is_ephemeral, lhs_is_retained);
#ifdef DEBUG
                if (emit_debug) {
                    std::cout << "[DEBUG] MatMul LHS Tensor " << lhs_tensor_idx
                              << (lhs_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes"
                                  : lhs_is_retained
                                      ? " is RETAINED! Takes 0 bytes"
                                      : " takes " + std::to_string(lhs_added_size) +
                                            " (req_w=" + std::to_string(lhs_tensor.width) +
                                            " req_h=" + std::to_string(lhs_tensor.height) + ")")
                              << " | total_mem=" << fast_memory_usage << "\n";
                }
#endif
                if (!queued_tensors.contains(lhs_tensor_idx)) {
                    queued_tensors.insert(lhs_tensor_idx);
                    q.emplace_back(lhs_tensor_idx, lhs_tensor, false);
                }
            } else {
                emit_ignored_debug("MatMul LHS", lhs_tensor_idx);
            }

            Tensor rhs_tensor{
                .width = curr_tensor_dim.width,
                .height = inner_k,
            };
            bool const rhs_is_ignored = is_ignored_tensor_in_backward(rhs_tensor_idx);
            if (!rhs_is_ignored) {
                bool const rhs_is_ephemeral = (subgraph_produced.contains(rhs_tensor_idx));
                bool const rhs_is_retained = (prev_retained_tensors.contains(rhs_tensor_idx));
                size_t const rhs_added_size = add_requirement_usage(
                    rhs_tensor_idx, rhs_tensor, rhs_is_ephemeral, rhs_is_retained);
#ifdef DEBUG
                if (emit_debug) {
                    std::cout << "[DEBUG] MatMul RHS Tensor " << rhs_tensor_idx
                              << (rhs_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes"
                                  : rhs_is_retained
                                      ? " is RETAINED! Takes 0 bytes"
                                      : " takes " + std::to_string(rhs_added_size) +
                                            " (req_w=" + std::to_string(rhs_tensor.width) +
                                            " req_h=" + std::to_string(rhs_tensor.height) + ")")
                              << " | total_mem=" << fast_memory_usage << "\n";
                }
#endif
                if (!queued_tensors.contains(rhs_tensor_idx)) {
                    queued_tensors.insert(rhs_tensor_idx);
                    q.emplace_back(rhs_tensor_idx, rhs_tensor, false);
                }
            } else {
                emit_ignored_debug("MatMul RHS", rhs_tensor_idx);
            }
        } else if (op.op_type == "Pointwise") {
            if (op.inputs.size() == 1) {
                size_t const in_tensor_idx = op.inputs[0];
                bool const in_is_ignored = is_ignored_tensor_in_backward(in_tensor_idx);
                if (!in_is_ignored) {
                    bool const in_is_ephemeral = (subgraph_produced.contains(in_tensor_idx));
                    bool const in_is_retained = (prev_retained_tensors.contains(in_tensor_idx));
                    size_t const in_added_size = add_requirement_usage(
                        in_tensor_idx, curr_tensor_dim, in_is_ephemeral, in_is_retained);
#ifdef DEBUG
                    if (emit_debug) {
                        std::cout << "[DEBUG] Pointwise Unary Input Tensor " << in_tensor_idx
                                  << (in_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes"
                                      : in_is_retained
                                          ? " is RETAINED! Takes 0 bytes"
                                          : " takes " + std::to_string(in_added_size) +
                                                " (req_w=" + std::to_string(curr_tensor_dim.width) +
                                                " req_h=" + std::to_string(curr_tensor_dim.height) +
                                                ")")
                                  << " | total_mem=" << fast_memory_usage << "\n";
                    }
#endif
                    if (!queued_tensors.contains(in_tensor_idx)) {
                        queued_tensors.insert(in_tensor_idx);
                        q.emplace_back(in_tensor_idx, curr_tensor_dim, false);
                    }
                } else {
                    emit_ignored_debug("Pointwise Unary Input", in_tensor_idx);
                }
            } else {
                for (size_t const in_tensor_idx : op.inputs) {
                    bool const in_is_ignored = is_ignored_tensor_in_backward(in_tensor_idx);
                    if (in_is_ignored) {
                        emit_ignored_debug("Pointwise Input", in_tensor_idx);
                        continue;
                    }

                    bool const in_is_ephemeral = (subgraph_produced.contains(in_tensor_idx));
                    bool const in_is_retained = (prev_retained_tensors.contains(in_tensor_idx));
                    size_t const in_added_size = add_requirement_usage(
                        in_tensor_idx, curr_tensor_dim, in_is_ephemeral, in_is_retained);
#ifdef DEBUG
                    if (emit_debug) {
                        std::cout << "[DEBUG] Pointwise Input Tensor " << in_tensor_idx
                                  << (in_is_ephemeral ? " is EPHEMERAL! Takes 0 bytes"
                                      : in_is_retained
                                          ? " is RETAINED! Takes 0 bytes"
                                          : " takes " + std::to_string(in_added_size) +
                                                " (req_w=" + std::to_string(curr_tensor_dim.width) +
                                                " req_h=" + std::to_string(curr_tensor_dim.height) +
                                                ")")
                                  << " | total_mem=" << fast_memory_usage << "\n";
                    }
#endif
                    if (!queued_tensors.contains(in_tensor_idx)) {
                        queued_tensors.insert(in_tensor_idx);
                        q.emplace_back(in_tensor_idx, curr_tensor_dim, false);
                    }
                }
            }
        }
    }

#ifdef DEBUG
    if (emit_debug) {
        std::cout << "[DEBUG] Subgraph " << subgraph_idx
                  << " Fast Memory Total: " << fast_memory_usage << " / "
                  << problem.fast_memory_capacity << "\n";
    }
#endif

    return fast_memory_usage <= problem.fast_memory_capacity;
}

} // namespace

auto SubgraphFitsFastMemory(const Problem& problem, const Solution& solution, size_t subgraph_idx,
                            const std::set<size_t>& prev_retained_tensors,
                            const std::vector<int>& producer_op) -> bool {
    return SubgraphFitsFastMemoryImpl(problem, solution, subgraph_idx, prev_retained_tensors,
                                      producer_op, false);
}

#ifdef DEBUG
auto DebugSubgraphFitsFastMemory(const Problem& problem, const Solution& solution,
                                 size_t subgraph_idx, const std::set<size_t>& prev_retained_tensors,
                                 const std::vector<int>& producer_op) -> bool {
    return SubgraphFitsFastMemoryImpl(problem, solution, subgraph_idx, prev_retained_tensors,
                                      producer_op, true);
}
#endif

StatusOr<TotalLatency> Evaluate(const Problem& problem, const Solution& solution) {
#ifdef DEBUG
    std::cout << "\n[DEBUG] Starting Evaluate Function\n";
#endif
    std::vector<bool> inputs_satisfied_global(problem.tensors.size(), true);
    std::vector<bool> inputs_satisfied_retained(problem.tensors.size(), false);

    // Use assertions to check the problem is valid
    for (size_t i = 0; i < problem.ops.size(); i++) {
        Op op = problem.ops[i];
        ASSERT_WITH_CONTEXT(op.inputs.size() > 0,
                            "op_index=" << i << ", op_type=" << op.op_type
                                        << ", inputs.size()=" << op.inputs.size());
        ASSERT_WITH_CONTEXT(op.outputs.size() == 1,
                            "op_index=" << i << ", op_type=" << op.op_type
                                        << ", outputs.size()=" << op.outputs.size());
        ASSERT_WITH_CONTEXT(op.op_type == "MatMul" || op.op_type == "Pointwise",
                            "op_index=" << i << ", op_type=" << op.op_type);

        // Check op tensors exist
        for (size_t in_tensor_idx : op.inputs) {
            ASSERT_WITH_CONTEXT(in_tensor_idx < problem.tensors.size(),
                                "op_index=" << i << ", op_type=" << op.op_type
                                            << ", input_tensor_idx=" << in_tensor_idx
                                            << ", num_tensors=" << problem.tensors.size());
        }
        ASSERT_WITH_CONTEXT(op.outputs[0] < problem.tensors.size(),
                            "op_index=" << i << ", op_type=" << op.op_type
                                        << ", output_tensor_idx=" << op.outputs[0]
                                        << ", num_tensors=" << problem.tensors.size());

        // Check operation tensor size matching
        if (op.op_type == "MatMul") {
            ASSERT_WITH_CONTEXT(op.inputs.size() == 2,
                                "op_index=" << i << ", op_type=" << op.op_type
                                            << ", inputs.size()=" << op.inputs.size()
                                            << ", outputs.size()=" << op.outputs.size());

            Tensor lhs_tensor = problem.tensors[op.inputs[0]];
            Tensor rhs_tensor = problem.tensors[op.inputs[1]];
            Tensor out_tensor = problem.tensors[op.outputs[0]];
            // width corresponds to columns and height corresponds to rows.
            // MatMul shape rules:
            // lhs: [out_h x k], rhs: [k x out_w], out: [out_h x out_w].
            ASSERT_WITH_CONTEXT(lhs_tensor.width == rhs_tensor.height,
                                "op_index=" << i << ", lhs_width=" << lhs_tensor.width
                                            << ", rhs_height=" << rhs_tensor.height);
            ASSERT_WITH_CONTEXT(out_tensor.height == lhs_tensor.height,
                                "op_index=" << i << ", out_height=" << out_tensor.height
                                            << ", lhs_height=" << lhs_tensor.height);
            ASSERT_WITH_CONTEXT(out_tensor.width == rhs_tensor.width,
                                "op_index=" << i << ", out_width=" << out_tensor.width
                                            << ", rhs_width=" << rhs_tensor.width);
        } else if (op.op_type == "Pointwise") {
            // check all the input and output have same shape
            for (size_t in : op.inputs) {
                ASSERT_WITH_CONTEXT(
                    problem.tensors[in].width == problem.tensors[op.outputs[0]].width,
                    "op_index=" << i << ", input_tensor_idx=" << in << ", output_tensor_idx="
                                << op.outputs[0] << ", input_width=" << problem.tensors[in].width
                                << ", output_width=" << problem.tensors[op.outputs[0]].width);
                ASSERT_WITH_CONTEXT(
                    problem.tensors[in].height == problem.tensors[op.outputs[0]].height,
                    "op_index=" << i << ", input_tensor_idx=" << in << ", output_tensor_idx="
                                << op.outputs[0] << ", input_height=" << problem.tensors[in].height
                                << ", output_height=" << problem.tensors[op.outputs[0]].height);
            }
        }
    }
#ifdef DEBUG
    std::cout << "[DEBUG] All assertions passed\n";
#endif

    // Validate solution-level fields up-front. Problem-level checks should use assertions;
    // malformed solutions should return errors.
    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        const auto& subgraph = solution.subgraphs[sg_idx];

        if (subgraph.granularity.width <= 0 || subgraph.granularity.height <= 0 ||
            subgraph.granularity.depth <= 0) {
            return absl::InvalidArgumentError(
                "[Invalid Granularity] Subgraph granularity dimensions must be positive");
        }

        std::set<size_t> subgraph_produced;
        std::set<size_t> subgraph_consumed;
        for (size_t const op_idx : subgraph.ops) {
            if (op_idx >= problem.ops.size()) {
                return absl::InvalidArgumentError(
                    "[Invalid Op Index] Invalid op index in subgraph");
            }
            size_t const out = problem.ops[op_idx].outputs[0];
            subgraph_produced.insert(out);
            for (size_t const in : problem.ops[op_idx].inputs) {
                subgraph_consumed.insert(in);
            }
        }

        for (size_t const t_idx : subgraph.tensors_to_retain) {
            if (t_idx >= problem.tensors.size()) {
                return absl::InvalidArgumentError(
                    "[Invalid Retained Tensor] Invalid tensor index in tensors_to_retain");
            }
        }

        if (subgraph.traversal_order.has_value()) {
            std::vector<size_t> final_outputs;
            final_outputs.reserve(
                subgraph_produced.size()); // Purely optimization no functional change
            for (size_t const t_idx : subgraph_produced) {
                if (!subgraph_consumed.contains(t_idx)) {
                    final_outputs.push_back(t_idx);
                }
            }

            if (final_outputs.empty()) {
                return absl::InvalidArgumentError(
                    "[Invalid Traversal Order] Subgraph has no final output for traversal");
            }

            auto tile_count_for_output = [&](size_t tensor_idx) -> int64_t {
                int64_t const out_w = problem.tensors[tensor_idx].width;
                int64_t const out_h = problem.tensors[tensor_idx].height;
                int64_t const gran_w = subgraph.granularity.width;
                int64_t const gran_h = subgraph.granularity.height;
                int64_t const tiles_w = (out_w + gran_w - 1) / gran_w;
                int64_t const tiles_h = (out_h + gran_h - 1) / gran_h;
                return tiles_w * tiles_h;
            };

            int64_t const expected_tiles = tile_count_for_output(final_outputs[0]);
            for (size_t const out_idx : final_outputs) {
                if (tile_count_for_output(out_idx) != expected_tiles) {
                    return absl::InvalidArgumentError(
                        "[Invalid Traversal Order] Inconsistent final output tile counts");
                }
            }

            const auto& traversal_order = subgraph.traversal_order.value();
            if (traversal_order.size() != static_cast<size_t>(expected_tiles)) {
                return absl::InvalidArgumentError(
                    "[Invalid Traversal Order] traversal_order length does not match tile count");
            }

            std::vector<bool> seen(static_cast<size_t>(expected_tiles), false);
            for (int64_t const tile_idx : traversal_order) {
                if (tile_idx < 0 || tile_idx >= expected_tiles) {
                    return absl::InvalidArgumentError(
                        "[Invalid Traversal Order] traversal_order index out of range");
                }
                if (seen[static_cast<size_t>(tile_idx)]) {
                    return absl::InvalidArgumentError(
                        "[Invalid Traversal Order] traversal_order contains duplicates");
                }
                seen[static_cast<size_t>(tile_idx)] = true;
            }
        }
    }
#ifdef DEBUG
    std::cout << "[DEBUG] All assertions passed\n";
#endif

    // Identify tensors that are not produced by any op
    std::vector<int> producer_op(
        problem.tensors.size(), -1); // producer_op[i] is the index of the op that produces tensor i
    std::vector<bool> produced_outputs(problem.tensors.size(), false);
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        size_t const out = problem.ops[i].outputs[0];
        inputs_satisfied_global[out] =
            false; // produced tensors are unavailable until a valid subgraph completes
        producer_op[out] = i;
    }

    TotalLatency total_latency = 0.0;
    std::set<size_t> prev_retained_tensors;

    for (size_t i = 0; i < solution.subgraphs.size(); ++i) {
        const auto& subgraph = solution.subgraphs[i];
        std::set<size_t> subgraph_produced;
        std::set<size_t> subgraph_consumed;
        std::set<size_t> final_output_tensors;

        for (size_t const op_idx : subgraph.ops) {
            if (op_idx >= problem.ops.size()) {
                return absl::InvalidArgumentError(
                    "[Invalid Op Index] Invalid op index in subgraph");
            }

            size_t const out = problem.ops[op_idx].outputs[0];
            subgraph_produced.insert(out);

            for (size_t const in : problem.ops[op_idx].inputs) {
                subgraph_consumed.insert(in);
            }
        }

        for (size_t const t_idx : subgraph_produced) {
            if (!subgraph_consumed.contains(t_idx)) {
                final_output_tensors.insert(t_idx);
            }
        }

        // --- Dependency Check ---
        std::vector<bool> inputs_satisfied_local(problem.tensors.size(), false);
        for (size_t const op_idx : subgraph.ops) {
            for (size_t const in : problem.ops[op_idx].inputs) {
                bool const input_available = inputs_satisfied_global[in] ||
                                             inputs_satisfied_retained[in] ||
                                             inputs_satisfied_local[in];
                if (!input_available) {
                    return absl::FailedPreconditionError(
                        "[Unmet Dependency] Dependency not met for tensor " + std::to_string(in));
                }
            }

            size_t const out = problem.ops[op_idx].outputs[0];
            inputs_satisfied_local[out] = true;
            produced_outputs[out] = true;
        }

        for (size_t const t_idx : final_output_tensors) {
            inputs_satisfied_global[t_idx] = true;
        }

        if (!SubgraphFitsFastMemoryImpl(problem, solution, i, prev_retained_tensors, producer_op,
                                        true)) {
            return absl::ResourceExhaustedError(
                "[Fast Memory Capacity Exceeded] Fast memory capacity exceeded in subgraph " +
                std::to_string(i));
        }

        // --- Total Latency ---
        total_latency += subgraph.subgraph_latency;

        // Update retained tensors for next subgraph
        prev_retained_tensors.clear();
        std::fill(inputs_satisfied_retained.begin(), inputs_satisfied_retained.end(), false);
        for (size_t const t : subgraph.tensors_to_retain) {
            prev_retained_tensors.insert(t);
            inputs_satisfied_retained[t] = true;
        }
    }

    // --- All Operations Done Check ---
    for (size_t i = 0; i < problem.tensors.size(); ++i) {
        if (producer_op[i] != -1 && !produced_outputs[i]) {
            return absl::FailedPreconditionError("[Missed Output] Output " + std::to_string(i) +
                                                 " was not produced");
        }
    }

    return total_latency;
}

auto WriteSolution(const Solution& solution, const std::string& filename) -> Status {
    ordered_json j;

    j["subgraphs"] = ordered_json::array();
    j["granularities"] = ordered_json::array();
    j["tensors_to_retain"] = ordered_json::array();
    j["traversal_orders"] = ordered_json::array();
    j["subgraph_latencies"] = ordered_json::array();

    for (const auto& sg : solution.subgraphs) {
        j["subgraphs"].push_back(sg.ops);
        j["granularities"].push_back(
            {sg.granularity.width, sg.granularity.height, sg.granularity.depth});
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

    out_file << j.dump(2) << '\n';
    out_file.close();

    return absl::OkStatus();
}

} // namespace mlsys
