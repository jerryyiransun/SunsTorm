#include "cost_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace mlsys {

namespace {

using TilesByTensor = std::unordered_map<size_t, std::vector<Tile>>;

struct SubgraphMeta {
    std::set<size_t> op_set;
    std::set<size_t> produced;
    std::set<size_t> consumed;
    std::set<size_t> final_outputs;
    std::set<size_t> boundary_inputs;
    std::set<size_t> boundary_outputs;
    std::set<size_t> ephemeral_tensors;
};

struct OpStepRequirement {
    std::vector<Tile> output_tiles;
    int64_t req_k = 1;
};

struct StepRequirements {
    TilesByTensor required_boundary_inputs;
    TilesByTensor required_boundary_outputs;
    std::unordered_map<size_t, OpStepRequirement> op_requirements;
};

auto CeilDiv(int64_t a, int64_t b) -> int64_t {
    return (a + b - 1) / b;
}

void AddTile(TilesByTensor& map, const Tile& tile) {
    if (tile.area() <= 0) {
        return;
    }
    map[tile.tensor_idx].push_back(tile);
}

void AddTiles(TilesByTensor& dst, const TilesByTensor& src) {
    for (const auto& [tensor_idx, tiles] : src) {
        auto& out = dst[tensor_idx];
        out.insert(out.end(), tiles.begin(), tiles.end());
    }
}

auto ClipTileToTensor(const Problem& problem, Tile tile) -> Tile {
    const Tensor& t = problem.tensors[tile.tensor_idx];

    tile.x0 = std::clamp<int64_t>(tile.x0, 0, t.width);
    tile.x1 = std::clamp<int64_t>(tile.x1, 0, t.width);
    tile.y0 = std::clamp<int64_t>(tile.y0, 0, t.height);
    tile.y1 = std::clamp<int64_t>(tile.y1, 0, t.height);

    if (tile.x1 < tile.x0) {
        tile.x1 = tile.x0;
    }
    if (tile.y1 < tile.y0) {
        tile.y1 = tile.y0;
    }

    return tile;
}

auto FlattenTiles(const TilesByTensor& map) -> std::vector<Tile> {
    std::vector<Tile> out;
    for (const auto& [_, tiles] : map) {
        out.insert(out.end(), tiles.begin(), tiles.end());
    }
    return out;
}

auto ComputeMapArea(const TilesByTensor& map) -> int64_t {
    return Tile::ComputeNonOverlappingArea(FlattenTiles(map));
}

auto ComputeMissingArea(const TilesByTensor& required, const TilesByTensor& resident) -> int64_t {
    int64_t missing = 0;

    for (const auto& [tensor_idx, req_tiles] : required) {
        auto it = resident.find(tensor_idx);
        std::vector<Tile> resident_tiles;
        if (it != resident.end()) {
            resident_tiles = it->second;
        }

        int64_t resident_area = Tile::ComputeNonOverlappingArea(resident_tiles);

        std::vector<Tile> combined = req_tiles;
        combined.insert(combined.end(), resident_tiles.begin(), resident_tiles.end());

        int64_t combined_area = Tile::ComputeNonOverlappingArea(combined);
        missing += std::max<int64_t>(0, combined_area - resident_area);
    }

    return missing;
}

auto BuildProducerMap(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        producer[problem.ops[op_idx].outputs[0]] = static_cast<int>(op_idx);
    }
    return producer;
}

auto BuildConsumersMap(const Problem& problem) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> consumers(problem.tensors.size());
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        for (size_t t_idx : problem.ops[op_idx].inputs) {
            consumers[t_idx].push_back(op_idx);
        }
    }
    return consumers;
}

auto BuildSubgraphMeta(const Problem& problem, const Subgraph& subgraph,
                       const std::vector<std::vector<size_t>>& consumers_by_tensor)
    -> SubgraphMeta {
    SubgraphMeta meta;

    for (size_t op_idx : subgraph.ops) {
        meta.op_set.insert(op_idx);

        size_t out = problem.ops[op_idx].outputs[0];
        meta.produced.insert(out);

        for (size_t in : problem.ops[op_idx].inputs) {
            meta.consumed.insert(in);
        }
    }

    for (size_t t_idx : meta.produced) {
        if (!meta.consumed.contains(t_idx)) {
            meta.final_outputs.insert(t_idx);
        }
    }

    for (size_t t_idx : meta.consumed) {
        if (!meta.produced.contains(t_idx)) {
            meta.boundary_inputs.insert(t_idx);
        }
    }

    for (size_t t_idx : meta.produced) {
        bool const graph_output = consumers_by_tensor[t_idx].empty();
        bool external_consumer = false;
        for (size_t consumer_op_idx : consumers_by_tensor[t_idx]) {
            if (!meta.op_set.contains(consumer_op_idx)) {
                external_consumer = true;
                break;
            }
        }

        if (graph_output || external_consumer) {
            meta.boundary_outputs.insert(t_idx);
        }
    }

    for (size_t t_idx : meta.produced) {
        bool const consumed_inside = meta.consumed.contains(t_idx);
        if (consumed_inside && !meta.boundary_outputs.contains(t_idx)) {
            meta.ephemeral_tensors.insert(t_idx);
        }
    }

    return meta;
}

auto BuildTraversalOrder(const Problem& problem, const Subgraph& subgraph,
                         size_t reference_output_tensor) -> absl::StatusOr<std::vector<int64_t>> {
    int64_t const out_w = problem.tensors[reference_output_tensor].width;
    int64_t const out_h = problem.tensors[reference_output_tensor].height;
    int64_t const gran_w = subgraph.granularity.width;
    int64_t const gran_h = subgraph.granularity.height;

    int64_t const tiles_w = CeilDiv(out_w, gran_w);
    int64_t const tiles_h = CeilDiv(out_h, gran_h);
    int64_t const total_tiles = tiles_w * tiles_h;

    if (!subgraph.traversal_order.has_value()) {
        std::vector<int64_t> order(static_cast<size_t>(total_tiles));
        for (int64_t i = 0; i < total_tiles; ++i) {
            order[static_cast<size_t>(i)] = i;
        }
        return order;
    }

    const auto& order = subgraph.traversal_order.value();
    if (order.size() != static_cast<size_t>(total_tiles)) {
        return absl::InvalidArgumentError(
            "CostModel: traversal_order length does not match tile count");
    }

    return order;
}

auto BuildTileForLinearIndex(const Problem& problem, const Subgraph& subgraph, size_t tensor_idx,
                             int64_t linear_idx, int64_t tiles_w) -> Tile {
    int64_t const tx = linear_idx % tiles_w;
    int64_t const ty = linear_idx / tiles_w;

    int64_t const gran_w = subgraph.granularity.width;
    int64_t const gran_h = subgraph.granularity.height;

    Tile tile{.tensor_idx = tensor_idx,
              .x0 = tx * gran_w,
              .x1 = (tx + 1) * gran_w,
              .y0 = ty * gran_h,
              .y1 = (ty + 1) * gran_h};

    return ClipTileToTensor(problem, tile);
}

auto BuildStickyResidentFromRetained(const Problem& problem, const std::set<size_t>& retained)
    -> TilesByTensor {
    TilesByTensor sticky;
    for (size_t t_idx : retained) {
        if (t_idx >= problem.tensors.size()) {
            continue;
        }
        const Tensor& t = problem.tensors[t_idx];
        AddTile(sticky, Tile{.tensor_idx = t_idx, .x0 = 0, .x1 = t.width, .y0 = 0, .y1 = t.height});
    }
    return sticky;
}

void AddLockedSplitKOutputs(const TilesByTensor& required_outputs,
                            const std::set<size_t>& split_k_output_tensors,
                            TilesByTensor& transient_next) {
    for (const auto& [tensor_idx, tiles] : required_outputs) {
        if (!split_k_output_tensors.contains(tensor_idx)) {
            continue;
        }
        auto& out = transient_next[tensor_idx];
        out.insert(out.end(), tiles.begin(), tiles.end());
    }
}

auto BuildNextRetainedSet(const Problem& problem, const Subgraph& subgraph,
                          const SubgraphMeta& meta, const std::set<size_t>& prev_retained)
    -> std::set<size_t> {
    std::set<size_t> next;

    for (size_t t_idx : subgraph.tensors_to_retain) {
        if (t_idx >= problem.tensors.size()) {
            continue;
        }
        if (meta.ephemeral_tensors.contains(t_idx)) {
            continue;
        }

        bool const touched = meta.consumed.contains(t_idx) || meta.produced.contains(t_idx) ||
                             prev_retained.contains(t_idx);
        if (touched) {
            next.insert(t_idx);
        }
    }

    return next;
}

auto CollectStepRequirements(const Problem& problem, const Subgraph& subgraph,
                             const SubgraphMeta& meta, const std::vector<int>& producer_op,
                             int64_t spatial_tile_linear_idx, int64_t tiles_w, int64_t k_start,
                             int64_t k_size) -> StepRequirements {
    StepRequirements reqs;

    std::vector<Tile> queue;

    for (size_t out_tensor_idx : meta.final_outputs) {
        Tile out_tile = BuildTileForLinearIndex(problem, subgraph, out_tensor_idx,
                                                spatial_tile_linear_idx, tiles_w);
        if (out_tile.area() <= 0) {
            continue;
        }

        queue.push_back(out_tile);

        if (meta.boundary_outputs.contains(out_tensor_idx) &&
            !meta.ephemeral_tensors.contains(out_tensor_idx)) {
            AddTile(reqs.required_boundary_outputs, out_tile);
        }
    }

    size_t head = 0;
    while (head < queue.size()) {
        Tile curr_tile = queue[head++];

        int const p_op_idx = producer_op[curr_tile.tensor_idx];
        if (p_op_idx < 0) {
            continue;
        }

        size_t const op_idx = static_cast<size_t>(p_op_idx);
        if (!meta.op_set.contains(op_idx)) {
            continue;
        }

        const Op& op = problem.ops[op_idx];

        OpStepRequirement& op_req = reqs.op_requirements[op_idx];
        op_req.output_tiles.push_back(curr_tile);

        int64_t req_k = 1;
        if (op.op_type == "MatMul") {
            size_t const lhs_idx = op.inputs[0];
            int64_t const full_k = problem.tensors[lhs_idx].width;

            bool const is_final_output_tensor = meta.final_outputs.contains(curr_tile.tensor_idx);
            req_k = is_final_output_tensor ? k_size : full_k;
        }
        op_req.req_k = std::max(op_req.req_k, req_k);

        if (op.op_type == "Pointwise") {
            for (size_t in_tensor_idx : op.inputs) {
                Tile in_tile{.tensor_idx = in_tensor_idx,
                             .x0 = curr_tile.x0,
                             .x1 = curr_tile.x1,
                             .y0 = curr_tile.y0,
                             .y1 = curr_tile.y1};
                in_tile = ClipTileToTensor(problem, in_tile);
                if (in_tile.area() <= 0) {
                    continue;
                }

                if (meta.produced.contains(in_tensor_idx)) {
                    queue.push_back(in_tile);
                } else {
                    AddTile(reqs.required_boundary_inputs, in_tile);
                }
            }
            continue;
        }

        if (op.op_type == "MatMul") {
            size_t const lhs_idx = op.inputs[0];
            size_t const rhs_idx = op.inputs[1];

            int64_t const full_inner_k = problem.tensors[lhs_idx].width;
            bool const is_final_output_tensor = meta.final_outputs.contains(curr_tile.tensor_idx);
            int64_t const k0 = is_final_output_tensor ? k_start : 0;
            int64_t const k1 = is_final_output_tensor ? (k_start + k_size) : full_inner_k;

            Tile lhs_tile{
                .tensor_idx = lhs_idx, .x0 = k0, .x1 = k1, .y0 = curr_tile.y0, .y1 = curr_tile.y1};
            Tile rhs_tile{
                .tensor_idx = rhs_idx, .x0 = curr_tile.x0, .x1 = curr_tile.x1, .y0 = k0, .y1 = k1};

            lhs_tile = ClipTileToTensor(problem, lhs_tile);
            rhs_tile = ClipTileToTensor(problem, rhs_tile);

            if (lhs_tile.area() > 0) {
                if (meta.produced.contains(lhs_idx)) {
                    queue.push_back(lhs_tile);
                } else {
                    AddTile(reqs.required_boundary_inputs, lhs_tile);
                }
            }

            if (rhs_tile.area() > 0) {
                if (meta.produced.contains(rhs_idx)) {
                    queue.push_back(rhs_tile);
                } else {
                    AddTile(reqs.required_boundary_inputs, rhs_tile);
                }
            }
        }
    }

    return reqs;
}

auto ComputeStepComputeTime(const Problem& problem, const Subgraph& subgraph,
                            const std::unordered_map<size_t, OpStepRequirement>& op_requirements)
    -> double {
    double const native_w = static_cast<double>(problem.native_granularity.width);
    double const native_h = static_cast<double>(problem.native_granularity.height);

    double const gran_w = static_cast<double>(subgraph.granularity.width);
    double const gran_h = static_cast<double>(subgraph.granularity.height);

    double const padded_w =
        static_cast<double>(std::max(problem.native_granularity.width, subgraph.granularity.width));
    double const padded_h = static_cast<double>(
        std::max(problem.native_granularity.height, subgraph.granularity.height));

    double const padding_ratio = (padded_w * padded_h) / (gran_w * gran_h);

    double compute_time = 0.0;

    for (const auto& [op_idx, req] : op_requirements) {
        const Op& op = problem.ops[op_idx];

        int64_t const required_output_area = Tile::ComputeNonOverlappingArea(req.output_tiles);
        if (required_output_area <= 0) {
            continue;
        }

        int64_t full_k = 1;
        int64_t req_k = 1;

        if (op.op_type == "MatMul") {
            full_k = problem.tensors[op.inputs[0]].width;
            req_k = req.req_k;
        }

        double const numerator =
            static_cast<double>(required_output_area) * static_cast<double>(req_k);
        double const denominator = native_w * native_h * static_cast<double>(full_k);

        compute_time +=
            static_cast<double>(op.base_cost) * (numerator / denominator) * padding_ratio;
    }

    return compute_time;
}

} // namespace

auto Tile::area() const -> int64_t {
    int64_t const w = std::max<int64_t>(0, x1 - x0);
    int64_t const h = std::max<int64_t>(0, y1 - y0);
    return w * h;
}

auto Tile::ComputeNonOverlappingArea(const std::vector<Tile>& tiles) -> int64_t {
    std::unordered_map<size_t, std::vector<Tile>> by_tensor;
    for (const Tile& t : tiles) {
        if (t.area() <= 0) {
            continue;
        }
        by_tensor[t.tensor_idx].push_back(t);
    }

    int64_t total_area = 0;

    for (const auto& [_, tensor_tiles] : by_tensor) {
        struct Event {
            int64_t x;
            int64_t y0;
            int64_t y1;
            int type; // +1 add, -1 remove
        };

        std::vector<Event> events;
        events.reserve(tensor_tiles.size() * 2);

        for (const Tile& t : tensor_tiles) {
            events.push_back(Event{.x = t.x0, .y0 = t.y0, .y1 = t.y1, .type = +1});
            events.push_back(Event{.x = t.x1, .y0 = t.y0, .y1 = t.y1, .type = -1});
        }

        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
            if (a.x != b.x) {
                return a.x < b.x;
            }
            return a.type > b.type;
        });

        std::map<std::pair<int64_t, int64_t>, int> active;

        auto covered_y = [&]() -> int64_t {
            if (active.empty()) {
                return 0;
            }

            std::vector<std::pair<int64_t, int64_t>> intervals;
            intervals.reserve(active.size());
            for (const auto& [interval, count] : active) {
                if (count > 0) {
                    intervals.push_back(interval);
                }
            }

            if (intervals.empty()) {
                return 0;
            }

            std::sort(intervals.begin(), intervals.end());

            int64_t total = 0;
            int64_t cur_y0 = intervals[0].first;
            int64_t cur_y1 = intervals[0].second;

            for (size_t i = 1; i < intervals.size(); ++i) {
                const auto& [y0, y1] = intervals[i];
                if (y0 > cur_y1) {
                    total += std::max<int64_t>(0, cur_y1 - cur_y0);
                    cur_y0 = y0;
                    cur_y1 = y1;
                } else {
                    cur_y1 = std::max(cur_y1, y1);
                }
            }

            total += std::max<int64_t>(0, cur_y1 - cur_y0);
            return total;
        };

        size_t i = 0;
        int64_t prev_x = events[0].x;

        while (i < events.size()) {
            int64_t const x = events[i].x;
            int64_t const dx = x - prev_x;
            if (dx > 0) {
                total_area += dx * covered_y();
            }

            while (i < events.size() && events[i].x == x) {
                const Event& e = events[i];
                auto key = std::make_pair(e.y0, e.y1);
                if (e.type == +1) {
                    active[key] += 1;
                } else {
                    auto it = active.find(key);
                    if (it != active.end()) {
                        it->second -= 1;
                        if (it->second <= 0) {
                            active.erase(it);
                        }
                    }
                }
                ++i;
            }

            prev_x = x;
        }
    }

    return total_area;
}

// Solution is checked to not overflow fast memory no need to check for that in the cost model.
// returns the total latency. Return a tuple of the updated solution with subgraph latencies filled
// in, and the total latency of the solution.
auto CostModel::estimate(const Problem& problem, const Solution& solution)
    -> StatusOr<std::tuple<Solution, SubgraphLatency>> {
    if (problem.slow_memory_bandwidth <= 0) {
        return absl::InvalidArgumentError("CostModel: slow_memory_bandwidth must be positive");
    }

    std::vector<int> const producer_op = BuildProducerMap(problem);
    std::vector<std::vector<size_t>> const consumers_by_tensor = BuildConsumersMap(problem);

    Solution estimated_solution = solution;
    SubgraphLatency total_latency = 0.0;

    std::set<size_t> prev_retained_tensors;

    for (size_t sg_idx = 0; sg_idx < estimated_solution.subgraphs.size(); ++sg_idx) {
        Subgraph& subgraph = estimated_solution.subgraphs[sg_idx];

        if (subgraph.granularity.width <= 0 || subgraph.granularity.height <= 0 ||
            subgraph.granularity.depth <= 0) {
            return absl::InvalidArgumentError("CostModel: subgraph granularity must be positive");
        }

        if (subgraph.granularity.width > problem.native_granularity.width ||
            subgraph.granularity.height > problem.native_granularity.height) {
            return absl::InvalidArgumentError(
                "CostModel: subgraph granularity width/height cannot exceed native granularity");
        }

        SubgraphMeta const meta = BuildSubgraphMeta(problem, subgraph, consumers_by_tensor);

        size_t reference_output_tensor = 0;
        if (!meta.final_outputs.empty()) {
            reference_output_tensor = *meta.final_outputs.begin();
        } else if (!meta.produced.empty()) {
            reference_output_tensor = *meta.produced.begin();
        } else {
            subgraph.subgraph_latency = 0.0;
            continue;
        }

        int64_t const out_w = problem.tensors[reference_output_tensor].width;
        int64_t const out_h = problem.tensors[reference_output_tensor].height;
        int64_t const tiles_w = CeilDiv(out_w, subgraph.granularity.width);
        int64_t const tiles_h = CeilDiv(out_h, subgraph.granularity.height);
        int64_t const num_spatial_tiles = tiles_w * tiles_h;

        auto traversal_order_or = BuildTraversalOrder(problem, subgraph, reference_output_tensor);
        if (!traversal_order_or.ok()) {
            return traversal_order_or.status();
        }
        const std::vector<int64_t>& traversal_order = traversal_order_or.value();

        int64_t split_k_full = 1;
        bool has_final_matmul = false;
        std::set<size_t> split_k_output_tensors;

        for (size_t out_tensor_idx : meta.final_outputs) {
            int const p_op_idx = producer_op[out_tensor_idx];
            if (p_op_idx < 0) {
                continue;
            }

            const Op& p_op = problem.ops[static_cast<size_t>(p_op_idx)];
            if (p_op.op_type != "MatMul") {
                continue;
            }

            has_final_matmul = true;
            split_k_output_tensors.insert(out_tensor_idx);
            int64_t const curr_full_k = problem.tensors[p_op.inputs[0]].width;

            if (split_k_full == 1) {
                split_k_full = curr_full_k;
            } else if (split_k_full != curr_full_k) {
                return absl::InvalidArgumentError(
                    "CostModel: inconsistent final MatMul reduction sizes in one subgraph");
            }
        }

        int64_t num_k_steps = 1;
        if (has_final_matmul) {
            num_k_steps = CeilDiv(split_k_full, subgraph.granularity.depth);
        }

        std::set<size_t> locked_split_k_output_tensors;
        if (num_k_steps > 1) {
            locked_split_k_output_tensors = split_k_output_tensors;
        }

        std::set<size_t> retained_for_next =
            BuildNextRetainedSet(problem, subgraph, meta, prev_retained_tensors);
        std::set<size_t> retained_for_next_lookup = retained_for_next;

        TilesByTensor transient_resident;

        SubgraphLatency subgraph_latency = 0.0;

        for (int64_t order_idx = 0; order_idx < num_spatial_tiles; ++order_idx) {
            int64_t const spatial_tile_idx = traversal_order[static_cast<size_t>(order_idx)];

            for (int64_t k_step_idx = 0; k_step_idx < num_k_steps; ++k_step_idx) {
                int64_t k_start = 0;
                int64_t k_size = 1;

                if (has_final_matmul) {
                    k_start = k_step_idx * subgraph.granularity.depth;
                    k_size = std::min<int64_t>(subgraph.granularity.depth, split_k_full - k_start);
                }

                StepRequirements const reqs =
                    CollectStepRequirements(problem, subgraph, meta, producer_op, spatial_tile_idx,
                                            tiles_w, k_start, k_size);

                double const step_compute_time =
                    ComputeStepComputeTime(problem, subgraph, reqs.op_requirements);

                TilesByTensor sticky_resident =
                    BuildStickyResidentFromRetained(problem, prev_retained_tensors);

                TilesByTensor resident_at_step = sticky_resident;
                AddTiles(resident_at_step, transient_resident);

                int64_t const memory_in_elements =
                    ComputeMissingArea(reqs.required_boundary_inputs, resident_at_step);

                bool const is_last_k_step = (k_step_idx == (num_k_steps - 1));

                TilesByTensor write_outputs;
                for (const auto& [tensor_idx, tiles] : reqs.required_boundary_outputs) {
                    if (retained_for_next_lookup.contains(tensor_idx)) {
                        continue;
                    }
                    if (locked_split_k_output_tensors.contains(tensor_idx) && !is_last_k_step) {
                        continue;
                    }

                    auto& out_tiles = write_outputs[tensor_idx];
                    out_tiles.insert(out_tiles.end(), tiles.begin(), tiles.end());
                }

                int64_t const memory_out_elements = ComputeMapArea(write_outputs);

                double const memory_time =
                    static_cast<double>(memory_in_elements + memory_out_elements) /
                    static_cast<double>(problem.slow_memory_bandwidth);

                subgraph_latency += std::max(step_compute_time, memory_time);

                TilesByTensor transient_next = reqs.required_boundary_inputs;
                if (!is_last_k_step) {
                    AddLockedSplitKOutputs(reqs.required_boundary_outputs,
                                           locked_split_k_output_tensors, transient_next);
                }

                transient_resident = std::move(transient_next);
            }
        }

        subgraph.subgraph_latency = subgraph_latency;
        total_latency += subgraph_latency;

        prev_retained_tensors = std::move(retained_for_next);
    }

    return std::tuple<Solution, SubgraphLatency>{estimated_solution, total_latency};
}

} // namespace mlsys
