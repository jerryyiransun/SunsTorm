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

// Group tiles by tensor id so we can compute union areas tensor-by-tensor.
using TilesByTensor = std::unordered_map<size_t, std::vector<Tile>>;

struct SubgraphMeta {
    // Ops that belong to the current subgraph (for quick membership checks).
    std::set<size_t> op_set;

    // Tensors produced by any op in the subgraph.
    std::set<size_t> produced;
    // Tensors consumed by any op in the subgraph.
    std::set<size_t> consumed;

    // Produced tensors not consumed again inside this subgraph.
    // These are the subgraph's logical outputs for tile traversal.
    std::set<size_t> final_outputs;

    // Tensors consumed in this subgraph but produced outside it.
    std::set<size_t> boundary_inputs;
    // Tensors produced in this subgraph and needed outside it (or final graph outputs).
    // Not always the same as final_outputs:
    // a tensor can be consumed both inside this subgraph and by a later subgraph.
    std::set<size_t> boundary_outputs;

    // Produced+consumed fully inside the same subgraph with no external visibility.
    // These are ephemeral: compute dependencies only, no slow-memory traffic accounting.
    std::set<size_t> ephemeral_tensors;
};

struct OpStepRequirement {
    // Output tiles of this op required in the current execution step.
    std::vector<Tile> output_tiles;

    // Required reduction depth for this step.
    // Pointwise => 1
    // MatMul final-output path => current k chunk
    // MatMul intermediate path => full K
    int64_t req_k = 1;
};

struct StepRequirements {
    // Boundary input tiles needed by this step.
    TilesByTensor required_boundary_inputs;
    // Boundary output tiles produced by this step.
    TilesByTensor required_boundary_outputs;

    // Per-op compute footprint used to derive step arithmetic time.
    std::unordered_map<size_t, OpStepRequirement> op_requirements;
};

// Integer ceil(a / b) for positive dimensions in this model.
auto CeilDiv(int64_t a, int64_t b) -> int64_t {
    return (a + b - 1) / b;
}

// Adds a tile into a tensor-indexed tile map if tile has non-zero area.
void AddTile(TilesByTensor& map, const Tile& tile) {
    if (tile.area() <= 0) {
        return;
    }
    map[tile.tensor_idx].push_back(tile);
}

// Appends all tile lists from src into dst (same tensor ids are concatenated).
void AddTiles(TilesByTensor& dst, const TilesByTensor& src) {
    for (const auto& [tensor_idx, tiles] : src) {
        auto& out = dst[tensor_idx];
        out.insert(out.end(), tiles.begin(), tiles.end());
    }
}

// Clamps tile coordinates to valid tensor bounds and keeps half-open semantics.
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

// Flattens a tensor-indexed tile map into a single vector.
auto FlattenTiles(const TilesByTensor& map) -> std::vector<Tile> {
    std::vector<Tile> out;
    for (const auto& [_, tiles] : map) {
        out.insert(out.end(), tiles.begin(), tiles.end());
    }
    return out;
}

// Computes union area of every tile in the map.
auto ComputeMapArea(const TilesByTensor& map) -> int64_t {
    return Tile::compute_non_overlapping_area(FlattenTiles(map));
}

// Computes how much additional area must be fetched for `required` given `resident`.
// For each tensor, this is: union(required ∪ resident) - union(resident).
auto ComputeMissingArea(const TilesByTensor& required, const TilesByTensor& resident) -> int64_t {
    int64_t missing = 0;

    for (const auto& [tensor_idx, req_tiles] : required) {
        auto it = resident.find(tensor_idx);
        std::vector<Tile> resident_tiles;
        if (it != resident.end()) {
            resident_tiles = it->second;
        }

        int64_t resident_area = Tile::compute_non_overlapping_area(resident_tiles);

        std::vector<Tile> combined = req_tiles;
        combined.insert(combined.end(), resident_tiles.begin(), resident_tiles.end());

        int64_t combined_area = Tile::compute_non_overlapping_area(combined);
        missing += std::max<int64_t>(0, combined_area - resident_area);
    }

    return missing;
}

// Builds tensor -> producer op index map (-1 for graph inputs).
auto BuildProducerMap(const Problem& problem) -> std::vector<int> {
    std::vector<int> producer(problem.tensors.size(), -1);
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        producer[problem.ops[op_idx].outputs[0]] = static_cast<int>(op_idx);
    }
    return producer;
}

// Builds tensor -> list of consumer op indices map.
auto BuildConsumersMap(const Problem& problem) -> std::vector<std::vector<size_t>> {
    std::vector<std::vector<size_t>> consumers(problem.tensors.size());
    for (size_t op_idx = 0; op_idx < problem.ops.size(); ++op_idx) {
        for (size_t t_idx : problem.ops[op_idx].inputs) {
            consumers[t_idx].push_back(op_idx);
        }
    }
    return consumers;
}

// Returns true if the given op consumes tensor_idx as one of its inputs.
auto OpConsumesTensor(const Problem& problem, size_t op_idx, size_t tensor_idx) -> bool {
    if (op_idx >= problem.ops.size()) {
        return false;
    }
    for (size_t in_tensor_idx : problem.ops[op_idx].inputs) {
        if (in_tensor_idx == tensor_idx) {
            return true;
        }
    }
    return false;
}

// Returns true if this produced tensor must survive the current subgraph boundary
// to satisfy a future consumer execution.
//
// Why schedule-aware:
// the same producer op can appear in later subgraphs (recomputation). If a later
// consumer is preceded by that producer within the same subgraph, the consumer is
// satisfied by local recomputation and does not require carrying this boundary value.
auto NeedsBoundaryCarryForFutureConsumer(const Problem& problem, const Solution& solution,
                                         const std::vector<int>& producer_op, size_t tensor_idx,
                                         size_t current_subgraph_idx) -> bool {
    if (tensor_idx >= producer_op.size()) {
        return false;
    }
    int const producer_op_idx_signed = producer_op[tensor_idx];
    if (producer_op_idx_signed < 0) {
        return false;
    }
    size_t const producer_op_idx = static_cast<size_t>(producer_op_idx_signed);

    for (size_t future_sg_idx = current_subgraph_idx + 1; future_sg_idx < solution.subgraphs.size();
         ++future_sg_idx) {
        const Subgraph& future_sg = solution.subgraphs[future_sg_idx];

        // Pre-collect producer positions in this future subgraph so we can check
        // "producer before consumer" quickly.
        std::vector<size_t> producer_positions;
        for (size_t pos = 0; pos < future_sg.ops.size(); ++pos) {
            if (future_sg.ops[pos] == producer_op_idx) {
                producer_positions.push_back(pos);
            }
        }

        for (size_t consumer_pos = 0; consumer_pos < future_sg.ops.size(); ++consumer_pos) {
            size_t const op_idx = future_sg.ops[consumer_pos];
            if (!OpConsumesTensor(problem, op_idx, tensor_idx)) {
                continue;
            }

            bool has_local_producer_before_consumer = false;
            for (size_t producer_pos : producer_positions) {
                if (producer_pos < consumer_pos) {
                    has_local_producer_before_consumer = true;
                    break;
                }
            }

            if (!has_local_producer_before_consumer) {
                return true;
            }
        }
    }

    return false;
}

// Derives subgraph-local tensor role classification used by the estimator.
auto BuildSubgraphMeta(const Problem& problem, const Subgraph& subgraph,
                       const std::vector<std::vector<size_t>>& consumers_by_tensor)
    -> SubgraphMeta {
    SubgraphMeta meta;

    // Gather produced/consumed sets and op membership.
    for (size_t op_idx : subgraph.ops) {
        meta.op_set.insert(op_idx);

        size_t out = problem.ops[op_idx].outputs[0];
        meta.produced.insert(out);

        for (size_t in : problem.ops[op_idx].inputs) {
            meta.consumed.insert(in);
        }
    }

    // Final outputs are produced tensors not consumed again in the same subgraph.
    for (size_t t_idx : meta.produced) {
        if (!meta.consumed.contains(t_idx)) {
            meta.final_outputs.insert(t_idx);
        }
    }

    // Boundary inputs are consumed tensors that are not produced internally.
    for (size_t t_idx : meta.consumed) {
        if (!meta.produced.contains(t_idx)) {
            meta.boundary_inputs.insert(t_idx);
        }
    }

    // Initial boundary-output classification from pure graph topology.
    // The estimator refines this later with schedule-aware recomputation analysis.
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

    // Ephemeral tensors are produced+consumed internally and never escape.
    for (size_t t_idx : meta.produced) {
        bool const consumed_inside = meta.consumed.contains(t_idx);
        if (consumed_inside && !meta.boundary_outputs.contains(t_idx)) {
            meta.ephemeral_tensors.insert(t_idx);
        }
    }

    return meta;
}

// Returns explicit spatial traversal order for this subgraph.
// If traversal is not provided, default row-major order is used.
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

// Converts a linear tile id (within traversal grid) to actual clipped tile bounds.
// Only used for output tensors with specific spatial granularity, so we can use that granularity to
// compute tile coordinates.
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

// Builds full-tensor resident coverage for retained tensors at subgraph boundaries.
auto BuildStickyResidentFromRetained(const Problem& problem, const std::set<size_t>& retained)
    -> TilesByTensor {
    TilesByTensor sticky;
    for (size_t t_idx : retained) {
        // Defensive guard for malformed retain lists; valid solutions never hit this.
        if (t_idx >= problem.tensors.size()) {
            continue;
        }
        const Tensor& t = problem.tensors[t_idx];
        AddTile(sticky, Tile{.tensor_idx = t_idx, .x0 = 0, .x1 = t.width, .y0 = 0, .y1 = t.height});
    }
    return sticky;
}

// Carries split-k output accumulators across k steps inside the same spatial tile.
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

// Computes retained full tensors that should survive into the next subgraph.
// Ephemeral tensors are intentionally ignored here.
auto BuildNextRetainedSet(const Problem& problem, const Subgraph& subgraph,
                          const SubgraphMeta& meta, const std::set<size_t>& prev_retained)
    -> std::set<size_t> {
    std::set<size_t> next;

    for (size_t t_idx : subgraph.tensors_to_retain) {
        if (t_idx >= problem.tensors.size()) {
            continue;
        }
        // Ephemeral tensors never cross subgraph boundaries by definition, so retaining
        // them has no semantic effect and would only distort residency accounting.
        if (meta.ephemeral_tensors.contains(t_idx)) {
            continue;
        }

        // We only retain tensors touched by this step context (consumed/produced/previously
        // retained).
        // Technically we could retain untouched tensors, but that would make no sense.
        bool const touched = meta.consumed.contains(t_idx) || meta.produced.contains(t_idx) ||
                             prev_retained.contains(t_idx);
        if (touched) {
            next.insert(t_idx);
        }
    }

    return next;
}

// Performs backward propagation for one execution step:
// - starts from subgraph final output tile(s)
// - traverses producers backward inside subgraph
// - emits boundary input/output tile demands
// - records per-op required output area + reduction depth for compute-time math
//
// Spatial tile and k-range (`k_start`, `k_size`) determine the exact per-step shape.
auto CollectStepRequirements(const Problem& problem, const Subgraph& subgraph,
                             const SubgraphMeta& meta, const std::vector<int>& producer_op,
                             int64_t spatial_tile_linear_idx, int64_t tiles_w, int64_t k_start,
                             int64_t k_size) -> StepRequirements {
    StepRequirements reqs;

    // BFS-like queue of tiles to propagate backward through producer ops.
    std::vector<Tile> queue;

    // Seed queue with current spatial tile for each final output tensor.
    for (size_t out_tensor_idx : meta.final_outputs) {
        Tile out_tile = BuildTileForLinearIndex(problem, subgraph, out_tensor_idx,
                                                spatial_tile_linear_idx, tiles_w);
        // Defensive clamp for malformed zero-sized tensors.
        if (out_tile.area() <= 0) {
            continue;
        }

        queue.push_back(out_tile);
    }

    size_t head = 0;
    while (head < queue.size()) {
        Tile curr_tile = queue[head++];

        // Any externally visible produced tensor touched in this step may need writeback.
        // We do this here (rather than only at seed time) so fan-out tensors that are
        // consumed both internally and externally are correctly accounted.
        if (meta.boundary_outputs.contains(curr_tile.tensor_idx) &&
            !meta.ephemeral_tensors.contains(curr_tile.tensor_idx)) {
            AddTile(reqs.required_boundary_outputs, curr_tile);
        }

        int const p_op_idx = producer_op[curr_tile.tensor_idx];
        // Graph input or malformed producer map: no local producer to continue with.
        if (p_op_idx < 0) {
            continue;
        }

        size_t const op_idx = static_cast<size_t>(p_op_idx);
        // Producer exists but is not inside current subgraph.
        if (!meta.op_set.contains(op_idx)) {
            continue;
        }

        const Op& op = problem.ops[op_idx];

        // Record this op's output footprint for compute-time aggregation.
        OpStepRequirement& op_req = reqs.op_requirements[op_idx];
        // The same op output tile can be discovered via multiple backward paths; we keep
        // all tiles and take a non-overlapping union later to avoid double-counting compute.
        op_req.output_tiles.push_back(curr_tile);

        // Determine required reduction size for this op in this step.
        int64_t req_k = 1;
        if (op.op_type == "MatMul") {
            size_t const lhs_idx = op.inputs[0];
            int64_t const full_k = problem.tensors[lhs_idx].width;

            bool const is_final_output_tensor = meta.final_outputs.contains(curr_tile.tensor_idx);
            req_k = is_final_output_tensor ? k_size : full_k;
        }
        op_req.req_k = std::max(op_req.req_k, req_k);

        if (op.op_type == "Pointwise") {
            // Pointwise inputs have exactly the same spatial bounds as outputs.
            for (size_t in_tensor_idx : op.inputs) {
                Tile in_tile{.tensor_idx = in_tensor_idx,
                             .x0 = curr_tile.x0,
                             .x1 = curr_tile.x1,
                             .y0 = curr_tile.y0,
                             .y1 = curr_tile.y1};
                in_tile = ClipTileToTensor(problem, in_tile);
                // Defensive guard for malformed or degenerate tensor dimensions.
                if (in_tile.area() <= 0) {
                    continue;
                }

                if (meta.produced.contains(in_tensor_idx)) {
                    // Internal dependency: keep traversing backward.
                    queue.push_back(in_tile);
                } else {
                    // Boundary dependency: contributes to slow-memory read demand.
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

            // Final-output MatMul uses split-k range for this step.
            // Intermediate MatMul uses full K because it must be fully reduced before use.
            int64_t const k0 = is_final_output_tensor ? k_start : 0;
            int64_t const k1 = is_final_output_tensor ? (k_start + k_size) : full_inner_k;

            // MatMul mapping (height is y axis, width is x axis):
            // lhs[h, k], rhs[k, w], out[h, w]
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

// Computes arithmetic time for one step from per-op demand.
// We scale by:
// - output area covered in this step
// - required k depth (for MatMul)
// - native padding ratio when granularity is smaller than native.
auto ComputeStepComputeTime(const Problem& problem, const Subgraph& subgraph,
                            const std::unordered_map<size_t, OpStepRequirement>& op_requirements)
    -> double {
    double const native_w = static_cast<double>(problem.native_granularity.width);
    double const native_h = static_cast<double>(problem.native_granularity.height);

    double const gran_w = static_cast<double>(subgraph.granularity.width);
    double const gran_h = static_cast<double>(subgraph.granularity.height);

    // PROBLEM.md specifies: if spatial granularity is below native, compute is padded
    // up to native. Since we enforce granularity.width/height <= native, this models
    // exactly that "pad up to native" behavior.
    double const padded_w =
        static_cast<double>(std::max(problem.native_granularity.width, subgraph.granularity.width));
    double const padded_h = static_cast<double>(
        std::max(problem.native_granularity.height, subgraph.granularity.height));

    double const padding_ratio = (padded_w * padded_h) / (gran_w * gran_h);

    double compute_time = 0.0;

    for (const auto& [op_idx, req] : op_requirements) {
        const Op& op = problem.ops[op_idx];

        // Non-overlapping output area this op contributes in current step.
        // This captures edge tiles (clipped area) and avoids duplicate counting when
        // backward traversal reaches the same op-output tile through multiple paths.
        int64_t const required_output_area = Tile::compute_non_overlapping_area(req.output_tiles);
        if (required_output_area <= 0) {
            continue;
        }

        int64_t full_k = 1;
        int64_t req_k = 1;

        if (op.op_type == "MatMul") {
            full_k = problem.tensors[op.inputs[0]].width;
            req_k = req.req_k;
        }

        // Fraction of a native full op represented by this step:
        // spatial fraction (output area) * reduction fraction (req_k / full_k).
        double const numerator =
            static_cast<double>(required_output_area) * static_cast<double>(req_k);
        double const denominator = native_w * native_h * static_cast<double>(full_k);

        compute_time +=
            static_cast<double>(op.base_cost) * (numerator / denominator) * padding_ratio;
    }

    return compute_time;
}

} // namespace

// Returns area in tensor elements for this tile.
auto Tile::area() const -> int64_t {
    int64_t const w = std::max<int64_t>(0, x1 - x0);
    int64_t const h = std::max<int64_t>(0, y1 - y0);
    return w * h;
}

// Exact union area across possibly overlapping tiles.
// Algorithm: for each tensor independently, do x-axis sweep-line and maintain active y-intervals.
auto Tile::compute_non_overlapping_area(const std::vector<Tile>& tiles) -> int64_t {
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
            int type; // +1 when rectangle opens at x0, -1 when it closes at x1.
        };

        std::vector<Event> events;
        events.reserve(tensor_tiles.size() * 2);

        for (const Tile& t : tensor_tiles) {
            events.push_back(Event{.x = t.x0, .y0 = t.y0, .y1 = t.y1, .type = +1});
            events.push_back(Event{.x = t.x1, .y0 = t.y0, .y1 = t.y1, .type = -1});
        }

        // Primary sort by x, secondary keeps +1 before -1 at same x to avoid transient undercount.
        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
            if (a.x != b.x) {
                return a.x < b.x;
            }
            return a.type > b.type;
        });

        // Active interval multiset represented as interval -> refcount.
        std::map<std::pair<int64_t, int64_t>, int> active;

        // Computes total covered y-length from current active interval set.
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
                    // Disjoint interval; finalize previous merged segment.
                    total += std::max<int64_t>(0, cur_y1 - cur_y0);
                    cur_y0 = y0;
                    cur_y1 = y1;
                } else {
                    // Overlap/touch; extend merged segment.
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

            // Area contributed by the slab [prev_x, x).
            if (dx > 0) {
                total_area += dx * covered_y();
            }

            // Apply all events at this x position.
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

// Estimates all subgraph latencies for a proposed solution and returns
// both the updated solution and total latency.
//
// State model used per subgraph:
// - required_boundary_tiles: demand for the current step
// - resident_boundary_tiles: what is already in fast memory (sticky + transient)
// - retained_tensors_next_subgraph: full tensors kept across subgraph boundary
//
// Ephemeral tensors are excluded from slow-memory fetch/write accounting.
auto CostModel::estimate(const Solution& solution)
    -> StatusOr<std::tuple<Solution, SubgraphLatency>> {
    if (problem_.slow_memory_bandwidth <= 0) {
        return absl::InvalidArgumentError("CostModel: slow_memory_bandwidth must be positive");
    }

    // tensor -> producer op index
    std::vector<int> const producer_op = BuildProducerMap(problem_);
    // tensor -> all consumer op indices
    std::vector<std::vector<size_t>> const consumers_by_tensor = BuildConsumersMap(problem_);

    Solution estimated_solution = solution;
    SubgraphLatency total_latency = 0.0;

    // Full tensors retained from previous subgraph boundary.
    std::set<size_t> prev_retained_tensors;

    for (size_t sg_idx = 0; sg_idx < estimated_solution.subgraphs.size(); ++sg_idx) {
        Subgraph& subgraph = estimated_solution.subgraphs[sg_idx];

        if (subgraph.granularity.width <= 0 || subgraph.granularity.height <= 0 ||
            subgraph.granularity.depth <= 0) {
            return absl::InvalidArgumentError("CostModel: subgraph granularity must be positive");
        }

        // As requested: keep w/h at or below native granularity.
        if (subgraph.granularity.width > problem_.native_granularity.width ||
            subgraph.granularity.height > problem_.native_granularity.height) {
            return absl::InvalidArgumentError(
                "CostModel: subgraph granularity width/height cannot exceed native granularity");
        }

        // Classify tensor roles for this subgraph.
        SubgraphMeta meta = BuildSubgraphMeta(problem_, subgraph, consumers_by_tensor);

        // Refine boundary outputs with schedule-aware recomputation behavior.
        // A produced tensor must escape only if:
        // 1) it is a graph output, or
        // 2) some future consumer needs it before a local recomputation.
        meta.boundary_outputs.clear();
        for (size_t t_idx : meta.produced) {
            bool const graph_output = consumers_by_tensor[t_idx].empty();
            bool const needed_for_future_consumer = NeedsBoundaryCarryForFutureConsumer(
                problem_, estimated_solution, producer_op, t_idx, sg_idx);
            if (graph_output || needed_for_future_consumer) {
                meta.boundary_outputs.insert(t_idx);
            }
        }

        // Recompute ephemeral set using the refined boundary-output definition.
        meta.ephemeral_tensors.clear();
        for (size_t t_idx : meta.produced) {
            bool const consumed_inside = meta.consumed.contains(t_idx);
            if (consumed_inside && !meta.boundary_outputs.contains(t_idx)) {
                meta.ephemeral_tensors.insert(t_idx);
            }
        }

        // Choose one output tensor to define spatial tile grid dimensions.
        // The estimator requires all final outputs to share the same tile count so a
        // single traversal order can index all of them consistently.
        size_t reference_output_tensor = 0;
        if (!meta.final_outputs.empty()) {
            reference_output_tensor = *meta.final_outputs.begin();
        } else if (!meta.produced.empty()) {
            reference_output_tensor = *meta.produced.begin();
        } else {
            subgraph.subgraph_latency = 0.0;
            continue;
        }

        int64_t const out_w = problem_.tensors[reference_output_tensor].width;
        int64_t const out_h = problem_.tensors[reference_output_tensor].height;
        int64_t const tiles_w = CeilDiv(out_w, subgraph.granularity.width);
        int64_t const tiles_h = CeilDiv(out_h, subgraph.granularity.height);
        int64_t const num_spatial_tiles = tiles_w * tiles_h;
        for (size_t out_tensor_idx : meta.final_outputs) {
            int64_t const curr_w = problem_.tensors[out_tensor_idx].width;
            int64_t const curr_h = problem_.tensors[out_tensor_idx].height;
            int64_t const curr_tiles_w = CeilDiv(curr_w, subgraph.granularity.width);
            int64_t const curr_tiles_h = CeilDiv(curr_h, subgraph.granularity.height);
            if (curr_tiles_w * curr_tiles_h != num_spatial_tiles) {
                return absl::InvalidArgumentError(
                    "CostModel: inconsistent final output tile counts in one subgraph");
            }
        }

        auto traversal_order_or = BuildTraversalOrder(problem_, subgraph, reference_output_tensor);
        if (!traversal_order_or.ok()) {
            return traversal_order_or.status();
        }
        // Maps loop index -> spatial tile id in traversal sequence.
        const std::vector<int64_t>& traversal_order = traversal_order_or.value();

        // Split-k analysis on final outputs.
        // If final outputs are MatMul-produced and depth < full K, each spatial tile has multiple k
        // steps.
        int64_t split_k_full = 1; // Shared full-K among final MatMul outputs.
        bool has_final_matmul = false;
        std::set<size_t> split_k_output_tensors;

        // We use the same split-k traversal for all final outputs, so we must check they are
        // compatible
        for (size_t out_tensor_idx : meta.final_outputs) {
            int const p_op_idx = producer_op[out_tensor_idx];
            // Defensive guard for malformed producer maps.
            if (p_op_idx < 0) {
                continue;
            }

            const Op& p_op = problem_.ops[static_cast<size_t>(p_op_idx)];
            if (p_op.op_type != "MatMul") {
                continue;
            }

            has_final_matmul = true;
            split_k_output_tensors.insert(out_tensor_idx);
            int64_t const curr_full_k = problem_.tensors[p_op.inputs[0]].width;

            if (split_k_full == 1) {
                split_k_full = curr_full_k;
            } else if (split_k_full != curr_full_k) {
                // When multiple final outputs come from different matmuls with different K, we
                // cannot fully compute all the outputs
                return absl::InvalidArgumentError(
                    "CostModel: inconsistent final MatMul reduction sizes in one subgraph");
            }
        }

        int64_t num_k_steps = 1;
        if (has_final_matmul) {
            num_k_steps = CeilDiv(split_k_full, subgraph.granularity.depth);
        }

        // Output accumulators that must stay resident between k slices of the same spatial tile.
        std::set<size_t> locked_split_k_output_tensors;
        if (num_k_steps > 1) {
            locked_split_k_output_tensors = split_k_output_tensors;
        }

        // What we intend to keep across this subgraph boundary.
        std::set<size_t> retained_for_next =
            BuildNextRetainedSet(problem_, subgraph, meta, prev_retained_tensors);

        // Slices kept only across adjacent steps inside current subgraph.
        // This captures short-term reuse from traversal locality and split-k accumulation.
        TilesByTensor transient_resident;

        SubgraphLatency subgraph_latency = 0.0;

        // Outer loop: spatial traversal.
        for (int64_t order_idx = 0; order_idx < num_spatial_tiles; ++order_idx) {
            int64_t const spatial_tile_idx = traversal_order[static_cast<size_t>(order_idx)];

            // Inner loop: split-k traversal for this spatial tile.
            for (int64_t k_step_idx = 0; k_step_idx < num_k_steps; ++k_step_idx) {
                int64_t k_start = 0; // inclusive K index for this step
                int64_t k_size = 1;  // number of K elements in this step

                if (has_final_matmul) {
                    k_start = k_step_idx * subgraph.granularity.depth;
                    // Tail split-k chunk can be smaller when full_k is not divisible by depth.
                    k_size = std::min<int64_t>(subgraph.granularity.depth, split_k_full - k_start);
                }

                // Backward-propagated tile requirements for this exact (spatial,k) step.
                StepRequirements const reqs =
                    CollectStepRequirements(problem_, subgraph, meta, producer_op, spatial_tile_idx,
                                            tiles_w, k_start, k_size);
                // Arithmetic component for this step.
                double const step_compute_time =
                    ComputeStepComputeTime(problem_, subgraph, reqs.op_requirements);

                // Sticky residency from previous subgraph boundary (full tensors).
                TilesByTensor sticky_resident =
                    BuildStickyResidentFromRetained(problem_, prev_retained_tensors);

                // Effective resident set for this step = sticky + intra-subgraph transient.
                TilesByTensor resident_at_step = sticky_resident;
                AddTiles(resident_at_step, transient_resident);

                // Slow-memory reads needed by boundary inputs that are not fully resident.
                int64_t const memory_in_elements =
                    ComputeMissingArea(reqs.required_boundary_inputs, resident_at_step);

                bool const is_last_k_step = (k_step_idx == (num_k_steps - 1));

                // Boundary outputs written this step.
                // Suppress write for retained outputs and intermediate split-k accumulations.
                TilesByTensor write_outputs;
                for (const auto& [tensor_idx, tiles] : reqs.required_boundary_outputs) {
                    if (retained_for_next.contains(tensor_idx)) {
                        continue;
                    }
                    if (locked_split_k_output_tensors.contains(tensor_idx) && !is_last_k_step) {
                        continue;
                    }
                    auto& write_tiles = write_outputs[tensor_idx];
                    write_tiles.insert(write_tiles.end(), tiles.begin(), tiles.end());
                }

                int64_t const memory_out_elements = ComputeMapArea(write_outputs);

                double const memory_time =
                    static_cast<double>(memory_in_elements + memory_out_elements) /
                    static_cast<double>(problem_.slow_memory_bandwidth);

                // Roofline step time.
                subgraph_latency += std::max(step_compute_time, memory_time);

                // Update transient resident set for next step.
                // We keep current boundary inputs for possible immediate reuse.
                TilesByTensor transient_next = reqs.required_boundary_inputs;
                // For split-k, also keep output accumulators until final k step.
                if (!is_last_k_step) {
                    AddLockedSplitKOutputs(reqs.required_boundary_outputs,
                                           locked_split_k_output_tensors, transient_next);
                }

                transient_resident = std::move(transient_next);
            }
        }

        subgraph.subgraph_latency = subgraph_latency;
        total_latency += subgraph_latency;

        // Carry retained full tensors into next subgraph.
        prev_retained_tensors = std::move(retained_for_next);
    }

    return std::tuple<Solution, SubgraphLatency>{estimated_solution, total_latency};
}

} // namespace mlsys
