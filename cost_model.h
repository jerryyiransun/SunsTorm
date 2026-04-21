#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;

namespace mlsys {

class Tile {
  public:
    // Tensor id that this rectangular tile belongs to.
    size_t tensor_idx;
    // Half-open horizontal range [x0, x1) in tensor coordinates.
    int64_t x0;
    int64_t x1;
    // Half-open vertical range [y0, y1) in tensor coordinates.
    int64_t y0;
    int64_t y1;

    // Returns tile area in tensor elements. Degenerate ranges return 0.
    [[nodiscard]] auto area() const -> int64_t;

    // Computes exact union area across all tiles using per-tensor sweep-line.
    // Tiles from different tensors are never merged together.
    [[nodiscard]] static auto compute_non_overlapping_area(const std::vector<Tile>& tiles)
        -> int64_t;
};

class CostModel {
  public:
    struct CacheStats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t estimate_subgraph_calls = 0;
    };

    explicit CostModel(Problem problem, bool enable_cache_stats = false);

    // Estimates each subgraph latency and returns:
    // 1) updated solution with subgraph_latency filled in
    // 2) total latency across subgraphs
    [[nodiscard]] auto estimate(const Solution& solution)
        -> StatusOr<std::tuple<Solution, SubgraphLatency>>;
    [[nodiscard]] auto estimate_subgraph_latency(const Solution& solution, size_t sg_idx,
                                                 const std::set<size_t>& prev_retained_tensors)
        -> StatusOr<SubgraphLatency>;
    [[nodiscard]] auto cache_stats() const -> CacheStats;

  private:
    struct SubgraphCacheKey {
        std::vector<size_t> ops;
        std::vector<size_t> tensors_to_retain;
        Granularity granularity;
        std::optional<TraversalOrder> traversal_order;
        std::vector<size_t> prev_retained_tensors;
        std::vector<std::vector<size_t>> suffix_ops;
        std::vector<std::vector<size_t>> suffix_tensors_to_retain;

        bool operator==(const SubgraphCacheKey& other) const = default;

        template <typename H> friend auto AbslHashValue(H h, const SubgraphCacheKey& key) -> H {
            return H::combine(std::move(h), key.ops, key.tensors_to_retain, key.granularity.width,
                              key.granularity.height, key.granularity.depth, key.traversal_order,
                              key.prev_retained_tensors, key.suffix_ops,
                              key.suffix_tensors_to_retain);
        }
    };

    auto estimate_subgraph(const Solution& solution, size_t sg_idx,
                           const std::set<size_t>& prev_retained_tensors)
        -> StatusOr<SubgraphLatency>;
    [[nodiscard]] auto build_subgraph_cache_key(const Solution& solution, size_t sg_idx,
                                                const std::set<size_t>& prev_retained_tensors) const
        -> SubgraphCacheKey;
    [[nodiscard]] auto
    compute_retained_for_next_subgraph(const Solution& solution, size_t sg_idx,
                                       const std::set<size_t>& prev_retained_tensors) const
        -> std::set<size_t>;

    Problem problem_;
    std::vector<int> producer_op_;
    std::vector<std::vector<size_t>> consumers_by_tensor_;
    std::set<size_t> pure_input_tensors_;
    std::set<size_t> pure_output_tensors_;
    std::unordered_map<SubgraphCacheKey, SubgraphLatency, absl::Hash<SubgraphCacheKey>>
        subgraph_latency_cache_;
    bool enable_cache_stats_ = false;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    uint64_t estimate_subgraph_calls_ = 0;
};

} // namespace mlsys
