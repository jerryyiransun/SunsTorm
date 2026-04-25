#include "tiler.h"

#include "tiler_common.h"

#include "absl/status/status.h"

#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mlsys {

auto BruteForceTiler::tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    std::vector<int> producer_op = tiler_internal::BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;

    for (size_t sg_idx = 0; sg_idx < tiled_solution.subgraphs.size(); ++sg_idx) {
        auto& sg = tiled_solution.subgraphs[sg_idx];
        std::string const cache_key =
            tiler_internal::BuildCacheKey(tiled_solution, sg_idx, prev_retained_tensors);
        auto cache_it = best_granularity_cache_.find(cache_key);
        if (cache_it != best_granularity_cache_.end()) {
            if (!cache_it->second.has_value()) {
                return absl::ResourceExhaustedError(
                    "Cannot fit working set even at minimum tile size");
            }

            sg.granularity = *cache_it->second;
            sg.traversal_order = tiler_internal::BuildSnakeTraversalOrder(problem, sg);
            prev_retained_tensors =
                std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
            continue;
        }

        std::vector<int64_t> width_candidates =
            tiler_internal::HalvingCandidates(problem.native_granularity.width);
        std::vector<int64_t> height_candidates =
            tiler_internal::HalvingCandidates(problem.native_granularity.height);
        std::optional<int64_t> const split_k_matmul_depth =
            tiler_internal::SplitKMatMulDepth(problem, sg);
        std::vector<int64_t> depth_candidates =
            split_k_matmul_depth.has_value()
                ? tiler_internal::HalvingCandidates(
                      std::min(split_k_matmul_depth.value(), problem.native_granularity.depth))
                : std::vector<int64_t>{1};

        Granularity best_granularity{.width = 0, .height = 0, .depth = 0};
        double best_score = std::numeric_limits<double>::infinity();

        // Exhaustively enumerate the halving-based candidate space for this subgraph.
        for (int64_t width : width_candidates) {
            for (int64_t height : height_candidates) {
                for (int64_t depth : depth_candidates) {
                    sg.granularity = {.width = width, .height = height, .depth = depth};
                    if (!tiler_internal::FitsFastMemory(problem, tiled_solution, sg_idx,
                                                        prev_retained_tensors, producer_op)) {
                        continue;
                    }

                    double const score = tiler_internal::EstimateCandidateScore(problem, sg);

                    if (score < best_score) {
                        best_score = score;
                        best_granularity = sg.granularity;
                    }
                }
            }
        }

        if (best_granularity.width == 0 || best_granularity.height == 0 ||
            best_granularity.depth == 0) {
            best_granularity_cache_[cache_key] = std::nullopt;
            return absl::ResourceExhaustedError("Cannot fit working set even at minimum tile size");
        }

        sg.granularity = best_granularity;
        sg.traversal_order = tiler_internal::BuildSnakeTraversalOrder(problem, sg);
        best_granularity_cache_[cache_key] = best_granularity;
        prev_retained_tensors =
            std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
    }

    return tiled_solution;
}

} // namespace mlsys
