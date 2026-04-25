#include "tiler.h"

#include "tiler_common.h"

#include "absl/status/status.h"

#include <cstdint>
#include <optional>
#include <set>
#include <vector>

namespace mlsys {

auto GreedyTiler::tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution> {
    Solution tiled_solution = solution;
    std::vector<int> producer_op = tiler_internal::BuildProducerMap(problem);
    std::set<size_t> prev_retained_tensors;

    for (size_t sg_idx = 0; sg_idx < tiled_solution.subgraphs.size(); ++sg_idx) {
        auto& sg = tiled_solution.subgraphs[sg_idx];
        Tensor const output_shape = tiler_internal::MaxFinalOutputShape(problem, sg);
        std::vector<int64_t> width_candidates = tiler_internal::CandidateTileSizes(
            output_shape.width, problem.native_granularity.width);
        std::vector<int64_t> height_candidates = tiler_internal::CandidateTileSizes(
            output_shape.height, problem.native_granularity.height);
        std::optional<int64_t> const split_k_matmul_depth =
            tiler_internal::SplitKMatMulDepth(problem, sg);
        std::vector<int64_t> depth_candidates =
            split_k_matmul_depth.has_value()
                ? tiler_internal::CandidateTileSizes(split_k_matmul_depth.value(),
                                                     problem.native_granularity.depth)
                : std::vector<int64_t>{1};

        tiler_internal::CostGuidedDivisorState state;

        while (true) {
            Granularity const current = tiler_internal::GranularityFromState(
                width_candidates, height_candidates, depth_candidates, state);
            tiler_internal::ApplyGranularityAndTraversal(problem, tiled_solution, sg_idx, current);

            if (tiler_internal::FitsFastMemory(problem, tiled_solution, sg_idx,
                                               prev_retained_tensors, producer_op)) {
                break;
            }

            enum class CandidateDimension {
                kDepth,
                kWidth,
                kHeight,
            };

            std::optional<CandidateDimension> dimension_to_advance;
            int64_t best_dimension_size = 0;
            auto consider_dimension = [&](CandidateDimension dimension, int64_t current_size,
                                          bool can_advance) {
                if (!can_advance) {
                    return;
                }
                if (!dimension_to_advance.has_value() || current_size > best_dimension_size) {
                    dimension_to_advance = dimension;
                    best_dimension_size = current_size;
                }
            };

            // Tie order is depth, then width, then height because equal sizes keep the
            // first considered dimension.
            consider_dimension(CandidateDimension::kDepth, current.depth,
                               state.depth_idx + 1 < depth_candidates.size());
            consider_dimension(CandidateDimension::kWidth, current.width,
                               state.width_idx + 1 < width_candidates.size());
            consider_dimension(CandidateDimension::kHeight, current.height,
                               state.height_idx + 1 < height_candidates.size());

            if (!dimension_to_advance.has_value()) {
                return absl::ResourceExhaustedError(
                    "Cannot fit working set even at minimum tile size");
            }

            switch (dimension_to_advance.value()) {
            case CandidateDimension::kDepth:
                ++state.depth_idx;
                break;
            case CandidateDimension::kWidth:
                ++state.width_idx;
                break;
            case CandidateDimension::kHeight:
                ++state.height_idx;
                break;
            }
        }

        prev_retained_tensors =
            std::set<size_t>(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end());
    }

    return tiled_solution;
}

} // namespace mlsys
