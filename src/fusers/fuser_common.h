#pragma once

#include "fuser.h"
#include "mlsys.h"

#include "absl/status/statusor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlsys::fuser_internal {

auto ScoreGreedyPotentialForTestingImpl(GreedyPotentialScorer potential_scorer,
                                        int64_t saved_elements, size_t shared_tensor_count,
                                        bool is_retain, size_t retain_span,
                                        int64_t retained_elements, int64_t fast_memory_capacity,
                                        int64_t direct_boundary_elements) -> int64_t;
auto RunGreedyFuser(const GreedyFuserConfig& config, Tiler* tiler,
                    BestSolutionCallback best_solution_callback, const Problem& problem)
    -> absl::StatusOr<Solution>;
auto RunBruteForceFuser(const Problem& problem) -> absl::StatusOr<std::vector<Solution>>;

} // namespace mlsys::fuser_internal
