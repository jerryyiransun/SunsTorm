#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;
namespace mlsys {
class Tiler;

enum class GreedyPotentialScorer {
    kLegacyAverage,
    kMemoryTrafficDensity,
};

struct GreedyFuserConfig {
    int search_depth;
    int beam_width;
    double topk_failure_penalty;
    GreedyPotentialScorer potential_scorer = GreedyPotentialScorer::kMemoryTrafficDensity;
};

using BestSolutionCallback = std::function<absl::Status(const Solution&, TotalLatency)>;

class BruteForceFuser {
  public:
    auto fuse(const Problem& problem) -> StatusOr<std::vector<Solution>>;
};

class GreedyFuser {
  public:
    explicit GreedyFuser(GreedyFuserConfig config);
    GreedyFuser(GreedyFuserConfig config, BestSolutionCallback best_solution_callback);
    GreedyFuser(GreedyFuserConfig config, std::unique_ptr<Tiler> tiler);
    GreedyFuser(GreedyFuserConfig config, std::unique_ptr<Tiler> tiler,
                BestSolutionCallback best_solution_callback);
    ~GreedyFuser();

    auto fuse(const Problem& problem) -> StatusOr<Solution>;

  private:
    GreedyFuserConfig config_;
    std::unique_ptr<Tiler> tiler_;
    BestSolutionCallback best_solution_callback_;
};

} // namespace mlsys
