#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;
namespace mlsys {
struct GreedyFuserConfig {
    int search_depth;
    int beam_width;
    double topk_failure_penalty;
};

class BruteForceFuser {
  public:
    auto fuse(const Problem& problem) -> StatusOr<std::vector<Solution>>;
};

class GreedyFuser {
  public:
    explicit GreedyFuser(GreedyFuserConfig config);
    auto fuse(const Problem& problem) -> StatusOr<Solution>;

  private:
    GreedyFuserConfig config_;
};

} // namespace mlsys
