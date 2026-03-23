#include "fuser.h"

namespace mlsys {

auto BruteForceFuser::fuse(const Problem& problem) -> StatusOr<std::vector<Solution>> {
    std::vector<Op> ops = problem.ops;

    return std::vector<Solution>();
}

} // namespace mlsys