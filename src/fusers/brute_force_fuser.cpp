#include "fuser.h"

#include "fuser_common.h"

namespace mlsys {

auto BruteForceFuser::fuse(const Problem& problem) -> StatusOr<std::vector<Solution>> {
    return fuser_internal::RunBruteForceFuser(problem);
}

} // namespace mlsys
