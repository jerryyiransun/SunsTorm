#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;
namespace mlsys {
class BruteForceFuser {
  public:
    auto fuse(const Problem& problem) -> StatusOr<std::vector<Solution>>;
};

} // namespace mlsys
