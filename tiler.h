#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;

namespace mlsys {

class Tiler {
  public:
    auto tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution>;
};

} // namespace mlsys
