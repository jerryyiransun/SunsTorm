#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "mlsys.h"

using namespace absl;

namespace mlsys {

class Tiler {
  public:
    auto tile(const Problem& problem, const Solution& solution) -> StatusOr<Solution>;

  private:
    std::unordered_map<std::string, std::optional<Granularity>> best_granularity_cache_;
};

} // namespace mlsys
