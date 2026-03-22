#pragma once
#include <optional>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mlsys.h"

using namespace absl;
namespace mlsys {
class Solver {
  public:
    virtual ~Solver() = default;
    virtual auto solve(const Problem& problem) -> StatusOr<Solution> = 0;
};

class BaseSolver : public Solver {
  public:
    auto solve(const Problem& problem) -> StatusOr<Solution> override;
};

class BruteForceSolver : public Solver {
  public:
    auto solve(const Problem& problem) -> StatusOr<Solution> override;
};

} // namespace mlsys
