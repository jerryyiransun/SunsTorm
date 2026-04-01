#pragma once

#include "absl/status/statusor.h"
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

class HeuristicSolver : public Solver {
  public:
    auto solve(const Problem& problem) -> StatusOr<Solution> override;
};

} // namespace mlsys
