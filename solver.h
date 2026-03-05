#ifndef MLSYS_SOLVER_H
#define MLSYS_SOLVER_H

#include <optional>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mlsys.h"

namespace mlsys {
class Solver {
public:
    virtual ~Solver() = default;
    virtual absl::StatusOr<Solution> Solve(const Problem& problem) = 0;
};

class BaseSolver : public Solver {
public:
    absl::StatusOr<Solution> Solve(const Problem& problem) override;
};

}  // namespace mlsys

#endif  // MLSYS_SOLVER_H