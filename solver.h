#pragma once

#include "absl/status/statusor.h"
#include "fuser.h"
#include "mlsys.h"

#include <optional>
#include <string>

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

class GreedySolver : public Solver {
  public:
    GreedySolver();
    explicit GreedySolver(std::string anytime_output_path);
    explicit GreedySolver(GreedyFuserConfig config);
    GreedySolver(GreedyFuserConfig config, std::string anytime_output_path);

    auto solve(const Problem& problem) -> StatusOr<Solution> override;

  private:
    bool use_problem_sized_config_ = false;
    GreedyFuserConfig config_;
    std::optional<std::string> anytime_output_path_;
};

} // namespace mlsys
