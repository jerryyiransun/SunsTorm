#include "fuser_logging.h"
#include "mlsys.h"
#include "solver.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace mlsys;

namespace {

constexpr int kDefaultGreedySearchDepth = 0;
constexpr int kDefaultGreedyBeamWidth = 32;
constexpr double kDefaultGreedyAlpha = 0.04;
constexpr std::string_view kFuserLogDirOption = "--fuser-log-dir=";
constexpr std::string_view kGreedyBeamWidthOption = "--greedy-beam-width=";
constexpr std::string_view kGreedySearchDepthOption = "--greedy-search-depth=";
constexpr std::string_view kGreedyAlphaOption = "--greedy-alpha=";

struct RunSolverOptions {
    std::string solver_choice;
    std::string input_path;
    std::string output_path;
    std::string fuser_log_dir = "logs";
    std::optional<int> greedy_search_depth;
    std::optional<int> greedy_beam_width;
    std::optional<double> greedy_alpha;
};

enum class SolverKind {
    kGreedy,
    kBase,
    kHeuristic,
    kBruteForce,
};

void PrintUsage() {
    std::cerr << "Usage: ./run_solver <solver> <path_to_input.json> <path_to_output.json> "
                 "[--fuser-log-dir=<dir>] [--greedy-beam-width=<positive int>] "
                 "[--greedy-search-depth=<non-negative int>] "
                 "[--greedy-alpha=<non-negative float>]\n";
    std::cerr << "  solver: greedy | base | heuristic | brute_force\n";
    std::cerr << "  greedy flags only apply to the greedy solver\n";
}

auto ToLower(std::string text) -> std::string {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

auto ParseSolverKind(const std::string& solver_choice, SolverKind& kind) -> bool {
    std::string const normalized = ToLower(solver_choice);

    if (normalized == "greedy") {
        kind = SolverKind::kGreedy;
        return true;
    }
    if (normalized == "base") {
        kind = SolverKind::kBase;
        return true;
    }
    if (normalized == "heuristic") {
        kind = SolverKind::kHeuristic;
        return true;
    }
    if (normalized == "bruteforce" || normalized == "brute_force" || normalized == "brute-force") {
        kind = SolverKind::kBruteForce;
        return true;
    }

    return false;
}

auto SolverName(SolverKind kind) -> std::string {
    switch (kind) {
    case SolverKind::kGreedy:
        return "GreedySolver";
    case SolverKind::kBase:
        return "BaseSolver";
    case SolverKind::kHeuristic:
        return "HeuristicSolver";
    case SolverKind::kBruteForce:
        return "BruteForceSolver";
    }
    return "UnknownSolver";
}

auto BuildGreedyConfig(const RunSolverOptions& options) -> GreedyFuserConfig {
    return GreedyFuserConfig{
        .search_depth = options.greedy_search_depth.value_or(kDefaultGreedySearchDepth),
        .beam_width = options.greedy_beam_width.value_or(kDefaultGreedyBeamWidth),
        .topk_failure_penalty = options.greedy_alpha.value_or(kDefaultGreedyAlpha),
    };
}

auto HasGreedyOverrides(const RunSolverOptions& options) -> bool {
    return options.greedy_search_depth.has_value() || options.greedy_beam_width.has_value() ||
           options.greedy_alpha.has_value();
}

auto BuildSolver(SolverKind kind, const RunSolverOptions& options) -> std::unique_ptr<Solver> {
    switch (kind) {
    case SolverKind::kGreedy:
        if (HasGreedyOverrides(options)) {
            return std::make_unique<GreedySolver>(BuildGreedyConfig(options), options.output_path);
        }
        return std::make_unique<GreedySolver>(options.output_path);
    case SolverKind::kBase:
        return std::make_unique<BaseSolver>();
    case SolverKind::kHeuristic:
        return std::make_unique<HeuristicSolver>(options.output_path);
    case SolverKind::kBruteForce:
        return std::make_unique<BruteForceSolver>();
    }
    return nullptr;
}

auto ParseIntFlag(const std::string& flag_name, const std::string& value, int minimum,
                  const std::string& expectation, int& parsed) -> bool {
    try {
        size_t parsed_chars = 0;
        int const number = std::stoi(value, &parsed_chars);
        if (parsed_chars != value.size() || number < minimum) {
            std::cerr << "Invalid " << flag_name << " value: " << value << " (expected "
                      << expectation << ")\n";
            return false;
        }
        parsed = number;
        return true;
    } catch (const std::invalid_argument&) {
        std::cerr << "Invalid " << flag_name << " value: " << value << " (expected " << expectation
                  << ")\n";
        return false;
    } catch (const std::out_of_range&) {
        std::cerr << "Invalid " << flag_name << " value: " << value << " (out of range)\n";
        return false;
    }
}

auto ParsePositiveIntFlag(const std::string& flag_name, const std::string& value, int& parsed)
    -> bool {
    return ParseIntFlag(flag_name, value, 1, "positive integer", parsed);
}

auto ParseNonNegativeIntFlag(const std::string& flag_name, const std::string& value, int& parsed)
    -> bool {
    return ParseIntFlag(flag_name, value, 0, "non-negative integer", parsed);
}

auto ParseNonNegativeDoubleFlag(const std::string& flag_name, const std::string& value,
                                double& parsed) -> bool {
    try {
        size_t parsed_chars = 0;
        double const number = std::stod(value, &parsed_chars);
        if (parsed_chars != value.size() || !std::isfinite(number) || number < 0.0) {
            std::cerr << "Invalid " << flag_name << " value: " << value
                      << " (expected non-negative float)\n";
            return false;
        }
        parsed = number;
        return true;
    } catch (const std::invalid_argument&) {
        std::cerr << "Invalid " << flag_name << " value: " << value
                  << " (expected non-negative float)\n";
        return false;
    } catch (const std::out_of_range&) {
        std::cerr << "Invalid " << flag_name << " value: " << value << " (out of range)\n";
        return false;
    }
}

auto BenchmarkNameFromInputPath(const std::string& input_path) -> std::string {
    std::filesystem::path path(input_path);
    std::string name = path.stem().string();
    if (name.empty()) {
        name = "unknown_benchmark";
    }
    return name;
}

auto ParseArgs(int argc, char* argv[], RunSolverOptions& options) -> bool {
    if (argc < 4) {
        return false;
    }

    options.solver_choice = argv[1];
    options.input_path = argv[2];
    options.output_path = argv[3];

    for (int i = 4; i < argc; ++i) {
        std::string const arg = argv[i];

        if (arg.starts_with(kFuserLogDirOption)) {
            options.fuser_log_dir = arg.substr(kFuserLogDirOption.size());
            if (options.fuser_log_dir.empty()) {
                std::cerr << "Invalid --fuser-log-dir value: empty\n";
                return false;
            }
            continue;
        }

        if (arg.starts_with(kGreedyBeamWidthOption)) {
            int parsed = 0;
            if (!ParsePositiveIntFlag("--greedy-beam-width",
                                      arg.substr(kGreedyBeamWidthOption.size()), parsed)) {
                return false;
            }
            options.greedy_beam_width = parsed;
            continue;
        }

        if (arg.starts_with(kGreedySearchDepthOption)) {
            int parsed = 0;
            if (!ParseNonNegativeIntFlag("--greedy-search-depth",
                                         arg.substr(kGreedySearchDepthOption.size()), parsed)) {
                return false;
            }
            options.greedy_search_depth = parsed;
            continue;
        }

        if (arg.starts_with(kGreedyAlphaOption)) {
            double parsed = 0.0;
            if (!ParseNonNegativeDoubleFlag("--greedy-alpha", arg.substr(kGreedyAlphaOption.size()),
                                            parsed)) {
                return false;
            }
            options.greedy_alpha = parsed;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return false;
    }

    return true;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    RunSolverOptions options;
    if (ParseArgs(argc, argv, options) == false) {
        PrintUsage();
        return 1;
    }

    SolverKind solver_kind;
    if (ParseSolverKind(options.solver_choice, solver_kind) == false) {
        std::cerr << "Unknown solver: " << options.solver_choice << "\n";
        PrintUsage();
        return 1;
    }
    if (solver_kind != SolverKind::kGreedy && HasGreedyOverrides(options)) {
        std::cerr << "Greedy hyperparameter flags can only be used with the greedy solver\n";
        return 1;
    }

    std::string const solver_name = SolverName(solver_kind);
    bool enable_fuser_logging = (solver_kind == SolverKind::kGreedy);

#if !MLSYS_ENABLE_FUSER_LOGGING
    if (enable_fuser_logging) {
        std::cerr << "Warning: fuser logging is compiled out. Rebuild with "
                     "-DMLSYS_ENABLE_FUSER_LOGGING=ON to enable fuser logs.\n";
        enable_fuser_logging = false;
    }
#endif

    FuserLoggingConfig logging_config;
    logging_config.enable_logging = enable_fuser_logging;
    logging_config.log_directory = options.fuser_log_dir;
    logging_config.benchmark_name = BenchmarkNameFromInputPath(options.input_path);
    logging_config.solver_name = solver_name;
    ConfigureFuserLogging(logging_config);

    std::cout << "Starting run_solver...\n";
    std::cout << "Solver: " << solver_name << "\n";
    std::cout << "Input file: " << options.input_path << "\n";
    std::cout << "Output file: " << options.output_path << "\n";
    if (solver_kind == SolverKind::kGreedy && HasGreedyOverrides(options)) {
        GreedyFuserConfig const config = BuildGreedyConfig(options);
        std::cout << "Greedy search depth: " << config.search_depth << "\n";
        std::cout << "Greedy beam width: " << config.beam_width << "\n";
        std::cout << "Greedy alpha: " << config.topk_failure_penalty << "\n";
    }

    if (logging_config.enable_logging) {
        std::string const log_path = GetFuserLogPath();
        if (log_path.empty() == false) {
            std::cout << "Fuser beam logging enabled: " << log_path << "\n";
        } else {
            std::cerr
                << "Warning: fuser beam logging requested but log file could not be opened.\n";
        }
    }

    auto problem_status = ReadProblem(options.input_path);
    if (problem_status.ok() == false) {
        std::cerr << "Error reading input: " << problem_status.status().message() << "\n";
        return 1;
    }
    Problem problem = problem_status.value();

    std::unique_ptr<Solver> solver = BuildSolver(solver_kind, options);
    if (solver == nullptr) {
        std::cerr << "Error: failed to build solver instance for " << solver_name << "\n";
        return 1;
    }

    auto solution_status = solver->solve(problem);
    if (solution_status.ok() == false) {
        std::cerr << "Error solving problem: " << solution_status.status().message() << "\n";
        return 1;
    }
    const Solution& solution = solution_status.value();

    auto eval_status = Evaluate(problem, solution);
    if (eval_status.ok()) {
        std::cout << "Evaluation successful.\n";
        std::cout << "Total Latency: " << eval_status.value() << "\n";
    } else {
        std::cerr << "Error evaluating solution: " << eval_status.status().message() << "\n";
    }

    auto write_status = WriteSolutionAtomically(solution, options.output_path);
    if (write_status.ok() == false) {
        std::cerr << "Error writing output: " << write_status.message() << "\n";
        return 1;
    }

    std::cout << "Done.\n";
    return 0;
}
