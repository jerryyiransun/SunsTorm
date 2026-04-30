/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "fuser_logging.h"
#include "mlsys.h"
#include "solver.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using namespace mlsys;

namespace {

struct RunSolverOptions {
    std::string solver_choice;
    std::string input_path;
    std::string output_path;
    std::string fuser_log_dir = "logs";
};

enum class SolverKind {
    kGreedy,
    kBase,
    kHeuristic,
    kBruteForce,
};

void PrintUsage() {
    std::cerr << "Usage: ./run_solver <solver> <path_to_input.json> <path_to_output.json> "
                 "[--fuser-log-dir=<dir>]\n";
    std::cerr << "  solver: greedy | base | heuristic | brute_force\n";
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

auto BuildSolver(SolverKind kind, const std::string& output_path) -> std::unique_ptr<Solver> {
    switch (kind) {
    case SolverKind::kGreedy:
        return std::make_unique<GreedySolver>(output_path);
    case SolverKind::kBase:
        return std::make_unique<BaseSolver>();
    case SolverKind::kHeuristic:
        return std::make_unique<HeuristicSolver>(output_path);
    case SolverKind::kBruteForce:
        return std::make_unique<BruteForceSolver>();
    }
    return nullptr;
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

        if (arg.rfind("--fuser-log-dir=", 0) == 0) {
            options.fuser_log_dir = arg.substr(std::string("--fuser-log-dir=").size());
            if (options.fuser_log_dir.empty()) {
                std::cerr << "Invalid --fuser-log-dir value: empty\n";
                return false;
            }
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

    std::unique_ptr<Solver> solver = BuildSolver(solver_kind, options.output_path);
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
