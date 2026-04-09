#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifndef MLSYS_ENABLE_FUSER_LOGGING
#define MLSYS_ENABLE_FUSER_LOGGING 0
#endif

namespace mlsys {

struct FuserLoggingConfig {
    bool enable_topk_candidate_logging = false;
    int top_k = 0;
    std::string log_directory = "logs";
    std::string benchmark_name = "unknown_benchmark";
    std::string solver_name = "GreedySolver";
};

struct FuserTopKCandidateLogData {
    std::string move_name;
    int64_t potential_score = 0;
    double penalized_score = 0.0;
    double op_hit_sum = 0.0;
    size_t producer_sg_idx = 0;
    size_t consumer_sg_idx = 0;
    std::vector<size_t> touched_tensors;
    std::vector<size_t> producer_subgraph_ops;
    std::vector<size_t> consumer_subgraph_ops;
};

void ConfigureFuserLogging(const FuserLoggingConfig& config);
auto GetFuserLogPath() -> std::string;
auto IsFuserTopKLoggingEnabled() -> bool;
auto GetFuserTopKLimit() -> size_t;
void LogFuserTopKCandidates(int depth, size_t total_candidates, size_t beam_width,
                            const std::vector<FuserTopKCandidateLogData>& candidates);

} // namespace mlsys
