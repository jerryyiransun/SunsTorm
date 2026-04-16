#include "fuser_logging.h"

#include <algorithm>

#if MLSYS_ENABLE_FUSER_LOGGING
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#endif

namespace mlsys {

namespace {

struct FuserLogState {
    bool enabled = false;
    std::string log_path;
#if MLSYS_ENABLE_FUSER_LOGGING
    std::ofstream stream;
#endif
};

FuserLogState g_fuser_log_state;

#if MLSYS_ENABLE_FUSER_LOGGING
auto SanitizeLogToken(const std::string& token) -> std::string {
    std::string out;
    out.reserve(token.size());
    for (char c : token) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        return "unknown";
    }
    return out;
}

auto BuildRunTimestamp() -> std::string {
    auto const now = std::chrono::system_clock::now();
    std::time_t const now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream out;
    out << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return out.str();
}

auto FormatIndexList(const std::vector<size_t>& values) -> std::string {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

void WriteFuserLogLine(const std::string& line) {
    if (!g_fuser_log_state.enabled || !g_fuser_log_state.stream.is_open()) {
        return;
    }
    g_fuser_log_state.stream << line << "\n";
    g_fuser_log_state.stream.flush();
}
#endif

} // namespace

void ConfigureFuserLogging(const FuserLoggingConfig& config) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (g_fuser_log_state.stream.is_open()) {
        g_fuser_log_state.stream.close();
    }

    g_fuser_log_state.enabled = false;
    g_fuser_log_state.log_path.clear();

    if (!config.enable_logging) {
        return;
    }

    std::filesystem::path const log_dir = config.log_directory.empty()
                                              ? std::filesystem::path("logs")
                                              : std::filesystem::path(config.log_directory);

    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    if (ec) {
        return;
    }

    std::string const benchmark = SanitizeLogToken(config.benchmark_name);
    std::string const solver = SanitizeLogToken(config.solver_name);
    std::string const timestamp = BuildRunTimestamp();
    std::filesystem::path const log_path =
        log_dir / (timestamp + "_" + benchmark + "_" + solver + "_fuser_beam.log");

    g_fuser_log_state.stream.open(log_path, std::ios::out | std::ios::trunc);
    if (g_fuser_log_state.stream.is_open() == false) {
        return;
    }

    g_fuser_log_state.enabled = true;
    g_fuser_log_state.log_path = log_path.string();

    std::ostringstream header;
    header << "[FuserLogStart] benchmark=" << benchmark << ", solver=" << solver
           << ", beam_logging_enabled=1";
    WriteFuserLogLine(header.str());
#else
    (void)config;
    g_fuser_log_state.enabled = false;
    g_fuser_log_state.log_path.clear();
#endif
}

auto GetFuserLogPath() -> std::string {
    return g_fuser_log_state.log_path;
}

auto IsFuserLoggingEnabled() -> bool {
#if MLSYS_ENABLE_FUSER_LOGGING
    return g_fuser_log_state.enabled;
#else
    return false;
#endif
}

void LogFuserTopKCandidates(int depth, size_t total_candidates, size_t beam_width,
                            const std::vector<FuserTopKCandidateLogData>& candidates) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }

    size_t const beam_explored = std::min(beam_width, candidates.size());
    size_t const top_k_logged = candidates.size();

    std::ostringstream header;
    header << "[BeamCandidates] depth=" << depth << ", total_ranked_candidates=" << total_candidates
           << ", beam_width=" << beam_width << ", beam_explored=" << beam_explored
           << ", top_k_logged=" << top_k_logged;
    WriteFuserLogLine(header.str());

    for (size_t idx = 0; idx < top_k_logged; ++idx) {
        FuserTopKCandidateLogData const& candidate = candidates[idx];

        std::ostringstream line;
        line << "rank=" << idx << ", move=" << candidate.move_name
             << ", potential_score=" << candidate.potential_score
             << ", penalized_score=" << candidate.penalized_score
             << ", op_hit_sum=" << candidate.op_hit_sum << ", pair=(" << candidate.producer_sg_idx
             << "->" << candidate.consumer_sg_idx << ")"
             << ", touched_tensors=" << FormatIndexList(candidate.touched_tensors);
        WriteFuserLogLine(line.str());

        std::ostringstream pair_ops_line;
        pair_ops_line << "rank=" << idx << ", pair_subgraph_ops=["
                      << FormatIndexList(candidate.producer_subgraph_ops) << ","
                      << FormatIndexList(candidate.consumer_subgraph_ops) << "]";
        WriteFuserLogLine(pair_ops_line.str());
    }
#else
    (void)depth;
    (void)total_candidates;
    (void)beam_width;
    (void)candidates;
#endif
}

void LogFuserDebugLine(const std::string& line) {
#if MLSYS_ENABLE_FUSER_LOGGING
    if (!IsFuserLoggingEnabled()) {
        return;
    }
    WriteFuserLogLine(line);
#else
    (void)line;
#endif
}

} // namespace mlsys
