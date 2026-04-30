/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "absl/status/status.h"
#include "mlsys.h"

#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

namespace mlsys {

class AnytimeSolutionWriter {
  public:
    explicit AnytimeSolutionWriter(std::string output_path);
    ~AnytimeSolutionWriter();

    AnytimeSolutionWriter(const AnytimeSolutionWriter&) = delete;
    auto operator=(const AnytimeSolutionWriter&) -> AnytimeSolutionWriter& = delete;

    auto Start() -> absl::Status;
    auto PublishIfBetter(const Solution& solution, TotalLatency cost) -> absl::Status;
    auto PublishAndWaitForWrite(const Solution& solution, TotalLatency cost) -> absl::Status;
    auto StopAndFlush() -> absl::Status;

  private:
    void WriterMain();

    std::string output_path_;
    std::thread writer_thread_;
    std::mutex state_mutex_;
    std::condition_variable pending_cv_;
    std::condition_variable flushed_cv_;
    bool writer_running_ = false;
    bool stop_requested_ = false;
    bool pending_dirty_ = false;
    bool force_immediate_write_ = false;
    uint64_t published_generation_ = 0;
    uint64_t flushed_generation_ = 0;
    TotalLatency pending_best_cost_ = std::numeric_limits<TotalLatency>::infinity();
    Solution pending_best_solution_;
    absl::Status last_write_status_ = absl::OkStatus();
};

} // namespace mlsys
