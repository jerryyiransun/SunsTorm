/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "solution_writer.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <system_error>
#include <utility>

namespace mlsys {

namespace {

constexpr auto kSolutionWriterDelay = std::chrono::milliseconds(250);

} // namespace

AnytimeSolutionWriter::AnytimeSolutionWriter(std::string output_path)
    : output_path_(std::move(output_path)) {}

AnytimeSolutionWriter::~AnytimeSolutionWriter() {
    (void)StopAndFlush();
}

auto AnytimeSolutionWriter::Start() -> absl::Status {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (writer_running_) {
        return absl::OkStatus();
    }

    stop_requested_ = false;
    pending_dirty_ = false;
    force_immediate_write_ = false;
    published_generation_ = 0;
    flushed_generation_ = 0;
    pending_best_cost_ = std::numeric_limits<TotalLatency>::infinity();
    last_write_status_ = absl::OkStatus();

    try {
        writer_running_ = true;
        writer_thread_ = std::thread(&AnytimeSolutionWriter::WriterMain, this);
    } catch (const std::system_error& error) {
        writer_running_ = false;
        return absl::InternalError(std::string("Failed to start solution writer thread: ") +
                                   error.what());
    }

    return absl::OkStatus();
}

auto AnytimeSolutionWriter::PublishIfBetter(const Solution& solution, TotalLatency cost)
    -> absl::Status {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!writer_running_) {
        return absl::FailedPreconditionError("Anytime solution writer is not running");
    }
    if (stop_requested_) {
        return absl::FailedPreconditionError("Anytime solution writer is stopping");
    }
    if (cost >= pending_best_cost_) {
        return absl::OkStatus();
    }

    pending_best_solution_ = solution;
    pending_best_cost_ = cost;
    pending_dirty_ = true;
    ++published_generation_;
    pending_cv_.notify_one();
    return absl::OkStatus();
}

auto AnytimeSolutionWriter::PublishAndWaitForWrite(const Solution& solution, TotalLatency cost)
    -> absl::Status {
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (!writer_running_) {
        return absl::FailedPreconditionError("Anytime solution writer is not running");
    }
    if (stop_requested_) {
        return absl::FailedPreconditionError("Anytime solution writer is stopping");
    }
    if (cost >= pending_best_cost_) {
        return absl::OkStatus();
    }

    pending_best_solution_ = solution;
    pending_best_cost_ = cost;
    pending_dirty_ = true;
    force_immediate_write_ = true;
    uint64_t const target_generation = ++published_generation_;
    pending_cv_.notify_one();

    flushed_cv_.wait(lock, [&] {
        return flushed_generation_ >= target_generation || !last_write_status_.ok() ||
               !writer_running_;
    });
    return last_write_status_;
}

auto AnytimeSolutionWriter::StopAndFlush() -> absl::Status {
    bool should_join = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (writer_running_) {
            stop_requested_ = true;
            force_immediate_write_ = true;
            should_join = writer_thread_.joinable();
            pending_cv_.notify_one();
        } else {
            should_join = writer_thread_.joinable();
        }
    }

    if (should_join) {
        writer_thread_.join();
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    writer_running_ = false;
    return last_write_status_;
}

void AnytimeSolutionWriter::WriterMain() {
    std::unique_lock<std::mutex> lock(state_mutex_);

    while (true) {
        pending_cv_.wait(lock, [&] { return pending_dirty_ || stop_requested_; });

        if (!pending_dirty_ && stop_requested_) {
            break;
        }

        if (pending_dirty_ && !force_immediate_write_ && !stop_requested_) {
            pending_cv_.wait_for(lock, kSolutionWriterDelay,
                                 [&] { return force_immediate_write_ || stop_requested_; });
        }

        if (!pending_dirty_) {
            if (stop_requested_) {
                break;
            }
            continue;
        }

        Solution solution_to_write = pending_best_solution_;
        uint64_t const generation_to_write = published_generation_;
        pending_dirty_ = false;
        force_immediate_write_ = false;

        lock.unlock();
        absl::Status const write_status = WriteSolutionAtomically(solution_to_write, output_path_);
        lock.lock();

        last_write_status_ = write_status;
        if (write_status.ok()) {
            flushed_generation_ = std::max(flushed_generation_, generation_to_write);
        }
        flushed_cv_.notify_all();

        if (stop_requested_ && !pending_dirty_) {
            break;
        }
    }

    writer_running_ = false;
    flushed_cv_.notify_all();
}

} // namespace mlsys
