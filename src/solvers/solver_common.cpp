/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "solver_common.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

namespace mlsys::solver_internal {

void MaybePauseAfterBaselineForTest() {
    const char* marker_path = std::getenv("MLSYS_TEST_PAUSE_AFTER_BASELINE");
    if (marker_path == nullptr || std::string(marker_path).empty()) {
        marker_path = std::getenv("MLSYS_GREEDY_TEST_PAUSE_AFTER_BASELINE");
    }
    if (marker_path == nullptr || std::string(marker_path).empty()) {
        return;
    }

    std::ofstream marker(marker_path);
    marker << "baseline_written\n";
    marker.close();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

} // namespace mlsys::solver_internal
