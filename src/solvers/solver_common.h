/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "mlsys.h"

#include "absl/status/statusor.h"

namespace mlsys::solver_internal {

void MaybePauseAfterBaselineForTest();

} // namespace mlsys::solver_internal
