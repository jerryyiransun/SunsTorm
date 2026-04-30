/*
 * Copyright 2026 Yiran (Jerry) Sun, Zigang (Richard) Sun and contributors
 *
 * Licensed under the Apache License, Version 2.0.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 */

#include "fuser.h"

#include "fuser_common.h"

namespace mlsys {

auto BruteForceFuser::fuse(const Problem& problem) -> StatusOr<std::vector<Solution>> {
    return fuser_internal::RunBruteForceFuser(problem);
}

} // namespace mlsys
