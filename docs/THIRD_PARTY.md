# Third-Party Components

The repository does not vendor third-party source trees. CMake downloads these
dependencies with `FetchContent` during configuration:

| Component | Version | Use | Upstream license |
| --- | --- | --- | --- |
| [Abseil C++](https://github.com/abseil/abseil-cpp) | `20260107.1` | status/statusor and utility libraries | Apache-2.0 |
| [nlohmann/json](https://github.com/nlohmann/json) | `v3.12.0` | JSON parsing and serialization | MIT |
| [GoogleTest](https://github.com/google/googletest) | `v1.17.0` | test binaries | BSD-3-Clause |

When redistributing source bundles or binaries that include fetched dependency
code, keep the corresponding upstream license files and notices with the
redistribution.
