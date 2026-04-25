# MLSys 2026 Google Graph Scheduling Competition

![Build Status](https://github.com/jerryyiransun/MLSys2026-Google-Graph-Scheduling-Competition/actions/workflows/ci.yaml/badge.svg)

This repository contains our algorithm for Track A of the [MLSys 2026 Google Graph Scheduling Competition](https://github.com/yarongmu-google/MLSys).

## Prerequisites

- **CMake** (Version 3.14 or higher)
- **C++ Compiler** supporting C++17
- **clang-format** (optional for formatting)
- **Docker**

## How to Build

Open your terminal at the project root and run:

```bash
cmake -B build <-DCMAKE_BUILD_TYPE=(Debug|RelWithDebInfo|Release)>
cmake --build build
```

## How to Run

`mlsys` follows the competition interface and accepts exactly two arguments:

```bash
./build/mlsys <path/to/input.json> <path/to/output.json>

# example
./build/mlsys benchmarks/mlsys-2026-1.json out.json
```

## Local Runner (`run_solver`)

Use `run_solver` for local experimentation when you want to choose a solver from the command line, this is meant for internal testing.

```bash
./build/run_solver <greedy|base|heuristic|brute_force> <input.json> <output.json> [--fuser-log-dir=<dir>]
```

## Benchmark Runner

Use `scripts/build_and_run_benchmarks.py` to build the Release benchmark tools, run a
solver across every benchmark JSON, and write timestamped solution files plus a
Markdown report.

See [BENCHMARK_RUNNER.md](docs/BENCHMARK_RUNNER.md).

## Docker Grading Smoke Test

Use the Docker scripts to build and run `mlsys` in an Ubuntu 22.04 `linux/amd64` environment that matches the competition runtime target.

Build the final `./mlsys` binary:

```bash
./scripts/run_docker.sh
```

Run one benchmark under the advertised 8-core / 32 GB cap:

```bash
./scripts/run_docker_benchmark.sh <benchmark-number> <timeout-seconds> [output-json]

# examples
./scripts/run_docker_benchmark.sh 1 5
./scripts/run_docker_benchmark.sh 17 30 output-17.json
```

The benchmark number maps to `benchmarks/mlsys-2026-<benchmark-number>.json`.
If no output path is provided, the script writes `output.json`.

Notes:

- `scripts/run_docker.sh` builds with `-march=x86-64-v3` inside Ubuntu 22.04 and writes
  the stripped executable to `./mlsys`.
- `scripts/run_docker_benchmark.sh` runs `./mlsys` in a clean Ubuntu 22.04 container with
  `--cpus=8 --memory=32g` and fails if the output JSON is empty.
- On Apple Silicon, Docker uses `linux/amd64` emulation, so this is useful for
  compatibility smoke testing but not reliable for final timing.

## Our Algorithm

See [ALGORITHM.md](docs/ALGORITHM.md)

## Debug

See [DEBUG.md](docs/DEBUG.md)

## Profiling

See [PROFILING.md](docs/PROFILING.md)

## Git Hooks: clang-format pre-commit (optional)

This repository includes a versioned pre-commit hook at `.githooks/pre-commit`
that auto-formats staged `*.cpp`, `*.h`, and `*.hpp` files using `.clang-format`.

Run this once per clone from the project root:

```bash
git config core.hooksPath .githooks
```

Optional verification:

```bash
ls .githooks
```

Then make a small commit. The hook will format staged C/C++ files and re-stage
them automatically before commit.
