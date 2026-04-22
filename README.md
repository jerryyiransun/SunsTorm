# MLSys 2026 Google Graph Scheduling Competition
![Build Status](https://github.com/jerryyiransun/MLSys2026-Google-Graph-Scheduling-Competition/actions/workflows/ci.yaml/badge.svg)

This repository contains our C++ scheduling algorithm for Track A of the MLSys 2026 competition.

## Prerequisites

- **CMake** (Version 3.14 or higher)
- **C++ Compiler** supporting C++17
- **clang-format** (for pre-commit formatting)

## How to Build

Open your terminal at the project root and run:

```bash
# Default build
cmake -B build
cmake --build build

# Debug build (with fuser logging enabled)
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DMLSYS_ENABLE_FUSER_LOGGING=ON
cmake --build build-debug

# Profiling build
cmake -B build-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-relwithdebinfo
```

## How to Run

`mlsys` follows the competition interface and accepts exactly two arguments:

```bash
./build/mlsys <path/to/input.json> <path/to/output.json>

# example
./build/mlsys benchmarks/mlsys-2026-1.json out.json
```

For profiling, prefer the `RelWithDebInfo` build. It keeps compiler optimizations
enabled while preserving debug symbols, which makes profiler output much more
readable without making the binary behave like a debug build.

```bash
./build-relwithdebinfo/mlsys <path/to/input.json> <path/to/output.json>

# example
./build-relwithdebinfo/mlsys benchmarks/mlsys-2026-1.json out.json
```

## Local Runner (`run_solver`)

Use `run_solver` for local experimentation when you want to choose a solver
from the command line.

```bash
./build-relwithdebinfo/run_solver <solver> <input.json> <output.json> [--fuser-log-dir=<dir>]

# solver options
# greedy | base | heuristic | brute_force
```

## Debug Evaluator (`evaluate_debug`)

Use `evaluate_debug` to run a custom problem JSON with debug evaluation output and
auto-write the generated solution JSON using the test name.

```bash
# default output name: Benchmark1_WithTestJson_DebugOutput.json
./build-debug/evaluate_debug <path/to/problem.json>

# optional custom test name -> writes <test_name>.json
./build-debug/evaluate_debug <path/to/problem.json> <test_name>
```

Examples:

```bash
./build-debug/evaluate_debug benchmarks/mlsys-2026-1.json
./build-debug/evaluate_debug benchmarks/mlsys-2026-17.json MyDebugRun
```

Notes:

- Build with `-DCMAKE_BUILD_TYPE=Debug` to see evaluator debug logs.
- Output JSON is written to the current working directory.
- Default output file is `Benchmark1_WithTestJson_DebugOutput.json`.

## Fuser Beam Logging (Build + Run)

To produce a fuser log file, you must enable logging at both compile time and runtime.

1. Compile time: build with `-DMLSYS_ENABLE_FUSER_LOGGING=ON`
2. Runtime: run `GreedySolver` (no logging enable flag required)

Build a logging-enabled binary:

```bash
cmake -S . -B build-relwithdebinfo-logging \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLSYS_ENABLE_FUSER_LOGGING=ON
cmake --build build-relwithdebinfo-logging -j
```

Run with logging enabled:

```bash
./build-relwithdebinfo-logging/run_solver greedy benchmarks/mlsys-2026-1.json out.json \
  --fuser-log-dir=logs
```

When logging is active, `run_solver` prints the exact output path:

```text
Fuser beam logging enabled: logs/<timestamp>_<benchmark>_<solver>_fuser_beam.log
```

Useful flags:

- `--fuser-log-dir=<dir>`: output directory for log files (default: `logs`)

Notes:

- `--fuser-log-top-k` has been removed.
- If logging is compiled out, `run_solver` prints a warning for `GreedySolver` runs.
- Logging is emitted automatically for `GreedySolver`.
- For other solvers, fuser logging is disabled.

Common log records include:

- `[SearchFrameState] ... subgraph_ops=[[...],[...],...]`
- `[BeamCandidates] ...`
- `[BeamCandidateResult] ...`
- `[EXPLORATION_SECTION_1_BASELINE] explore_id=... baseline_score=...`
- `[EXPLORATION_SECTION_2_EXPLORED] explore_id=... explored_score=...`
- `[EXPLORATION_SECTION_3_DELTA] explore_id=... delta_cost=... improved=...`
- `===SELECTED_SOLUTION=============================================`
- `[SELECTED_SOLUTION] ... selected_latency_cost=... selected_explore_id=...`
- `[SELECTED_SOLUTION_TOP_CANDIDATES] count=...` followed by up to 5 candidates
- `===SEARCH_FINAL_BEST=============================================` and `[SEARCH_FINAL_BEST] ... final_latency_cost=...`

Marker behavior:

- `FUSION_EXPLORATION_MARKER` has been removed.
- `SELECTED_SOLUTION` is emitted only when global best latency strictly improves.
- `SELECTED_SOLUTION` includes the selected latency and top evaluated candidates for that frame,
  sorted by evaluated latency ascending, each with `explore_id`.
- You can copy an `explore_id` from `SELECTED_SOLUTION` and search for matching
  `EXPLORATION_SECTION_1/2/3` entries.

Inspect the newest log:

```bash
ls -1t logs/*_fuser_beam.log | head -n 1
tail -n 100 logs/*_fuser_beam.log

```
## Docker Grading Smoke Test

Use the Docker scripts to build and run `mlsys` in an Ubuntu 22.04 `linux/amd64`
environment that matches the competition runtime target.

Build the final `./mlsys` binary:

```bash
./run_docker.sh
```

Run one benchmark under the advertised 8-core / 32 GB cap:

```bash
./run_docker_benchmark.sh <benchmark-number> <timeout-seconds> [output-json]

# examples
./run_docker_benchmark.sh 1 5
./run_docker_benchmark.sh 17 30 output-17.json
```

The benchmark number maps to `benchmarks/mlsys-2026-<benchmark-number>.json`.
If no output path is provided, the script writes `output.json`.

Notes:

- `run_docker.sh` builds with `-march=x86-64-v3` inside Ubuntu 22.04 and writes
  the stripped executable to `./mlsys`.
- `run_docker_benchmark.sh` runs `./mlsys` in a clean Ubuntu 22.04 container with
  `--cpus=8 --memory=32g` and fails if the output JSON is empty.
- On Apple Silicon, Docker uses `linux/amd64` emulation, so this is useful for
  compatibility smoke testing but not reliable for final timing.

## Profiling with `perf`

Install `perf` on Ubuntu:

```bash
sudo apt update
sudo apt install linux-tools-common linux-tools-generic
```

Record a profile for the optimized, symbolized binary:

```bash
perf record -g -- ./build-relwithdebinfo/mlsys benchmarks/mlsys-2026-1.json out.json
```

Inspect the resulting profile:

```bash
perf report
```

### `perf` on WSL2

On WSL2, the usual `linux-tools-<kernel>` package often does not exist for the
Microsoft kernel, even when the `perf` launcher suggests it. In that case, build
`perf` from the WSL kernel source and use that binary directly.

Install the build dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  flex \
  bison \
  pkg-config \
  python3-dev \
  libssl-dev \
  libelf-dev \
  libtraceevent-dev \
  libdw-dev \
  libdebuginfod-dev \
  libunwind-dev \
  libslang2-dev \
  libperl-dev \
  liblzma-dev \
  libzstd-dev \
  libcap-dev \
  libnuma-dev \
  libbabeltrace-dev \
  libpfm4-dev \
  systemtap-sdt-dev
```

Clone the WSL kernel source and build `perf`:

```bash
mkdir -p ~/tools
git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git ~/tools/WSL2-Linux-Kernel
cd ~/tools/WSL2-Linux-Kernel/tools/perf
make
```

Confirm the binary exists:

```bash
~/tools/WSL2-Linux-Kernel/tools/perf/perf --version
```

Add it to your shell `PATH`:

```bash
echo 'export PATH="$HOME/tools/WSL2-Linux-Kernel/tools/perf:$PATH"' >> ~/.bashrc
source ~/.bashrc
which perf
perf --version
```

Then profile `mlsys` as usual:

```bash
perf record -g -- ./build-relwithdebinfo/mlsys benchmarks/mlsys-2026-1.json out.json
perf report
```

WSL2 notes:

- Some hardware counters may still be unavailable under WSL2.
- `perf record -g` stack sampling is still the recommended first step for
  identifying hotspots in `mlsys`.
- If `make` fails because a library is missing, install the package named in the
  error and rerun `make`.

## Git Hooks (clang-format pre-commit)

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
