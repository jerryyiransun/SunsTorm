# Debug

## Build

```bash
# Debug build (with fuser logging enabled)
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DMLSYS_ENABLE_FUSER_LOGGING=ON
cmake --build build-debug
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
