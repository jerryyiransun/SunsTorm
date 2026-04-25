# Benchmark Runner

`scripts/build_and_run_benchmarks.py` is a convenience script for building the local
benchmark tooling in Release mode, running one solver across every benchmark JSON,
evaluating each generated solution, and writing a Markdown summary report.

Use it from the repository root:

```bash
python3 scripts/build_and_run_benchmarks.py
```

The default command:

- configures CMake with `-DCMAKE_BUILD_TYPE=Release`
- builds the `run_solver` and `evaluate_debug` targets
- reads benchmark inputs from `benchmarks/*.json`
- runs the `greedy` solver on each benchmark
- writes solution JSON files under `build-release/runs/<timestamp>/solutions`
- writes a report to `build-release/runs/<timestamp>/report.md`

## Requirements

Before running the script, make sure the normal project build requirements are
available:

- CMake 3.14 or newer
- a C++ compiler supported by this project
- Python 3
- the benchmark JSON files in `benchmarks/`

The script invokes CMake itself, so you do not need to manually create
`build-release` first.

## Common Commands

Run all benchmarks with the default greedy solver:

```bash
python3 scripts/build_and_run_benchmarks.py
```

Run all benchmarks with another solver:

```bash
python3 scripts/build_and_run_benchmarks.py --solver heuristic
python3 scripts/build_and_run_benchmarks.py --solver base
python3 scripts/build_and_run_benchmarks.py --solver brute_force
```

Use a single timeout for every benchmark:

```bash
python3 scripts/build_and_run_benchmarks.py --timeout-seconds 30
```

Write this run into a fixed directory:

```bash
python3 scripts/build_and_run_benchmarks.py --run-dir build-release/runs/manual-check
```

Use a different benchmark directory:

```bash
python3 scripts/build_and_run_benchmarks.py --benchmark-dir path/to/benchmarks
```

Enable a custom fuser log directory for `run_solver`:

```bash
python3 scripts/build_and_run_benchmarks.py --fuser-log-dir logs
```

`--fuser-log-dir` is passed through to `run_solver`. Logging still depends on
the binary being built with the relevant compile-time logging option.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--build-dir` | `build-release` | CMake build directory to create or reuse. |
| `--benchmark-dir` | `benchmarks` | Directory containing benchmark `.json` files. |
| `--runs-dir` | `build-release/runs` | Parent directory for timestamped run directories. |
| `--run-dir` | `<runs-dir>/<timestamp>` | Output directory for this script invocation. |
| `--output-dir` | `<run-dir>/solutions` | Directory for generated solution JSON files. |
| `--report-path` | `<run-dir>/report.md` | Markdown report path. |
| `--timeout-seconds` | benchmark-specific | Override timeout for every benchmark. |
| `--solver` | `greedy` | Solver passed to `run_solver`. |
| `--fuser-log-dir` | `run_solver` default | Optional fuser log directory passed through to `run_solver`. |

Accepted solver names are:

- `greedy`
- `base`
- `heuristic`
- `brute_force`
- `bruteforce`
- `brute-force`

The `bruteforce` and `brute-force` aliases are normalized to `brute_force`
before invoking `run_solver`.

## Timeout Defaults

When `--timeout-seconds` is not provided, the script chooses a timeout from the
benchmark number at the end of the JSON filename stem:

| Benchmark numbers | Timeout |
| --- | ---: |
| 1-4 | 2 seconds |
| 5-8 | 5 seconds |
| 9-12 | 15 seconds |
| 13-16 | 30 seconds |
| 17-20 | 60 seconds |
| 21-24 | 120 seconds |
| no recognized number | 60 seconds |

For example, `mlsys-2026-17.json` uses the 60 second default.

## Outputs

Each benchmark writes one solution file named:

```text
<benchmark-stem>-solution.json
```

For example:

```text
build-release/runs/20260424-153012/solutions/mlsys-2026-1-solution.json
```

The report contains one row per benchmark with:

- benchmark filename
- solver
- status
- timeout
- total latency from `evaluate_debug`
- wall-clock execution time
- solution JSON path

Possible status values include:

- `completed`
- `failed (exit <code>)`
- `timed out`

If a solution file is missing, empty, or cannot be evaluated, the report records
`N/A` for total latency.

## Exit Codes

The script continues running remaining benchmarks after an individual benchmark
fails or times out.

At the end:

- returns `0` if every benchmark completed successfully
- returns the first nonzero benchmark exit code if a benchmark process failed
- returns `124` if the first failure was a timeout
- returns `1` for setup errors such as a missing benchmark directory, no JSON
  files, invalid timeout, or missing expected build artifact

## Troubleshooting

If CMake configuration or build fails, rerun the printed CMake command manually
to inspect the full error.

If the script says no benchmark JSON files were found, check the value passed to
`--benchmark-dir`; only files ending in `.json` are discovered.

If the report shows `N/A` latency, inspect the corresponding solution JSON and
run the evaluator manually:

```bash
./build-release/evaluate_debug benchmarks/mlsys-2026-1.json \
  build-release/runs/<timestamp>/solutions/mlsys-2026-1-solution.json
```
