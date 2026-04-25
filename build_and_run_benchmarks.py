#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_build_dir = script_dir / "build-release"
    default_benchmark_dir = script_dir / "benchmarks"
    default_runs_dir = default_build_dir / "runs"

    parser = argparse.ArgumentParser(
        description=(
            "Configure and build the Release CMake target, then run the mlsys "
            "executable for every benchmark JSON."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir,
        help=f"Build directory to create/use (default: {default_build_dir})",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=default_benchmark_dir,
        help=f"Directory containing benchmark JSON files (default: {default_benchmark_dir})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir,
        help=f"Directory for timestamped run outputs (default: {default_runs_dir})",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for this run's outputs (default: <runs-dir>/<timestamp>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated solution JSON files (default: <run-dir>/solutions)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Markdown report to write at the end (default: <run-dir>/report.md)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override timeout in seconds for every benchmark (default: benchmark-specific)",
    )
    parser.add_argument(
        "--solver",
        choices=["greedy", "base", "heuristic", "brute_force", "bruteforce", "brute-force"],
        default="greedy",
        help="Solver to run for each benchmark (default: greedy)",
    )
    parser.add_argument(
        "--fuser-log-dir",
        type=Path,
        default=None,
        help="Fuser log directory passed to run_solver (default: run_solver default)",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_command_with_timeout(command: list[str], cwd: Path, timeout_seconds: float) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout_seconds)


def run_command_capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def benchmark_sort_key(path: Path) -> tuple[int, str]:
    benchmark_number = benchmark_number_from_path(path)
    return benchmark_number if benchmark_number is not None else sys.maxsize, path.name


def benchmark_number_from_path(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return None
    return int(match.group(1))


def default_timeout_for_benchmark(path: Path) -> float:
    benchmark_number = benchmark_number_from_path(path)
    if benchmark_number is None:
        return 60.0
    if 1 <= benchmark_number <= 4:
        return 2.0
    if 5 <= benchmark_number <= 8:
        return 5.0
    if 9 <= benchmark_number <= 12:
        return 15.0
    if 13 <= benchmark_number <= 16:
        return 30.0
    if 17 <= benchmark_number <= 20:
        return 60.0
    if 21 <= benchmark_number <= 24:
        return 120.0
    return 60.0


def timeout_for_benchmark(path: Path, override_seconds: float | None) -> float:
    return override_seconds if override_seconds is not None else default_timeout_for_benchmark(path)


def discover_benchmarks(benchmark_dir: Path) -> list[Path]:
    benchmarks = sorted(
        (path for path in benchmark_dir.glob("*.json") if path.is_file()),
        key=benchmark_sort_key,
    )
    if not benchmarks:
        raise FileNotFoundError(f"No benchmark JSON files found in {benchmark_dir}")
    return benchmarks


def parse_total_latency(output: str) -> str:
    match = re.search(r"Total Latency:\s*(\S+)", output)
    if not match:
        raise ValueError("Could not find 'Total Latency:' in evaluator output")
    return match.group(1)


def evaluate_solution(evaluator: Path, benchmark: Path, output_path: Path, cwd: Path) -> str:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return "N/A"

    try:
        evaluation = run_command_capture(
            [str(evaluator), str(benchmark), str(output_path)],
            cwd=cwd,
        )
        return parse_total_latency(evaluation.stdout)
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"Could not evaluate {output_path}: {exc}", file=sys.stderr)
        return "N/A"


def normalize_solver_choice(solver: str) -> str:
    if solver in {"bruteforce", "brute-force"}:
        return "brute_force"
    return solver


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_report(report_path: Path, run_dir: Path, rows: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Solver: `{rows[0]['solver'] if rows else 'unknown'}`",
        "",
        "| Benchmark | Solver | Status | Timeout (s) | Total Latency | Execution Time (s) | Solution JSON |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['solver']} | {row['status']} | {row['timeout_seconds']} | {row['total_latency']} | {row['execution_time_seconds']} | {row['solution_path']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    build_dir = args.build_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else runs_dir / make_timestamp()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dir / "solutions"
    report_path = args.report_path.resolve() if args.report_path is not None else run_dir / "report.md"
    timeout_override_seconds = args.timeout_seconds
    solver = normalize_solver_choice(args.solver)

    if timeout_override_seconds is not None and timeout_override_seconds <= 0:
        print("--timeout-seconds must be greater than 0", file=sys.stderr)
        return 1

    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        return 1

    try:
        benchmarks = discover_benchmarks(benchmark_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Solver: {solver}", flush=True)

    try:
        run_command(
            [
                "cmake",
                "-S",
                str(repo_root),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=repo_root,
        )
        run_command(
            ["cmake", "--build", str(build_dir), "--target", "run_solver", "evaluate_debug"],
            cwd=repo_root,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Build failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    executable = build_dir / "run_solver"
    evaluator = build_dir / "evaluate_debug"
    if not executable.is_file():
        print(f"Expected executable was not built: {executable}", file=sys.stderr)
        return 1
    if not evaluator.is_file():
        print(f"Expected evaluator was not built: {evaluator}", file=sys.stderr)
        return 1

    report_rows: list[dict[str, str]] = []
    final_exit_code = 0
    for benchmark in benchmarks:
        output_name = f"{benchmark.stem}-solution.json"
        output_path = output_dir / output_name
        timeout_seconds = timeout_for_benchmark(benchmark, timeout_override_seconds)
        timeout_label = f"{timeout_seconds:g}"
        print(
            f"Running {benchmark.name} -> {output_path} (timeout={timeout_label}s)",
            flush=True,
        )
        start_time = time.perf_counter()
        try:
            solver_command = [str(executable), solver, str(benchmark), str(output_path)]
            if args.fuser_log_dir is not None:
                solver_command.append(f"--fuser-log-dir={args.fuser_log_dir.resolve()}")
            run_command_with_timeout(
                solver_command,
                cwd=repo_root,
                timeout_seconds=timeout_seconds,
            )
            execution_time_seconds = time.perf_counter() - start_time
            total_latency = evaluate_solution(evaluator, benchmark, output_path, repo_root)
            report_rows.append(
                {
                    "benchmark": benchmark.name,
                    "solver": solver,
                    "status": "completed",
                    "timeout_seconds": timeout_label,
                    "total_latency": total_latency,
                    "execution_time_seconds": f"{execution_time_seconds:.6f}",
                    "solution_path": str(output_path),
                }
            )
            print(
                (
                    f"Finished {benchmark.name} "
                    f"(execution_time={execution_time_seconds:.6f}s, total_latency={total_latency})"
                ),
                flush=True,
            )
        except subprocess.CalledProcessError as exc:
            execution_time_seconds = time.perf_counter() - start_time
            total_latency = evaluate_solution(evaluator, benchmark, output_path, repo_root)
            report_rows.append(
                {
                    "benchmark": benchmark.name,
                    "solver": solver,
                    "status": f"failed (exit {exc.returncode})",
                    "timeout_seconds": timeout_label,
                    "total_latency": total_latency,
                    "execution_time_seconds": f"{execution_time_seconds:.6f}",
                    "solution_path": str(output_path),
                }
            )
            print(
                f"Benchmark run failed for {benchmark.name} with exit code {exc.returncode}",
                file=sys.stderr,
            )
            final_exit_code = exc.returncode if final_exit_code == 0 else final_exit_code
            continue
        except subprocess.TimeoutExpired:
            execution_time_seconds = time.perf_counter() - start_time
            total_latency = evaluate_solution(evaluator, benchmark, output_path, repo_root)
            report_rows.append(
                {
                    "benchmark": benchmark.name,
                    "solver": solver,
                    "status": "timed out",
                    "timeout_seconds": timeout_label,
                    "total_latency": total_latency,
                    "execution_time_seconds": f"{execution_time_seconds:.6f}",
                    "solution_path": str(output_path),
                }
            )
            print(
                (
                    f"Benchmark run timed out for {benchmark.name} "
                    f"after {execution_time_seconds:.6f}s "
                    f"(timeout={timeout_label}s, total_latency={total_latency})"
                ),
                file=sys.stderr,
            )
            final_exit_code = 124 if final_exit_code == 0 else final_exit_code
            continue

    write_report(report_path, run_dir, report_rows)
    print(f"Wrote report to {report_path}", flush=True)
    print(f"Generated {len(benchmarks)} solution files in {output_dir}", flush=True)
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
