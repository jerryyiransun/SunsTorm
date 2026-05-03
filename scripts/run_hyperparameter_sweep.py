#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from xml.sax.saxutils import escape


DOCKER_IMAGE = "ubuntu:22.04"
CONTAINER_ROOT = PurePosixPath("/work")
DOCKER_CPUS = "8"
DOCKER_MEMORY = "32g"

DEFAULT_BEAM_WIDTH = 32
DEFAULT_LOOKAHEAD_DEPTH = 0
DEFAULT_ALPHA = 0.04

BEAM_VALUES = [4, 8, 16, 32]
LOOKAHEAD_DEPTH_VALUES = [0, 1, 2, 4]
ALPHA_VALUES = [0, 0.04, 0.08, 0.16]

RESULT_FIELDNAMES = [
    "benchmark",
    "benchmark_number",
    "timeout_seconds",
    "beam",
    "lookahead_depth",
    "alpha",
    "status",
    "exit_code",
    "execution_time_seconds",
    "total_latency",
    "solution_path",
    "stdout_log",
    "stderr_log",
    "command",
]


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_build_dir = repo_root / "build-ubuntu22-sweep"
    default_benchmark_dir = repo_root / "benchmarks"
    default_sweep_root = default_build_dir / "hyperparameter-sweeps" / make_timestamp()

    parser = argparse.ArgumentParser(
        description=(
            "Run the greedy solver hyperparameter grid inside the Docker benchmark envelope "
            "and write CSV/XLSX results."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir,
        help=f"Docker-built CMake directory (default: {default_build_dir})",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=default_benchmark_dir,
        help=f"Directory containing mlsys-2026 benchmark JSON files (default: {default_benchmark_dir})",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=default_sweep_root,
        help=f"Directory for sweep outputs (default: {default_sweep_root})",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first non-timeout solver failure.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse an existing Docker build directory instead of rebuilding run_solver/evaluate_debug.",
    )
    return parser.parse_args()


def format_value(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def benchmark_number_from_path(path: Path) -> int | None:
    match = re.search(r"^mlsys-2026-(\d+)$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def benchmark_sort_key(path: Path) -> tuple[int, str]:
    number = benchmark_number_from_path(path)
    return number if number is not None else sys.maxsize, path.name


def default_timeout_for_benchmark(path: Path) -> float:
    number = benchmark_number_from_path(path)
    if number is None:
        return 60.0
    if 1 <= number <= 4:
        return 2.0
    if 5 <= number <= 8:
        return 5.0
    if 9 <= number <= 12:
        return 15.0
    if 13 <= number <= 16:
        return 30.0
    if number == 17:
        return 60.0
    return 60.0


def discover_benchmarks(benchmark_dir: Path) -> list[Path]:
    benchmarks = sorted(
        (path for path in benchmark_dir.glob("mlsys-2026-*.json") if path.is_file()),
        key=benchmark_sort_key,
    )
    if not benchmarks:
        raise FileNotFoundError(f"No mlsys-2026 benchmark JSON files found in {benchmark_dir}")
    return benchmarks


def ensure_under_repo(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository for Docker mount access: {path}") from exc
    return resolved


def container_path(host_path: Path, repo_root: Path) -> PurePosixPath:
    relative = host_path.resolve().relative_to(repo_root)
    return CONTAINER_ROOT / PurePosixPath(relative.as_posix())


def docker_base_command(repo_root: Path) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--platform=linux/amd64",
        f"--cpus={DOCKER_CPUS}",
        f"--memory={DOCKER_MEMORY}",
        "-v",
        f"{repo_root}:/work",
        "-w",
        "/work",
        DOCKER_IMAGE,
    ]


def run_build(repo_root: Path, build_dir: Path) -> None:
    build_container = container_path(build_dir, repo_root)
    script = f"""
set -e
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \\
  ca-certificates git build-essential g++-11 cmake ninja-build file

cmake -S /work -B {shlex.quote(str(build_container))} -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_CXX_COMPILER=g++-11 \\
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \\
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"

cmake --build {shlex.quote(str(build_container))} --target run_solver evaluate_debug -j8
test -x {shlex.quote(str(build_container / "run_solver"))}
test -x {shlex.quote(str(build_container / "evaluate_debug"))}
file {shlex.quote(str(build_container / "run_solver"))}
"""
    command = docker_base_command(repo_root) + ["bash", "-lc", script]
    print("+ " + command_to_string(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def config_id(beam: int, lookahead_depth: int, alpha: float) -> str:
    return (
        f"beam_{format_value(beam)}__"
        f"depth_{format_value(lookahead_depth)}__"
        f"alpha_{format_value(alpha).replace('.', 'p')}"
    )


def parse_execution_time(stdout: str) -> str:
    match = re.search(r"__SWEEP_EXECUTION_TIME_SECONDS=(\S+)", stdout)
    if match is None:
        return "N/A"
    return match.group(1)


def run_solver_once(
    repo_root: Path,
    build_dir: Path,
    benchmark: Path,
    sweep_root: Path,
    beam: int,
    lookahead_depth: int,
    alpha: float,
) -> dict[str, Any]:
    number = benchmark_number_from_path(benchmark)
    timeout_seconds = default_timeout_for_benchmark(benchmark)
    timeout_label = format_value(timeout_seconds)
    cfg_id = config_id(beam, lookahead_depth, alpha)

    solution_dir = sweep_root / "solutions" / cfg_id
    log_dir = sweep_root / "logs" / cfg_id
    solution_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    solution_path = solution_dir / f"{benchmark.stem}-solution.json"
    stdout_log = log_dir / f"{benchmark.stem}.stdout.log"
    stderr_log = log_dir / f"{benchmark.stem}.stderr.log"

    executable = container_path(build_dir / "run_solver", repo_root)
    benchmark_container = container_path(benchmark, repo_root)
    solution_container = container_path(solution_path, repo_root)

    solver_command = [
        str(executable),
        "greedy",
        str(benchmark_container),
        str(solution_container),
        f"--greedy-beam-width={beam}",
        f"--greedy-search-depth={lookahead_depth}",
        f"--greedy-alpha={format_value(alpha)}",
    ]
    quoted_solver_command = " ".join(shlex.quote(part) for part in solver_command)
    script = f"""
set +e
start="$(date +%s.%N)"
timeout {shlex.quote(timeout_label)}s {quoted_solver_command}
rc=$?
end="$(date +%s.%N)"
elapsed="$(awk -v start="$start" -v end="$end" 'BEGIN {{ printf "%.6f", end - start }}')"
echo "__SWEEP_EXECUTION_TIME_SECONDS=${{elapsed}}"
exit "$rc"
"""

    docker_command = docker_base_command(repo_root) + ["bash", "-lc", script]
    print(
        (
            f"Running {benchmark.name} beam={beam} depth={lookahead_depth} "
            f"alpha={format_value(alpha)} timeout={timeout_label}s"
        ),
        flush=True,
    )
    completed = subprocess.run(docker_command, cwd=repo_root, capture_output=True, text=True)
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode == 0:
        status = "completed"
    elif completed.returncode == 124:
        status = "timed_out"
    else:
        status = "failed"

    execution_time_seconds = parse_execution_time(completed.stdout)
    total_latency = evaluate_solution(repo_root, build_dir, benchmark, solution_path)

    return {
        "benchmark": benchmark.name,
        "benchmark_number": number if number is not None else "",
        "timeout_seconds": timeout_label,
        "beam": beam,
        "lookahead_depth": lookahead_depth,
        "alpha": format_value(alpha),
        "status": status,
        "exit_code": completed.returncode,
        "execution_time_seconds": execution_time_seconds,
        "total_latency": total_latency,
        "solution_path": str(solution_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "command": command_to_string(docker_command),
    }


def parse_total_latency(output: str) -> str:
    match = re.search(r"Total Latency:\s*(\S+)", output)
    if match is None:
        return "N/A"
    return match.group(1)


def evaluate_solution(repo_root: Path, build_dir: Path, benchmark: Path, solution_path: Path) -> str:
    if not solution_path.is_file() or solution_path.stat().st_size == 0:
        return "N/A"

    evaluator = container_path(build_dir / "evaluate_debug", repo_root)
    benchmark_container = container_path(benchmark, repo_root)
    solution_container = container_path(solution_path, repo_root)
    command = docker_base_command(repo_root) + [
        str(evaluator),
        str(benchmark_container),
        str(solution_container),
    ]
    completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    if completed.returncode != 0:
        print(
            f"Could not evaluate {solution_path}: {completed.stderr.strip()}",
            file=sys.stderr,
            flush=True,
        )
        return "N/A"
    return parse_total_latency(completed.stdout)


def write_json_index(sweep_root: Path, benchmarks: list[Path], rows: list[dict[str, Any]]) -> None:
    payload = {
        "defaults": {
            "beam": DEFAULT_BEAM_WIDTH,
            "lookahead_depth": DEFAULT_LOOKAHEAD_DEPTH,
            "alpha": DEFAULT_ALPHA,
        },
        "grid": {
            "beam": BEAM_VALUES,
            "lookahead_depth": LOOKAHEAD_DEPTH_VALUES,
            "alpha": ALPHA_VALUES,
        },
        "benchmarks": [benchmark.name for benchmark in benchmarks],
        "runs": rows,
    }
    path = sweep_root / "hyperparameter_sweep_index.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def write_csv_results(sweep_root: Path, rows: list[dict[str, Any]]) -> None:
    path = sweep_root / "hyperparameter_sweep_results.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RESULT_FIELDNAMES})


def column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def numeric_cell_value(value: Any) -> str | None:
    if value == "" or value == "N/A" or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return format_value(value)
    try:
        number = float(str(value))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return str(value)


def worksheet_xml(rows: list[list[Any]], numeric_columns: set[int]) -> str:
    xml_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_name(col_idx)}{row_idx}"
            if row_idx > 1 and col_idx in numeric_columns:
                numeric_value = numeric_cell_value(value)
                if numeric_value is not None:
                    cells.append(f'<c r="{ref}"><v>{escape(numeric_value)}</v></c>')
                    continue
            text = "" if value is None else str(value)
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
            )
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    dimension = f"A1:{column_name(len(rows[0]))}{len(rows)}" if rows else "A1"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )


def write_xlsx_results(sweep_root: Path, rows: list[dict[str, Any]]) -> None:
    path = sweep_root / "hyperparameter_sweep_results.xlsx"
    sheet_rows: list[list[Any]] = [RESULT_FIELDNAMES]
    sheet_rows.extend([[row[field] for field in RESULT_FIELDNAMES] for row in rows])
    numeric_columns = {
        RESULT_FIELDNAMES.index(field) + 1
        for field in [
            "benchmark_number",
            "timeout_seconds",
            "beam",
            "lookahead_depth",
            "alpha",
            "exit_code",
            "execution_time_seconds",
            "total_latency",
        ]
    }

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sweep Results" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr("[Content_Types].xml", content_types)
        workbook_zip.writestr("_rels/.rels", root_rels)
        workbook_zip.writestr("xl/workbook.xml", workbook)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        workbook_zip.writestr("xl/styles.xml", styles)
        workbook_zip.writestr("xl/worksheets/sheet1.xml", worksheet_xml(sheet_rows, numeric_columns))


def write_outputs(sweep_root: Path, benchmarks: list[Path], rows: list[dict[str, Any]]) -> None:
    write_json_index(sweep_root, benchmarks, rows)
    write_csv_results(sweep_root, rows)
    write_xlsx_results(sweep_root, rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    args = parse_args()

    try:
        build_dir = ensure_under_repo(args.build_dir, repo_root, "--build-dir")
        benchmark_dir = ensure_under_repo(args.benchmark_dir, repo_root, "--benchmark-dir")
        sweep_root = ensure_under_repo(args.sweep_root, repo_root, "--sweep-root")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        return 1

    try:
        benchmarks = discover_benchmarks(benchmark_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sweep_root.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        try:
            run_build(repo_root, build_dir)
        except subprocess.CalledProcessError as exc:
            print(f"Docker build failed with exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode

    rows: list[dict[str, Any]] = []
    final_exit_code = 0
    for beam, lookahead_depth, alpha in itertools.product(
        BEAM_VALUES, LOOKAHEAD_DEPTH_VALUES, ALPHA_VALUES
    ):
        for benchmark in benchmarks:
            row = run_solver_once(
                repo_root,
                build_dir,
                benchmark,
                sweep_root,
                beam,
                lookahead_depth,
                alpha,
            )
            rows.append(row)
            write_outputs(sweep_root, benchmarks, rows)

            print(
                (
                    f"{row['benchmark']} beam={row['beam']} depth={row['lookahead_depth']} "
                    f"alpha={row['alpha']} -> {row['status']} "
                    f"latency={row['total_latency']} execution_time={row['execution_time_seconds']}s"
                ),
                flush=True,
            )

            if row["status"] == "failed" and final_exit_code == 0:
                final_exit_code = int(row["exit_code"])
            if row["status"] == "failed" and args.stop_on_failure:
                return int(row["exit_code"])

    expected_rows = len(BEAM_VALUES) * len(LOOKAHEAD_DEPTH_VALUES) * len(ALPHA_VALUES) * len(benchmarks)
    print(f"Wrote {len(rows)} result rows (expected {expected_rows})", flush=True)
    print(f"Wrote JSON index to {sweep_root / 'hyperparameter_sweep_index.json'}", flush=True)
    print(f"Wrote CSV results to {sweep_root / 'hyperparameter_sweep_results.csv'}", flush=True)
    print(f"Wrote Excel results to {sweep_root / 'hyperparameter_sweep_results.xlsx'}", flush=True)
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
