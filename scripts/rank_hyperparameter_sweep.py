#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SWEEP_PARENT = Path("build-ubuntu22-sweep") / "hyperparameter-sweeps"


@dataclass(frozen=True)
class Config:
    beam: int
    lookahead_depth: int
    alpha: float


@dataclass
class ConfigScore:
    config: Config
    final_score: float
    benchmark_scores: dict[str, float]
    benchmark_latencies: dict[str, float]
    statuses: Counter[str]
    missing_benchmarks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank hyperparameter configurations by summed inverse latency ratios against "
            "the best latency observed for each benchmark."
        )
    )
    parser.add_argument(
        "index",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to hyperparameter_sweep_index.json, or a sweep directory containing it "
            "(default: newest local sweep index)."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only print the top N configurations (default: print all).",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_sweep_parent() -> Path:
    return repo_root() / DEFAULT_SWEEP_PARENT


def find_latest_sweep_index() -> Path | None:
    sweep_parent = default_sweep_parent()
    if not sweep_parent.is_dir():
        return None
    candidates = sorted(
        sweep_parent.glob("*/hyperparameter_sweep_index.json"),
        key=lambda path: path.parent.name,
    )
    if not candidates:
        return None
    return candidates[-1]


def resolve_index_arg(path: Path | None) -> Path:
    if path is not None:
        return path

    latest = find_latest_sweep_index()
    if latest is None:
        raise FileNotFoundError(
            "No sweep index was provided, and no hyperparameter_sweep_index.json "
            f"was found under {default_sweep_parent()}"
        )
    return latest


def resolve_index(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "hyperparameter_sweep_index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Hyperparameter sweep index not found: {path}")
    return path


def benchmark_sort_key(benchmark: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.json$)", benchmark)
    if match:
        return int(match.group(1)), benchmark
    return sys.maxsize, benchmark


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip()
    if not value_str or value_str.upper() == "N/A":
        return math.nan
    try:
        return float(value_str)
    except ValueError:
        return math.nan


def parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def config_from_run(run: dict[str, Any]) -> Config:
    return Config(
        beam=parse_int(run.get("beam"), "beam"),
        lookahead_depth=parse_int(run.get("lookahead_depth"), "lookahead_depth"),
        alpha=parse_float(run.get("alpha")),
    )


def format_number(value: float, precision: int = 6) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "N/A"
    if value == 0:
        return "0"
    if abs(value) >= 100000 or abs(value) < 0.001:
        return f"{value:.{precision}g}"
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def format_alpha(value: float) -> str:
    return format_number(value, precision=8)


def load_runs(index_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Sweep index must be a JSON object")

    benchmarks = data.get("benchmarks")
    runs = data.get("runs")
    if not isinstance(benchmarks, list) or not all(isinstance(item, str) for item in benchmarks):
        raise ValueError("Sweep index is missing a string list at key 'benchmarks'")
    if not isinstance(runs, list) or not all(isinstance(item, dict) for item in runs):
        raise ValueError("Sweep index is missing a dict list at key 'runs'")

    return sorted(benchmarks, key=benchmark_sort_key), runs


def collect_latencies(
    benchmarks: list[str], runs: list[dict[str, Any]]
) -> tuple[dict[Config, dict[str, float]], dict[Config, Counter[str]], dict[str, float]]:
    latencies_by_config: dict[Config, dict[str, float]] = defaultdict(dict)
    statuses_by_config: dict[Config, Counter[str]] = defaultdict(Counter)
    best_latency_by_benchmark: dict[str, float] = {}

    known_benchmarks = set(benchmarks)
    for run in runs:
        benchmark = str(run.get("benchmark", ""))
        if benchmark not in known_benchmarks:
            continue

        config = config_from_run(run)
        status = str(run.get("status", "unknown"))
        statuses_by_config[config][status] += 1

        latency = parse_float(run.get("total_latency"))
        if not math.isfinite(latency) or latency <= 0:
            continue

        previous_latency = latencies_by_config[config].get(benchmark)
        if previous_latency is None or latency < previous_latency:
            latencies_by_config[config][benchmark] = latency

        best = best_latency_by_benchmark.get(benchmark)
        if best is None or latency < best:
            best_latency_by_benchmark[benchmark] = latency

    return dict(latencies_by_config), dict(statuses_by_config), best_latency_by_benchmark


def score_configs(
    benchmarks: list[str],
    latencies_by_config: dict[Config, dict[str, float]],
    statuses_by_config: dict[Config, Counter[str]],
    best_latency_by_benchmark: dict[str, float],
) -> list[ConfigScore]:
    scores: list[ConfigScore] = []

    for config in sorted(
        set(latencies_by_config) | set(statuses_by_config),
        key=lambda item: (item.beam, item.lookahead_depth, item.alpha),
    ):
        final_score = 0.0
        missing_benchmarks = 0
        benchmark_scores: dict[str, float] = {}
        benchmark_latencies = latencies_by_config.get(config, {})

        for benchmark in benchmarks:
            latency = benchmark_latencies.get(benchmark)
            best_latency = best_latency_by_benchmark.get(benchmark)
            if latency is None or best_latency is None:
                benchmark_scores[benchmark] = math.inf
                final_score = math.inf
                missing_benchmarks += 1
                continue

            ratio = best_latency / latency
            benchmark_scores[benchmark] = ratio
            if not math.isinf(final_score):
                final_score += ratio

        scores.append(
            ConfigScore(
                config=config,
                final_score=final_score,
                benchmark_scores=benchmark_scores,
                benchmark_latencies=benchmark_latencies,
                statuses=statuses_by_config.get(config, Counter()),
                missing_benchmarks=missing_benchmarks,
            )
        )

    return sorted(
        scores,
        key=lambda item: (
            -item.final_score,
            item.config.beam,
            item.config.lookahead_depth,
            item.config.alpha,
        ),
    )


def make_table(rows: list[list[str]]) -> str:
    widths = [0] * len(rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            "  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row))
        )
        if row_index == 0:
            rendered.append("  ".join("-" * width for width in widths))
    return "\n".join(rendered)


def status_summary(statuses: Counter[str]) -> str:
    if not statuses:
        return ""
    return ",".join(f"{status}:{count}" for status, count in sorted(statuses.items()))


def print_ranking(
    index_path: Path,
    benchmarks: list[str],
    best_latency_by_benchmark: dict[str, float],
    scores: list[ConfigScore],
    top: int,
) -> None:
    print(f"Sweep index: {index_path}")
    print("Scoring: sum(best_latency / current_latency) per benchmark; higher is better.")
    print("Tie-breakers: smallest beam, then smallest depth, then smallest alpha.")
    print()

    best_rows = [["benchmark", "best_latency"]]
    for benchmark in benchmarks:
        best_rows.append(
            [benchmark, format_number(best_latency_by_benchmark.get(benchmark, math.nan))]
        )
    print("Best latency by benchmark:")
    print(make_table(best_rows))
    print()

    visible_scores = scores if top <= 0 else scores[:top]
    header = ["rank", "score", "beam", "depth", "alpha"]
    header.extend(f"{Path(benchmark).stem} ratio" for benchmark in benchmarks)
    header.extend(["missing", "statuses"])

    rows = [header]
    for rank, score in enumerate(visible_scores, start=1):
        row = [
            str(rank),
            format_number(score.final_score),
            str(score.config.beam),
            str(score.config.lookahead_depth),
            format_alpha(score.config.alpha),
        ]
        row.extend(format_number(score.benchmark_scores[benchmark]) for benchmark in benchmarks)
        row.extend([str(score.missing_benchmarks), status_summary(score.statuses)])
        rows.append(row)

    print("Ranked hyperparameter configurations:")
    print(make_table(rows))
    if top > 0 and len(scores) > top:
        print(f"\nShowing top {top} of {len(scores)} configurations.")


def main() -> int:
    args = parse_args()
    try:
        index_path = resolve_index(resolve_index_arg(args.index))
        benchmarks, runs = load_runs(index_path)
        latencies_by_config, statuses_by_config, best_latency_by_benchmark = collect_latencies(
            benchmarks, runs
        )

        missing_best = [
            benchmark for benchmark in benchmarks if benchmark not in best_latency_by_benchmark
        ]
        if missing_best:
            print(
                "No valid latency found for benchmark(s): " + ", ".join(missing_best),
                file=sys.stderr,
            )
            return 1

        scores = score_configs(
            benchmarks, latencies_by_config, statuses_by_config, best_latency_by_benchmark
        )
        if not scores:
            print("No hyperparameter configurations with sweep runs found.", file=sys.stderr)
            return 1

        print_ranking(index_path, benchmarks, best_latency_by_benchmark, scores, args.top)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
