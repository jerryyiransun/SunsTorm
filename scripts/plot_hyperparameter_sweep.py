#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_SWEEP_PARENT = Path("build-ubuntu22-sweep") / "hyperparameter-sweeps"

METRICS = {
    "total_latency": {
        "report_column": "Total Latency",
        "label": "Latency Speedup",
        "filename": "latency_speedup.png",
        "ylabel": "Speedup (default latency / config latency)",
        "baseline_label": "1.0 default",
    },
}

FALLBACK_DEFAULT_VALUES = {
    "beam_width": "32",
    "search_depth": "0",
    "alpha": "0.04",
}

DISPLAY_NAMES = {
    "beam_width": "Beam Width",
    "search_depth": "Lookahead Depth",
    "alpha": "Alpha",
}

TICK_NAMES = {
    "beam_width": "beam",
    "search_depth": "depth",
    "alpha": "alpha",
}

TITLE_NAMES = {
    "beam_width": "Beam Width",
    "search_depth": "Depth",
    "alpha": "Alpha",
}

ROW_KEYS = {
    "beam_width": "beam",
    "search_depth": "lookahead_depth",
    "alpha": "alpha",
}

DEFAULT_KEY_ALIASES = {
    "beam": "beam_width",
    "beam_width": "beam_width",
    "lookahead_depth": "search_depth",
    "search_depth": "search_depth",
    "alpha": "alpha",
}

TITLE_FONT_SIZE = 32
AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 24
LEGEND_TITLE_FONT_SIZE = 24
LEGEND_FONT_SIZE = 22
BAR_LABEL_FONT_SIZE = 24
HYPERPARAMETERS = ["beam_width", "search_depth", "alpha"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped bar plots from a hyperparameter sweep index."
    )
    parser.add_argument(
        "sweep",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Sweep root directory, hyperparameter_sweep_index.json, or "
            "hyperparameter_sweep_index.csv. Defaults to the newest local sweep index."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots (default: <sweep-root>/plots)",
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


def resolve_sweep_arg(path: Path | None) -> Path:
    if path is not None:
        return path

    latest = find_latest_sweep_index()
    if latest is None:
        raise FileNotFoundError(
            "No sweep index was provided, and no hyperparameter_sweep_index.json "
            f"was found under {default_sweep_parent()}"
        )
    return latest


def benchmark_sort_key(benchmark: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.json$)", benchmark)
    if match:
        return int(match.group(1)), benchmark
    return sys.maxsize, benchmark


def numeric_or_nan(value: str) -> float:
    value = value.strip()
    if value.upper() == "N/A" or not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_report(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.is_file():
        print(f"Warning: report not found: {report_path}", file=sys.stderr)
        return []

    lines = report_path.read_text(encoding="ascii").splitlines()
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = split_markdown_row(line)
        if cells and cells[0] == "Benchmark":
            header = cells
            continue
        if header is None or not cells or cells[0].startswith("---"):
            continue
        if len(cells) != len(header):
            continue

        raw_row = dict(zip(header, cells))
        rows.append(
            {
                "benchmark": raw_row["Benchmark"],
                "status": raw_row["Status"],
                "timeout_seconds": numeric_or_nan(raw_row["Timeout (s)"]),
                "total_latency": numeric_or_nan(raw_row["Total Latency"]),
                "execution_time_seconds": numeric_or_nan(raw_row["Execution Time (s)"]),
            }
        )
    return rows


def format_config_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def normalize_defaults(defaults: dict[str, Any] | None) -> dict[str, str]:
    normalized = dict(FALLBACK_DEFAULT_VALUES)
    if defaults is None:
        return normalized
    for hyperparameter, value in defaults.items():
        normalized_key = DEFAULT_KEY_ALIASES.get(hyperparameter)
        if normalized_key is not None:
            normalized[normalized_key] = format_config_value(value)
    return normalized


def load_index(index_or_root: Path) -> tuple[Path, list[dict[str, Any]], dict[str, str]]:
    path = index_or_root.expanduser().resolve()
    if path.is_dir():
        sweep_root = path
        json_path = sweep_root / "hyperparameter_sweep_index.json"
        csv_path = sweep_root / "hyperparameter_sweep_index.csv"
        if json_path.is_file():
            path = json_path
        elif csv_path.is_file():
            path = csv_path
        else:
            raise FileNotFoundError(
                f"No hyperparameter_sweep_index.json or .csv found in {sweep_root}"
            )
    else:
        sweep_root = path.parent

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="ascii"))
        return sweep_root, payload["runs"], normalize_defaults(None)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="ascii") as csv_file:
            return sweep_root, list(csv.DictReader(csv_file)), normalize_defaults(None)

    raise ValueError(f"Unsupported index file type: {path}")


def value_order_for(rows: list[dict[str, Any]], hyperparameter: str) -> list[str]:
    if rows and "hyperparameter" not in rows[0]:
        return []

    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row["hyperparameter"] != hyperparameter:
            continue
        value = str(row["value"])
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def numeric_sort_key(value: str) -> tuple[float, str]:
    try:
        return float(value), value
    except ValueError:
        return math.inf, value


def value_order_for_plot(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    hyperparameter: str,
) -> list[str]:
    values = value_order_for(rows, hyperparameter)
    if values:
        return values
    return sorted(results.get(hyperparameter, {}).keys(), key=numeric_sort_key)


def normalize_grid_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": row["benchmark"],
        "status": row.get("status", ""),
        "timeout_seconds": numeric_or_nan(str(row.get("timeout_seconds", ""))),
        "total_latency": numeric_or_nan(str(row.get("total_latency", ""))),
        "execution_time_seconds": numeric_or_nan(str(row.get("execution_time_seconds", ""))),
    }


def row_matches_defaults_except(
    row: dict[str, Any], varied_hyperparameter: str, default_values: dict[str, str]
) -> bool:
    for hyperparameter, row_key in ROW_KEYS.items():
        if hyperparameter == varied_hyperparameter:
            continue
        if format_config_value(row.get(row_key)) != default_values[hyperparameter]:
            return False
    return True


def collect_results(
    rows: list[dict[str, Any]], default_values: dict[str, str]
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    if rows and "hyperparameter" not in rows[0]:
        for row in rows:
            for hyperparameter, row_key in ROW_KEYS.items():
                if not row_matches_defaults_except(row, hyperparameter, default_values):
                    continue
                value = format_config_value(row.get(row_key))
                benchmark = row["benchmark"]
                results.setdefault(hyperparameter, {}).setdefault(value, {})[
                    benchmark
                ] = normalize_grid_row(row)
        return results

    for row in rows:
        hyperparameter = row["hyperparameter"]
        value = str(row["value"])
        report_path = Path(row["report_path"])
        results.setdefault(hyperparameter, {}).setdefault(value, {})
        for report_row in parse_report(report_path):
            results[hyperparameter][value][report_row["benchmark"]] = report_row
    return results


def plot_metric_value(
    hyperparameter: str,
    value: str,
    benchmark: str,
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    metric_key: str,
    default_values: dict[str, str],
) -> float:
    default_value = default_values[hyperparameter]
    row = results.get(hyperparameter, {}).get(value, {}).get(benchmark)
    default_row = results.get(hyperparameter, {}).get(default_value, {}).get(benchmark)
    if row is None or default_row is None:
        return math.nan

    metric_value = row[metric_key]
    default_metric_value = default_row[metric_key]
    if (
        not math.isfinite(metric_value)
        or not math.isfinite(default_metric_value)
        or metric_value == 0
    ):
        return math.nan

    return default_metric_value / metric_value


def normalized_axis_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.995, 1.005

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        lower_padding = max(abs(minimum) * 0.0025, 0.0025)
        upper_padding = max(abs(maximum) * 0.004, 0.004)
    else:
        spread = maximum - minimum
        lower_padding = max(spread * 0.25, 0.0005)
        upper_padding = max(spread * 0.85, 0.0015)

    return minimum - lower_padding, maximum + upper_padding


def format_bar_label(value: float) -> str:
    return f"{value:.4f}"


def format_x_tick_label(hyperparameter: str, value: str, default_values: dict[str, str]) -> str:
    label = f"{TICK_NAMES.get(hyperparameter, hyperparameter)}={value}"
    if value == default_values[hyperparameter]:
        return f"{label} (default)"
    return label


def display_name(hyperparameter: str) -> str:
    return DISPLAY_NAMES.get(hyperparameter, hyperparameter.replace("_", " ").title())


def subplot_title(hyperparameter: str) -> str:
    name = TITLE_NAMES.get(hyperparameter, display_name(hyperparameter))
    return f"{name} Parameter Speedup Impact"


def plot_stacked_metric(
    output_path: Path,
    metric_key: str,
    rows: list[dict[str, Any]],
    benchmarks: list[str],
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    default_values: dict[str, str],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root() / ".venv" / "matplotlib-cache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = METRICS[metric_key]
    figure_width = 22.0
    figure_height = 18.0
    fig, axes = plt.subplots(
        len(HYPERPARAMETERS),
        1,
        figsize=(figure_width, figure_height),
    )

    legend_handles = None
    legend_labels = None

    for axis, hyperparameter in zip(axes, HYPERPARAMETERS):
        values = value_order_for_plot(rows, results, hyperparameter)
        group_positions = list(range(len(values)))
        bar_width = 0.8 / max(len(benchmarks), 1)

        all_heights = [
            plot_metric_value(
                hyperparameter,
                value,
                benchmark,
                results,
                metric_key,
                default_values,
            )
            for value in values
            for benchmark in benchmarks
        ]
        finite_heights = [height for height in all_heights if math.isfinite(height)]
        y_min, y_max = normalized_axis_range(finite_heights)
        axis.set_ylim(y_min, y_max)

        for benchmark_index, benchmark in enumerate(benchmarks):
            offset = (benchmark_index - (len(benchmarks) - 1) / 2) * bar_width
            heights = [
                plot_metric_value(
                    hyperparameter,
                    value,
                    benchmark,
                    results,
                    metric_key,
                    default_values,
                )
                for value in values
            ]
            bars = axis.bar(
                [position + offset for position in group_positions],
                heights,
                width=bar_width,
                label=benchmark.replace(".json", ""),
            )
            for bar, height in zip(bars, heights):
                if not math.isfinite(height):
                    continue
                axis.annotate(
                    format_bar_label(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_FONT_SIZE,
                    rotation=90,
                    clip_on=False,
                )

        if legend_handles is None or legend_labels is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

        axis.set_title(subplot_title(hyperparameter), fontsize=TITLE_FONT_SIZE, pad=12)
        axis.set_xlabel("")
        axis.set_xticks(group_positions)
        axis.set_xticklabels(
            [format_x_tick_label(hyperparameter, value, default_values) for value in values],
            fontsize=TICK_LABEL_FONT_SIZE,
        )
        axis.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        axis.axhline(
            1.0,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=metric["baseline_label"],
        )
        axis.grid(axis="y", linestyle=":", alpha=0.45)
        axis.margins(x=0.02)

    fig.suptitle(metric["label"], fontsize=TITLE_FONT_SIZE + 6, y=0.985)
    fig.supylabel(metric["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE, x=0.01)
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(legend_labels),
            frameon=True,
        )

    fig.tight_layout(rect=(0.045, 0.035, 1.0, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        sweep_root, rows, default_values = load_index(resolve_sweep_arg(args.sweep))
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not load sweep index: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir is not None else sweep_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = collect_results(rows, default_values)
        benchmarks = sorted(
            {
                benchmark
                for hyperparameter_results in results.values()
                for value_results in hyperparameter_results.values()
                for benchmark in value_results
            },
            key=benchmark_sort_key,
        )
        if not benchmarks:
            print("No benchmark rows found in sweep reports", file=sys.stderr)
            return 1

        for hyperparameter in HYPERPARAMETERS:
            if not value_order_for_plot(rows, results, hyperparameter):
                print(f"Warning: no runs found for {hyperparameter}", file=sys.stderr)

        metric_key = "total_latency"
        metric = METRICS[metric_key]
        output_path = output_dir / metric["filename"]
        plot_stacked_metric(
            output_path,
            metric_key,
            rows,
            benchmarks,
            results,
            default_values,
        )
        print(f"Wrote {output_path}", flush=True)
    except ImportError as exc:
        print(
            "Matplotlib is required to create plots. Install matplotlib and rerun this script.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
