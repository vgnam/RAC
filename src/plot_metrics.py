"""Plot comparable learning curves from per-run long-form CSV metric logs."""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_curves(root, metric, env=None, scenario=None, algorithms=None):
    curves = defaultdict(lambda: defaultdict(list))
    algorithms = set(algorithms) if algorithms else None

    for csv_path in Path(root).rglob("metrics.csv"):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["metric"] != metric:
                    continue
                if env is not None and row["env"] != env:
                    continue
                if scenario is not None and row["scenario"] != scenario:
                    continue
                if algorithms is not None and row["algorithm"] not in algorithms:
                    continue
                curves[row["algorithm"]][row["run_id"]].append(
                    (int(row["t_env"]), float(row["value"]))
                )

    return curves


def _sorted_curve(points):
    by_step = defaultdict(list)
    for step, value in points:
        by_step[step].append(value)
    sorted_steps = sorted(by_step)
    steps = np.asarray(sorted_steps, dtype=np.float64)
    values = np.asarray([np.mean(by_step[step]) for step in sorted_steps], dtype=np.float64)
    return steps, values


def plot_metric(root, metric, output, env=None, scenario=None, algorithms=None):
    curves = _load_curves(root, metric, env=env, scenario=scenario, algorithms=algorithms)
    if not curves:
        return False

    is_percentage = "coverage_rate" in metric
    scale = 100.0 if is_percentage else 1.0

    figure, axis = plt.subplots(figsize=(10, 6))
    for algorithm, runs in sorted(curves.items()):
        prepared = [_sorted_curve(points) for points in runs.values() if points]
        if not prepared:
            continue

        for steps, values in prepared:
            marker = "o" if len(steps) == 1 else None
            axis.plot(steps, scale * values, alpha=0.18, linewidth=1, marker=marker)

        if len(prepared) == 1:
            mean_steps, mean_values = prepared[0]
            marker = "o" if len(mean_steps) == 1 else None
            axis.plot(
                mean_steps,
                scale * mean_values,
                linewidth=2.2,
                marker=marker,
                label=algorithm,
            )
            continue

        common_start = max(steps[0] for steps, _ in prepared)
        common_end = min(steps[-1] for steps, _ in prepared)
        if common_end < common_start:
            continue
        grid_size = min(300, max(len(steps) for steps, _ in prepared))
        grid_size = max(grid_size, 2)
        grid = np.linspace(common_start, common_end, grid_size)
        samples = np.vstack([np.interp(grid, steps, values) for steps, values in prepared])
        mean = scale * samples.mean(axis=0)
        std = scale * samples.std(axis=0)
        axis.plot(grid, mean, linewidth=2.2, label=f"{algorithm} (n={len(prepared)})")
        axis.fill_between(grid, mean - std, mean + std, alpha=0.2)

    axis.set_xlabel("Training environment steps (t_env)")
    axis.set_ylabel("Coverage (%)" if is_percentage else metric)
    title_parts = [metric]
    if env:
        title_parts.append(env)
    if scenario:
        title_parts.append(scenario)
    axis.set_title(" — ".join(title_parts))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/metrics", help="Root containing metrics.csv files")
    parser.add_argument("--metric", required=True, help="Metric name, e.g. test_coverage_rate_mean")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--env", default=None, help="Optional environment filter")
    parser.add_argument("--scenario", default=None, help="Optional map/config filter")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Optional algorithm filters")
    args = parser.parse_args()

    plotted = plot_metric(
        root=args.root,
        metric=args.metric,
        output=args.output,
        env=args.env,
        scenario=args.scenario,
        algorithms=args.algorithms,
    )
    if not plotted:
        raise SystemExit("No matching metric data found.")
    print("Saved plot to {}".format(args.output))


if __name__ == "__main__":
    main()
