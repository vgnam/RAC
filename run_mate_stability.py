"""HC-RAC stability ladder on MATE 4v8-0 (A -> D), see paper_revision/hc_rac_audit.md.

The ladder changes exactly one thing per step, so a regression can be attributed:

    A  corac_mate          fixed kl_weight=1, teacher = per-action max over contexts
    B  corac_mate_anneal   + kl_weight 1 -> 0 over [50k, 250k]
    C  corac_mate_argmax   + teacher_context_select: context_argmax
    D  corac_mate_stable   + top-k teacher + scale-free KL temperature

    py -3.11 run_mate_stability.py run --seeds 3 --workers 2 --max-steps 500000
    py -3.11 run_mate_stability.py summary

The summary is about *stability*, not only final return: a run that solves MATE
and then collapses should not look better than one that solves it and stays.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "mate_stability")
ENV_CONFIG = "mate_belief_entity_gru_4v8_0"

LADDER = {
    "A": "corac_mate",
    "B": "corac_mate_anneal",
    "C": "corac_mate_argmax",
    "D": "corac_mate_stable",
}
# test_return_mean is on the MATE return scale (~500 for a good policy).
DIAGNOSTICS = (
    "twin_target_over_return_max",
    "kl_constraint",
    "td_loss",
    "kl_weight",
    "kl_logit_temperature",
    "test_coverage_rate_mean",
)


def launch(job, args):
    tag, seed = job
    config = LADDER[tag]
    log_dir = os.path.join(OUT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "{}_s{}.log".format(tag, seed))
    if os.path.exists(log_path + ".done"):
        return job, 0, 0.0
    cmd = [
        sys.executable, os.path.join("src", "main.py"),
        "--env-config", ENV_CONFIG,
        "--config", config,
        "--max-steps", str(args.max_steps),
        "with",
        "seed={}".format(seed),
        "name={}".format(config),
        "local_results_path={}".format(os.path.join(OUT, tag)),
        "use_tensorboard=False",
        "save_model=False",
    ]
    if args.gpu:
        cmd += ["use_cuda=True", "device={}".format(args.device)]
    else:
        cmd += ["use_cuda=False"]
    start = time.time()
    with open(log_path, "w") as log:
        code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if code == 0:
        open(log_path + ".done", "w").close()
    return job, code, time.time() - start


def run(args):
    jobs = [(tag, seed) for seed in range(args.seeds) for tag in args.tags]
    print("{} jobs, {} workers".format(len(jobs), args.workers), flush=True)

    def staggered(item):
        i, job = item
        time.sleep(min(i, args.workers) * 5)  # avoid sacred run-id races at start
        return launch(job, args)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job, code, seconds in pool.map(staggered, enumerate(jobs)):
            print("{} seed={} exit={} {:.0f}s".format(job[0], job[1], code, seconds), flush=True)
    summary(args)


def load_series(tag, seed):
    """Long CSV -> {metric: [(t_env, value), ...]} for one (tag, seed) run."""
    pattern = os.path.join(OUT, tag, "metrics", "mate", "*", "*", "*", "metrics.csv")
    best = None
    for path in glob.glob(pattern):
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or int(rows[0]["seed"]) != seed:
            continue
        if best is None or os.path.getmtime(path) > os.path.getmtime(best[0]):
            best = (path, rows)
    if best is None:
        return None
    series = {}
    for row in best[1]:
        series.setdefault(row["metric"], []).append((int(row["t_env"]), float(row["value"])))
    for values in series.values():
        values.sort()
    return series


def stability(series, last_k):
    """Stability summary from one run's logged series."""
    test = np.array([v for _, v in series.get("test_return_mean", [])])
    times = np.array([t for t, _ in series.get("test_return_mean", [])])
    if len(test) == 0:
        return None
    final = float(test[-last_k:].mean())
    peak = float(test.max())
    peak_at = int(times[int(test.argmax())])
    solved = times[test >= 0.9 * peak]
    solve_at = int(solved[0]) if len(solved) else -1
    post = test[times > peak_at] if (times > peak_at).any() else np.array([])
    collapses = int((post < 0.7 * peak).sum())
    locked_at = int(times[test < 0.7 * final][-1]) if (test < 0.7 * final).any() else 0
    frac_high = float((test[times > 0.5 * times.max()] >= 0.9 * peak).mean()) if len(test) else 0.0

    def last(metric):
        values = series.get(metric, [])
        return values[-1][1] if values else float("nan")

    kl, td = last("kl_constraint"), last("td_loss")
    return {
        "final": final,
        "peak": peak,
        "solve_at": solve_at,
        "collapses": collapses,
        "locked_at": locked_at,
        "frac_high": frac_high,
        "twin_over_return": last("twin_target_over_return_max"),
        "kl_over_td": kl / td if td and np.isfinite(td) and td > 0 else float("nan"),
        "kl_weight": last("kl_weight"),
        "coverage": last("test_coverage_rate_mean"),
    }


def summary(args):
    print("\nFinal greedy test return (mean of last {} evals) and stability".format(args.last_k))
    header = ("{:>3} {:>10} {:>7} {:>10} {:>10} {:>9} {:>9} {:>8} {:>7} {:>9}"
              .format("tag", "config", "seeds", "final", "peak", "solve@", "collapses",
                      "lock@", ">90%pk", "twin/rng"))
    print(header)
    for tag in args.tags:
        finals, peaks, solves, collapses, locks, fracs, ratios, covers = [], [], [], [], [], [], [], []
        for seed in range(args.seeds):
            series = load_series(tag, seed)
            if series is None:
                continue
            stats = stability(series, args.last_k)
            if stats is None:
                continue
            finals.append(stats["final"])
            peaks.append(stats["peak"])
            solves.append(stats["solve_at"])
            collapses.append(stats["collapses"])
            locks.append(stats["locked_at"])
            fracs.append(stats["frac_high"])
            ratios.append(stats["twin_over_return"])
            covers.append(stats["coverage"])
        if not finals:
            continue
        print("{:>3} {:>10} {:>7} {:>10.1f} {:>10.1f} {:>9.0f} {:>9.1f} {:>8.0f} {:>7.2f} {:>9.2f}".format(
            tag, LADDER[tag].replace("corac_mate", "mate").replace("corac_", ""),
            len(finals), np.mean(finals), np.mean(peaks), np.mean(solves),
            np.mean(collapses), np.mean(locks), np.mean(fracs), np.nanmean(ratios)))
        print("{:>3} per seed: final={} collapses={}".format(
            "", " ".join("{:.0f}".format(v) for v in finals),
            " ".join("{:.0f}".format(v) for v in collapses)))
    print("\nsolve@     = first eval at >= 90% of that run's own peak")
    print("collapses  = evals below 70% of the peak *after* the peak was reached")
    print("lock@      = last eval below 70% of the final value (lower is better)")
    print(">90%pk     = fraction of evals at >= 90% of peak after the halfway point")
    print("twin/rng   = twin_target_over_return_max at the end (<= 1.5 is healthy)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "summary"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--tags", nargs="+", default=list(LADDER))
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--gpu", action="store_true", help="use_cuda=True instead of CPU")
    parser.add_argument("--device", type=int, default=0)
    cli = parser.parse_args()
    run(cli) if cli.mode == "run" else summary(cli)
