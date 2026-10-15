"""CoRAC vs RAC on repeated climbing games (paper_revision/corac_proposal.md, Sec. 6).

Runs every (env, method, seed) through src/main.py in parallel worker processes,
then summarises the final greedy test return.

    py -3.11 run_corac_matrix.py run --seeds 5 --workers 10
    py -3.11 run_corac_matrix.py summary
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
OUT = os.path.join(ROOT, "results", "corac_matrix")

ENVS = {
    "climbing": "matrix_game_rep_climbing",
    "pstoch": "matrix_game_rep_pstoch",
}
# method -> (alg config, run name prefix); hysteretic needs 'hysteretic_q' in its name.
METHODS = {
    "IQL": ("iql", "iql"),
    "Hysteretic": ("hysteretic_q", "hysteretic_q"),
    "RAC": ("dual_iql_ree", "rac"),
    "CoRAC": ("corac", "corac"),
}


def run_name(env_key, method):
    return "{}_{}".format(METHODS[method][1], env_key)


def launch(job, args):
    env_key, method, seed = job
    log_dir = os.path.join(OUT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "{}_{}_s{}.log".format(env_key, run_name(env_key, method), seed))
    done_path = log_path + ".done"
    if os.path.exists(done_path):
        return job, 0, 0.0
    cmd = [
        sys.executable, os.path.join("src", "main.py"),
        "--env-config", ENVS[env_key],
        "--config", METHODS[method][0],
        "--max-steps", str(args.max_steps),
        "with",
        "seed={}".format(seed),
        "name={}".format(run_name(env_key, method)),
        "env_args.episode_limit={}".format(args.horizon),
        "local_results_path={}".format(os.path.join(OUT, env_key)),
        "use_cuda=False",
        "use_tensorboard=False",
        "save_model=False",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.time()
    with open(log_path, "w") as log:
        code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if code == 0:
        open(done_path, "w").close()
    return job, code, time.time() - start


def run(args):
    jobs = [(env_key, method, seed)
            for seed in range(args.seeds)
            for env_key in args.envs
            for method in args.methods]
    print("{} jobs, {} workers".format(len(jobs), args.workers), flush=True)

    def staggered(i_job):
        i, job = i_job
        time.sleep(min(i, args.workers) * 3)  # avoid sacred run-id races at start
        return launch(job, args)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job, code, seconds in pool.map(staggered, enumerate(jobs)):
            print("{} {} seed={} exit={} {:.0f}s".format(*job, code, seconds), flush=True)
    summary(args)


def final_metrics(env_key, method, last_k):
    pattern = os.path.join(OUT, env_key, "metrics", "*", "*", run_name(env_key, method), "*", "metrics.csv")
    finals = {}
    for path in glob.glob(pattern):
        series = {}
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row["metric"] in ("test_return_mean", "test_battle_won_mean"):
                    series.setdefault(row["metric"], []).append((int(row["t_env"]), float(row["value"])))
        if "test_return_mean" not in series:
            continue
        seed = int(path.split("_seed_")[1].split("_")[0])
        done = os.path.join(OUT, "logs", "{}_{}_s{}.log.done".format(env_key, run_name(env_key, method), seed))
        if not os.path.exists(done):
            continue  # still running
        values = {k: np.mean([v for _, v in sorted(s)[-last_k:]]) for k, s in series.items()}
        # Keep the newest run of a seed.
        finals[seed] = (os.path.getmtime(path), values)
    return {seed: values for seed, (_, values) in finals.items()}


def summary(args):
    optimum = 11.0 * args.horizon
    print("\nFinal greedy test return (mean of last {} evals), optimum = {}".format(args.last_k, optimum))
    print("{:<10} {:<13} {:>5} {:>16} {:>12} {:>10}".format(
        "env", "method", "seeds", "return mean±se", "P(all-(a,a))", "hit opt"))
    for env_key in args.envs:
        for method in args.methods:
            finals = final_metrics(env_key, method, args.last_k)
            if not finals:
                continue
            returns = np.array([v["test_return_mean"] for v in finals.values()])
            won = np.array([v.get("test_battle_won_mean", np.nan) for v in finals.values()])
            se = returns.std(ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0.0
            hits = int((returns >= optimum - 1e-6).sum())
            print("{:<10} {:<13} {:>5} {:>9.2f} ± {:<5.2f} {:>12.2f} {:>6}/{:<3}".format(
                env_key, method, len(returns), returns.mean(), se, np.nanmean(won), hits, len(returns)))
            print("{:<29} per seed: {}".format("", " ".join(
                "s{}={:.1f}".format(s, finals[s]["test_return_mean"]) for s in sorted(finals))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "summary"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--envs", nargs="+", default=list(ENVS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--last-k", type=int, default=3)
    cli = parser.parse_args()
    run(cli) if cli.mode == "run" else summary(cli)
