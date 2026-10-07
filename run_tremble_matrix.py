"""Tremble-Robust IQL on tabular matrix games (experiments 1-3 of the proposal).

Every method is trained with execution tremble eps (BQL is trained at eps=0) and
deployed greedily, with the same eps. A run "reaches max" when the greedy joint
intention attains max r_eps, the best expected reward any joint policy can get
under that tremble.

    py -3.11 run_tremble_matrix.py --seeds 10
"""
import argparse
import json
import os
import time

from itertools import product

import numpy as np

from src.tremble import games as G
from src.tremble import learners as L

METHODS = ["TR-IQL (ours)", "BQL (eps=0 train)", "ours, exec-keyed Q",
           "sample-max", "IQL", "hysteretic", "TR-IQL lcb", "TR-IQL pooled"]
HURWICZ = "Hurwicz per agent (eta=(1-eps)^(n-1))"
EXACT_METHODS = ["TR-IQL (ours)", "BQL (eps=0 train)", "ours, exec-keyed Q", HURWICZ]


def budget(game, args):
    n_blocks = max(args.min_blocks, args.blocks_per_joint * game.n_actions ** game.n_agents)
    return n_blocks, min(n_blocks * args.block_len, args.max_td_episodes)


def train_all_seeds(method, game, eps, args):
    """List of (q, rng) per seed; IQL variants run all seeds in one vectorised call."""
    if method in ("IQL", "hysteretic"):
        rng = np.random.default_rng(10 ** 6)
        _, td_episodes = budget(game, args)
        beta = 0.01 if method == "hysteretic" else None
        qs = L.iql(game, eps, rng, td_episodes, n_runs=args.seeds, beta=beta)
        return [(q, rng) for q in qs]
    out = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        out.append((train(method, game, eps, rng, args), rng))
    return out


def train(method, game, eps, rng, args):
    n_blocks, _ = budget(game, args)
    blk = dict(n_blocks=n_blocks, block_len=args.block_len)
    if method == "TR-IQL (ours)":
        return L.tremble_robust_iql(game, eps, rng, **blk)
    if method == "BQL (eps=0 train)":
        return L.tremble_robust_iql(game, 0.0, rng, **blk)
    if method == "ours, exec-keyed Q":
        return L.tremble_robust_iql(game, eps, rng, executed_keys=True, **blk)
    if method == "TR-IQL lcb":
        return L.tremble_robust_iql_stats(game, eps, rng, mode="lcb", **blk)
    if method == "TR-IQL pooled":
        return L.tremble_robust_iql_stats(game, eps, rng, mode="pooled", **blk)
    if method == "sample-max":
        return L.optimistic_sample_max(game, eps, rng, **blk)
    if method.startswith("Hurwicz"):
        eta = (1 - eps) ** (game.n_agents - 1)  # P(no teammate slips)
        return L.hurwicz_per_agent(game, eta, rng, **blk)
    raise ValueError(method)


def evaluate(game, eps, methods, args):
    r_eps = G.noisy_payoff(game, eps)
    best = r_eps.max()
    target_q = G.individual_optimum(r_eps)
    rows = {}
    for method in methods:
        hits, values, joints, q_err = 0, [], {}, []
        for q, rng in train_all_seeds(method, game, eps, args):
            joint = L.greedy_joint(q, rng)
            value = float(r_eps[joint])
            hits += value >= best - 1e-9
            values.append(value)
            joints[str(joint)] = joints.get(str(joint), 0) + 1
            if method == "TR-IQL (ours)":
                q_err.append(float(np.abs(q - target_q).max()))
        rows[method] = {"reach_max": hits / args.seeds, "mean_value": float(np.mean(values)),
                        "joints": joints}
        if q_err:
            rows[method]["max_abs_q_error"] = float(np.max(q_err))
    exact = {}
    for method, q in L.exact_fixed_points(game, eps).items():
        # Probability that independent greedy agents (uniform tie-breaks) reach max r_eps.
        ties = [np.flatnonzero(np.isclose(row, row.max())) for row in q]
        joints = list(product(*ties))
        exact[method] = {"reach_max": float(np.mean([r_eps[j] >= best - 1e-9 for j in joints])),
                         "mean_value": float(np.mean([r_eps[j] for j in joints]))}
    best_joints = [str(tuple(int(x) for x in j)) for j in np.argwhere(r_eps >= best - 1e-9)]
    return {"game": game.name, "eps": eps, "max_value": float(best),
            "optimal_joints": best_joints, "methods": rows, "exact": exact}


def fmt(res, methods, key="methods"):
    cells = []
    for m in methods:
        r = res[key][m]
        cells.append("{:>3.0f}% {:>6.2f}".format(100 * r["reach_max"], r["mean_value"]))
    opt = ",".join(res["optimal_joints"][:2]) + ("…" if len(res["optimal_joints"]) > 2 else "")
    return "| {:<22} | {:>5.3f} | {:>6.2f} {:<14} | ".format(
        res["game"], res["eps"], res["max_value"], opt) + " | ".join(cells) + " |"


def header(methods):
    return ("| game                   |  eps  | max r_eps (argmax)    | "
            + " | ".join(methods) + " |\n|" + "---|" * (3 + len(methods)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=METHODS,
                   help="subset of: " + ", ".join(METHODS + [HURWICZ]))
    p.add_argument("--block-len", type=int, default=200)
    p.add_argument("--blocks-per-joint", type=int, default=25)
    p.add_argument("--min-blocks", type=int, default=250)
    p.add_argument("--max-td-episodes", type=int, default=200000)
    p.add_argument("--out", default="results/tremble_matrix")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    start = time.time()

    exp1 = [(G.climbing(), [0.0, 0.05, 0.1, 0.15, 0.2]),
            (G.penalty(), [0.0, 0.1, 0.2]),
            (G.matrix_game_3(), [0.0, 0.1, 0.2]),
            (G.climbing_partially_stochastic(), [0.0, 0.1, 0.2]),
            (G.climbing_fully_stochastic(), [0.0, 0.1, 0.2])]
    exp2 = [(G.asymmetric_2x2(), [0.25])]
    exp3 = [(G.n_agent_coordination(n), [0.05, 0.1]) for n in (2, 3, 4, 6, 8)]
    results = {"args": vars(args), "exp1": [], "exp2": [], "exp3": []}
    methods = args.methods
    exact_methods = [m for m in methods if m in EXACT_METHODS]

    for name, suite in [("exp1", exp1), ("exp2", exp2), ("exp3", exp3)]:
        print("\n## {}: learned\n".format(name))
        print(header(methods))
        for game, eps_list in suite:
            for eps in eps_list:
                res = evaluate(game, eps, methods, args)
                results[name].append(res)
                print(fmt(res, methods), flush=True)
        if exact_methods:
            print("\n## {}: exact fixed point (P over uniform tie-breaks)\n".format(name))
            print(header(exact_methods))
            for res in results[name]:
                print(fmt(res, exact_methods, key="exact"))

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nCells: % seeds reaching max r_eps, mean r_eps of greedy joint. {:.0f}s".format(
        time.time() - start))


if __name__ == "__main__":
    main()
