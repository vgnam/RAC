"""Tremble-rate shift on tabular matrix games: train at eps, deploy at eps' in [eps, eps_max].

TR-IQL-DRO scores each block by mean - kappa * std with kappa = chi2_kappa(eps, eps_max, n):
the chi^2 ball of that radius around K_eps contains K_eps' for every eps' in [eps, eps_max],
so the score lower-bounds the intention's value under any of those tremble rates.

For each method the greedy joint intention is evaluated at eps (no shift), at eps_max, and
by its max regret over a grid of eps' in [eps, eps_max] (0 = optimal at every rate).

    py -3.11 run_tremble_shift.py --seeds 3 --block-len 1000 --games climbing penalty
"""
import argparse
import json
import os
import time
from itertools import product

import numpy as np

import run_tremble_matrix as R
from src.tremble import games as G
from src.tremble import learners as L

GAMES = {"climbing": G.climbing, "penalty": G.penalty, "matrix_game_3": G.matrix_game_3,
         "climbing_partial_stoch": G.climbing_partially_stochastic,
         "climbing_full_stoch": G.climbing_fully_stochastic,
         "n_agent_3": lambda: G.n_agent_coordination(3),
         "n_agent_4": lambda: G.n_agent_coordination(4)}
PAIRS = [(0.05, 0.1), (0.05, 0.2), (0.1, 0.15), (0.1, 0.2), (0.15, 0.2)]
METHODS = ["TR-IQL", "TR-IQL-DRO", "IQL", "hysteretic"]


def max_regret(game, joint, eps, eps_max, n_grid=11):
    regrets = []
    for e in np.linspace(eps, eps_max, n_grid):
        r = G.noisy_payoff(game, e)
        regrets.append(r.max() - r[joint])
    return float(max(regrets))


def exact_joints(game, eps, kappa):
    """Greedy joints at the exact fixed point (all uniform tie-break combinations)."""
    q = G.individual_optimum(G.noisy_payoff(game, eps) - kappa * G.noisy_std(game, eps))
    ties = [np.flatnonzero(np.isclose(row, row.max())) for row in q]
    return [tuple(int(a) for a in j) for j in product(*ties)]


def summarise(game, joints, eps, eps_max):
    r_tr, r_max = G.noisy_payoff(game, eps), G.noisy_payoff(game, eps_max)
    return {"reach_train": float(np.mean([r_tr[j] >= r_tr.max() - 1e-9 for j in joints])),
            "reach_max": float(np.mean([r_max[j] >= r_max.max() - 1e-9 for j in joints])),
            "max_regret": float(np.mean([max_regret(game, j, eps, eps_max) for j in joints])),
            "joints": {str(j): joints.count(j) for j in set(joints)}}


def train_joints(method, game, eps, kappa, args):
    n_blocks, td_episodes = R.budget(game, args)
    if method in ("IQL", "hysteretic"):
        rng = np.random.default_rng(10 ** 6)
        beta = 0.01 if method == "hysteretic" else None
        qs = L.iql(game, eps, rng, td_episodes, n_runs=args.seeds, beta=beta)
        return [L.greedy_joint(q, rng) for q in qs]
    joints = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        q = L.tremble_robust_iql(game, eps, rng, n_blocks=n_blocks, block_len=args.block_len,
                                 kappa=kappa if method == "TR-IQL-DRO" else 0.0)
        joints.append(L.greedy_joint(q, rng))
    return joints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--games", nargs="+", default=list(GAMES), choices=list(GAMES))
    p.add_argument("--block-len", type=int, default=1000)
    p.add_argument("--blocks-per-joint", type=int, default=25)
    p.add_argument("--min-blocks", type=int, default=250)
    p.add_argument("--max-td-episodes", type=int, default=10 ** 7)
    p.add_argument("--out", default="results/tremble_shift")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    results = []
    print("| game | eps -> eps_max | kappa | " + " | ".join(METHODS)
          + " |\n|" + "---|" * (3 + len(METHODS)))
    for name in args.games:
        game = GAMES[name]()
        cache = {}  # methods that do not depend on eps_max are trained once per eps
        for eps, eps_max in PAIRS:
            kappa = G.chi2_kappa(eps, eps_max, game.n_agents)
            row = {"game": name, "eps": eps, "eps_max": eps_max, "kappa": kappa,
                   "methods": {}, "exact": {}}
            for method in METHODS:
                key = (method, eps, eps_max if method == "TR-IQL-DRO" else None)
                if key not in cache:
                    cache[key] = train_joints(method, game, eps, kappa, args)
                row["methods"][method] = summarise(game, cache[key], eps, eps_max)
            for method, k in [("TR-IQL", 0.0), ("TR-IQL-DRO", kappa)]:
                row["exact"][method] = summarise(game, exact_joints(game, eps, k), eps, eps_max)
            results.append(row)
            cells = ["{:>3.0f}/{:>3.0f}% {:.2f}".format(100 * m["reach_train"], 100 * m["reach_max"],
                                                        m["max_regret"])
                     for m in row["methods"].values()]
            print("| {} | {} -> {} | {:.2f} | {} |".format(name, eps, eps_max, kappa, " | ".join(cells)),
                  flush=True)
        with open(os.path.join(args.out, "summary_{}.json".format("_".join(args.games))), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
    print("\nCells: % seeds optimal at eps / at eps_max, mean max regret over [eps, eps_max]. "
          "{:.0f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
