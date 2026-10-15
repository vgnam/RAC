"""Upper-CVaR IQL vs IQL / hysteretic / distributed Q on tabular matrix games.

No execution tremble: the question is relative overgeneralisation (IQL) versus
optimism about environment noise (hysteretic, distributed Q, UCVaR in stochastic games).
A run "reaches max" when its greedy joint action attains the best expected payoff.
``q_err`` is |Q_i(g_i) - max_{a_-i} r(g_i, a_-i)| at the greedy actions g, averaged over
agents and seeds: the distance to the BQL / distributed-Q target.

    py -3.11 run_ucvar_matrix.py --seeds 10
"""
import argparse
import json
import os
import time

from itertools import product

import numpy as np

from src.tremble import games as G
from src.tremble import learners as L

ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]


def methods(alphas):
    out = {"IQL": dict(kind="iql"),
           "hyst 0.1": dict(kind="iql", beta=0.1),
           "hyst 0.01": dict(kind="iql", beta=0.01),
           "distQ (hyst 0)": dict(kind="iql", beta=0.0)}
    for a in alphas:
        out["UCVaR {}".format(a)] = dict(kind="ucvar", alpha=a)
    return out


def train(spec, game, args):
    rng = np.random.default_rng(10 ** 6)
    common = dict(n_runs=args.seeds, lr=args.lr, explore_end=args.explore_end)
    if spec["kind"] == "iql":
        return L.iql(game, 0.0, rng, args.episodes, beta=spec.get("beta"), **common)
    return L.ucvar_iql(game, 0.0, rng, args.episodes, alpha=spec["alpha"], lr_eta=args.lr_eta, **common)


def evaluate(game, specs, args):
    payoff = game.mean_payoff
    best = payoff.max()
    target = G.individual_optimum(payoff)  # BQL / distributed-Q target
    rng = np.random.default_rng(0)
    rows = {}
    for name, spec in specs.items():
        hits, values, errs = 0, [], []
        for q in train(spec, game, args):
            joint = L.greedy_joint(q, rng)
            hits += payoff[joint] >= best - 1e-9
            values.append(float(payoff[joint]))
            errs.append(float(np.mean([abs(q[i, a] - target[i, a]) for i, a in enumerate(joint)])))
        rows[name] = {"reach_max": hits / args.seeds, "mean_value": float(np.mean(values)),
                      "q_err": float(np.mean(errs))}
    exact = {"BQL oracle": target}
    for a in args.alphas:
        exact["UCVaR {} (uniform teammates)".format(a)] = L.exact_ucvar(game, a)
    exact_rows = {}
    for name, q in exact.items():
        ties = [np.flatnonzero(np.isclose(row, row.max())) for row in q]
        exact_rows[name] = float(np.mean([payoff[j] >= best - 1e-9 for j in product(*ties)]))
    return {"game": game.name, "max_value": float(best), "methods": rows, "exact_reach": exact_rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-eta", type=float, default=0.1)
    p.add_argument("--explore-end", type=float, default=0.05)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    p.add_argument("--out", default="results/ucvar_matrix")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    specs = methods(args.alphas)
    suite = [G.climbing(), G.penalty(), G.matrix_game_3(),
             G.climbing_partially_stochastic(), G.climbing_fully_stochastic()]
    suite += [G.n_agent_coordination(n) for n in (2, 3, 4, 5)]
    suite += [G.n_agent_climbing(n) for n in (2, 3, 4, 5)]  # alpha* = 1.2 * 3^-(n-1): .4 .13 .044 .015

    names = list(specs)
    print("reach_max % (mean q_err) over {} seeds, {} episodes\n".format(args.seeds, args.episodes))
    print("| game | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1))
    results = []
    for game in suite:
        res = evaluate(game, specs, args)
        results.append(res)
        cells = ["{:.0f}% ({:.1f})".format(100 * r["reach_max"], r["q_err"]) for r in res["methods"].values()]
        print("| {} | ".format(game.name) + " | ".join(cells) + " |", flush=True)

    exact_names = list(results[0]["exact_reach"])
    print("\nexact: P(greedy reaches max) of the fixed point under uniform teammates\n")
    print("| game | " + " | ".join(exact_names) + " |\n|" + "---|" * (len(exact_names) + 1))
    for res in results:
        print("| {} | ".format(res["game"])
              + " | ".join("{:.0f}%".format(100 * v) for v in res["exact_reach"].values()) + " |")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print("\n{:.0f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
