"""UCVaR-IQL on LBF or Predator-Prey, ready for Kaggle.

Runs one training per (alpha, seed), sequentially. ``alpha=1`` is IQL (sanity check).

    python kaggle_run_ucvar.py --env lbf --alphas 0.5 0.3 0.1 --seeds 0
    python kaggle_run_ucvar.py --env pred_prey --alphas 0.5 0.3 0.1 --seeds 0 1 2

``--lenient`` (pred_prey only) runs the lenient schedule: each ``--alphas`` value is then
the starting alpha of a (observation, action) pair, which rises to 1 with its visits
(``--count-scale`` visits per e-fold).

    python kaggle_run_ucvar.py --env pred_prey --lenient --alphas 0.05 --count-scale 1000
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ENVS = {
    # env-config, alg-config, default t_max, run-name tag
    "lbf": ("lbf", "lbf_ucvar_iql", 2_000_000, "lbf5x5-2M"),
    "pred_prey": ("pred_prey_punish", "pred_prey_ucvar_iql", 1_000_000, "predprey-1M"),
}
LENIENT_ALG = {"pred_prey": "pred_prey_lucvar_iql"}
# Must match the key of the IQL / TR-IQL LBF runs being compared against.
LBF_KEY = "Foraging-5x5-2p-1f-coop-v3"


def build_command(root, env, alpha, seed, t_max, wandb_project, lbf_key, tremble_eps,
                  lenient=False, count_scale=None):
    env_config, alg_config, default_t_max, tag = ENVS[env]
    t_max = t_max or default_t_max
    if lenient:
        alg_config = LENIENT_ALG[env]
        run_name = "LUCVaR{}-k{:g}-{}-eps{}-seed{}".format(alpha, count_scale, tag, tremble_eps, seed)
        alpha_args = ["ucvar_alpha_start={}".format(alpha), "ucvar_count_scale={:g}".format(count_scale)]
    else:
        run_name = "UCVaR{}-{}-eps{}-seed{}".format(alpha, tag, tremble_eps, seed)
        alpha_args = ["ucvar_alpha={}".format(alpha)]
    args = [
        "seed={}".format(seed),
        *alpha_args,
        "use_cuda=True",
        "device=0",
        "buffer_cpu_only=True",
        "use_tensorboard=False",
        "use_wandb=True",
        "wandb_project={}".format(wandb_project),
        "wandb_mode=" + ("online" if os.environ.get("WANDB_API_KEY") else "offline"),
        "wandb_run_name={}".format(run_name),
        "env_args.tremble_eps={}".format(tremble_eps),
    ]
    if env == "lbf":
        args.append("env_args.key={}".format(lbf_key))
    return [sys.executable, str(root / "src/main.py"),
            "--env-config={}".format(env_config), "--config={}".format(alg_config),
            "--max-steps", str(t_max), "with", *args]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENVS), required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="default: 0.3, or 0.05 with --lenient")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--t-max", type=int, default=None, help="default: 2M for lbf, 1M for pred_prey")
    p.add_argument("--wandb-project", default="rac-marl")
    p.add_argument("--lbf-key", default=LBF_KEY)
    p.add_argument("--tremble-eps", type=float, default=0.0)
    p.add_argument("--lenient", action="store_true",
                   help="per (obs, action) alpha schedule from --alphas to 1 (pred_prey only)")
    p.add_argument("--count-scale", type=float, default=1000,
                   help="lenient: visits per e-fold of the alpha schedule")
    args = p.parse_args()
    if args.lenient and args.env not in LENIENT_ALG:
        p.error("--lenient is only available for: " + ", ".join(sorted(LENIENT_ALG)))
    if args.alphas is None:
        args.alphas = [0.05] if args.lenient else [0.3]
    root = Path(__file__).resolve().parent
    for seed in args.seeds:
        for alpha in args.alphas:
            cmd = build_command(root, args.env, alpha, seed, args.t_max, args.wandb_project,
                                args.lbf_key, args.tremble_eps, args.lenient,
                                args.count_scale if args.lenient else None)
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
