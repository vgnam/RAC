"""Launch the linked DM2 code against this checkout's MATE/Predator-Prey."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", choices=["mate", "pred_prey_punish"], required=True)
    parser.add_argument("--mode", choices=["dm2_sil", "dm2", "ippo"], default="dm2_sil")
    parser.add_argument("--dm2-root", type=Path,
                        default=Path(os.environ.get("DM2_ROOT", str(root / "dm2"))))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args, overrides = parser.parse_known_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    dm2_root = args.dm2_root.expanduser().resolve()
    if not (dm2_root / "src/config/default_rac_dm2_sil.yaml").is_file():
        parser.error(f"DM2 checkout with RAC integration not found: {dm2_root}")
    if overrides and overrides[0] == "with":
        overrides = overrides[1:]
    command = [sys.executable, str(dm2_root / "src/main.py"),
               f"--config=default_rac_{args.mode}", f"--env-config={args.env_config}",
               "--alg-config=ippo", f"--seed={args.seed}", "with",
               "local_results_path=" + (root / "results/dm2" / args.env_config).as_posix()]
    if args.max_steps is not None:
        command.append(f"t_max={args.max_steps}")
    command.extend(overrides)
    environment = os.environ.copy()
    environment["RAC_ROOT"] = str(root)
    return subprocess.call(command, cwd=str(dm2_root), env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
