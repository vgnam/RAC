"""The supplied Belief-RAC Entity-GRU no-prior experiment, ready for Kaggle."""
import os
from pathlib import Path
import subprocess
import sys


def build_command(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    main_file = root / "src/main.py"
    shared_args = [
        "seed=2",
        "use_cuda=True",
        "device=0",
        "buffer_cpu_only=True",
        "batch_size=8",
        "buffer_size=500",
        "training_iters=5",
        "kl_weight=0.5",
        "target_update_interval_steps=10000",
        "test_interval=10000",
        "test_nepisode=20",
        "save_model=True",
        "save_model_interval=10000",
        "use_tensorboard=False",
        "use_wandb=True",
        "wandb_project=rac-mate",
        "wandb_mode=" + ("online" if os.environ.get("WANDB_API_KEY") else "offline"),
    ]
    return [
        sys.executable,
        str(main_file),
        "--env-config=mate_belief_entity_gru",
        "--config=belief_dual_iql_ree",
        "--max-steps", "500000",
        "with",
        *shared_args,
        "use_context_belief=True",
        "belief_direct_action=True",
        "belief_hidden_dim=64",
        "belief_decoder_hidden_dim=128",
        "belief_projection_dim=32",
        "belief_include_agent_id=True",
        "belief_temperature=0.7",
        "belief_transition_stay=0.98",
        "belief_prior_strength=0.0",
        "belief_gumbel_tau=1.0",
        "belief_dynamics_weight=1.0",
        "belief_reward_loss_weight=1.0",
        "belief_kl_weight=0.0001",
        "belief_balance_weight=0.02",
        "belief_optimism_min=0.1",
        "belief_optimism_max=1.0",
        "wandb_run_name=Belief-RAC-EntityGRU-buffer500-kl05-target10k-seed2-no-prior",
    ]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    subprocess.run(build_command(root), cwd=root, check=True)
