# Kaggle: MATE Entity-Attention GRU IQL

After extracting the archive, install the Kaggle-specific dependencies from
the repository root:

```bash
pip install -q -r requirements-kaggle.txt
```

Run one seed on one T4:

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py \
  --env-config mate_entity_gru \
  --config iql \
  --max-steps 1000000 \
  with \
  seed=0 \
  use_cuda=True \
  device=0 \
  buffer_cpu_only=True \
  batch_size=8 \
  buffer_size=1000 \
  training_iters=5 \
  lr=0.0005 \
  epsilon_anneal_time=100000 \
  test_interval=25000 \
  test_nepisode=50 \
  save_model=True \
  use_wandb=True \
  wandb_project=rac-mate \
  wandb_group=iql-entity-gru \
  wandb_run_name=IQL-EntityAttn-GRU-seed0 \
  use_tensorboard=False
```

The code uses a single GPU per process. On a Kaggle T4 x2 session, the most
efficient use of both GPUs is to launch a second seed as a separate process,
pin it with `CUDA_VISIBLE_DEVICES=1`, and still pass `device=0` inside that
process because the selected physical GPU is remapped to local CUDA index 0.

The replay learner batches entity encoding and attention over the entire
episode and calls the GRU once per sampled sequence. Online action selection
continues to use the equivalent one-step recurrent path.

## Belief-RAC + Entity-Attention GRU

Use the dedicated environment configuration so both the main Q-network and
the context-conditioned twin receive the Entity-GRU architecture:

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py \
  --env-config mate_belief_entity_gru \
  --config belief_dual_iql_ree \
  --max-steps 500000 \
  with \
  seed=0 \
  use_cuda=True \
  device=0 \
  buffer_cpu_only=True \
  batch_size=8 \
  buffer_size=1000 \
  training_iters=5 \
  lr=0.0001 \
  kl_weight=1.0 \
  target_update_interval_steps=1000 \
  epsilon_anneal_time=150000 \
  test_interval=10000 \
  test_nepisode=20 \
  save_model=True \
  save_model_interval=10000 \
  use_wandb=True \
  wandb_project=rac-mate \
  wandb_group=belief-entity-gru \
  wandb_run_name=Belief-RAC-EntityGRU-500k-seed0 \
  use_tensorboard=False
```

The twin encodes each observation once and evaluates all categorical contexts
through one vectorized hyper-head. The belief GRU is also fused over replay
sequences; only the causal posterior update remains sequential.
