# DM2 on RAC's MATE and Predator-Prey environments

Run from `D:\RAC\RAC_code` using the existing Python 3.10 installation:

```powershell
py -3.10 run_dm2.py --env-config mate --max-steps 1000000
py -3.10 run_dm2.py --env-config pred_prey_punish --max-steps 1000000
```

These commands launch the modified DM2 checkout at `D:\dm2` through the local
`dm2` directory link. The launcher sets `RAC_ROOT` so DM2 imports the actual
environment files and reads environment arguments from this RAC checkout.
The original `src/main.py` commands for RAC, Belief-RAC, IQL and IPPO still work.

The default mode is **DM2 with self-imitation learning (SIL)**: IPPO plus a
discriminator trained against the highest-return episodes collected so far.
It runs without external demonstrations. It is not the paper's expert-data
experiment, and the new task hyperparameters have not been tuned for convergence.

Modes:

| `--mode` | Behavior |
| --- | --- |
| `dm2_sil` (default) | DM2 with self-imitation, no expert data required |
| `ippo` | DM2 repository's independent PPO baseline, environment reward only |
| `dm2` | DM2 with expert demonstrations supplied through `gail_data_paths` |

For example, run the DM2 IPPO baseline or another MATE scenario:

```powershell
py -3.10 run_dm2.py --env-config mate --mode ippo --seed 1
py -3.10 run_dm2.py --env-config mate with env_args.mate_config=MATE-4v2-9.yaml
```

MATE retains RAC's camera actions, target policy, frame skip, observation
wrappers and coverage reward. Predator-Prey retains the same grid, capture rules,
penalties and action numbering; frozen agents use action 4 (`stay`).
DM2 keeps its own policy/discriminator architecture, separate parameters per
agent and training schedule. It does not inherit RAC's algorithm hyperparameters.

## Installation on another machine

Keep both modified checkouts. Set `DM2_ROOT` or pass `--dm2-root` when the
directory link is unavailable. Installing dependencies alone does not install
the integration changes into a fresh upstream DM2 clone.

```powershell
py -3.10 -m pip install -r D:\dm2\requirements-rac.txt
py -3.10 run_dm2.py --dm2-root D:\dm2 --env-config pred_prey_punish
```

The MATE package is only imported for MATE. Neither backend requires SMAC or
StarCraft II. The original DM2 `requirements.txt` contains obsolete pins; use
`requirements-rac.txt` for this integration. TensorBoard is disabled by default.
The tested runtime uses Python 3.10, PyTorch 2.8 CPU, NumPy 1.26 and MATE 0.1.

Direct launch from `D:\dm2` is also supported:

```powershell
$env:RAC_ROOT = 'D:\RAC\RAC_code'
py -3.10 src/main.py --config=default_rac_dm2_sil --env-config=mate --alg-config=ippo --seed=0
```

## Short functional check

This runs real episodes and at least one policy/discriminator update with a
shortened horizon and small networks. Substitute `mate` to check the other task.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
py -3.10 run_dm2.py --env-config pred_prey_punish --max-steps 16 with use_cuda=False env_args.episode_limit=8 buffer_size=2 batch_size=2 gail_batch_size=2 ppo_epoch=1 gail_epoch=1 rnn_hidden_dim=16 mlp_hidden_dim=16 test_nepisode=1 save_model=False
```

Training completes whole episodes, so the actual step count may exceed the
requested budget. Normal runs keep the source environment's full horizon:
500 decisions for MATE and 200 for Predator-Prey.

## Outputs and expert data

Launcher model/data outputs go to `RAC_code/results/dm2/<env-config>/`.
Sacred run configuration, status and metrics are under `D:\dm2\results\sacred`.
MATE evaluation logs coverage; both tasks log returns and learner losses.

For expert DM2, use demonstrations collected in the same environment, scenario,
observation settings and action space. Each data directory contains one
`agent_<index>.npz` per agent in DM2's saved trajectory format. SMAC demonstration
data from the original repository cannot be used for MATE or Predator-Prey.

To collect trajectories from an IPPO policy (set `checkpoint_paths` to a trained
checkpoint's parent directory when collecting expert-quality data):

```powershell
py -3.10 run_dm2.py --env-config mate --mode ippo with save_agent_batches=True save_agent_batchsize=64 save_agent_batches_interval=30000
```

Then supply the actual saved timestep directory:

```powershell
py -3.10 run_dm2.py --env-config mate --mode dm2 with "gail_data_paths=['D:/path/to/agents_batches/run/30000']" gail_exp_eps=64
```

`gail_exp_eps` and `gail_batch_size` count stored episodes in this implementation;
the available expert episode count must be at least `gail_batch_size`.

## Verification

Integration tests run real MATE and Predator-Prey, check actor/discriminator
parameter changes, finite losses, action masks, module isolation and SIL storage:

```powershell
cd D:\dm2
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
py -3.10 -m pytest tests/test_rac_integration.py -q --disable-warnings
```

The environment variable avoids unrelated globally installed pytest plugins.
