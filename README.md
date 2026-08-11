# In-Context Fully Decentralized Cooperative Multi-Agent Reinforcement Learning

Official code for the paper "In-Context Fully Decentralized Cooperative Multi-Agent Reinforcement Learning" submitted to NeurIPS 2025.

This repository develops RAC algorithm on both Matrix Game, Predator and Prey, and StarCraft Multi-Agent Challenge benchmarks.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

For the MATE environment (Multi-Agent Tracking Environment), install it from source (requires `gym>=0.13,<1.0`, the repo pins `gym==0.22.0`):

```setup
git config --global core.symlinks true  # required on Windows
pip install git+https://github.com/XuehaiPan/mate.git#egg=mate
```

## Training

Sacred capture mode is selected automatically (`fd` on Linux/Kaggle and `sys`
on Windows).

To train the approach in the paper, run this command:

```train
python main.py
```

You can select the training task between matrix game, predator and prey, MATE, and SMAC by setting ```--env-config='matrix_game_3'
 or 'pred_prey_punish' or 'mate' or 'sc2'```.
Also you can select the training algorithm by setting ```--config='dual_iql_ree' or 'belief_dual_iql_ree' or 'iql' or 'hysteretic_q'```

Here ```dual_iql_ree``` refers to ```RAC``` in the submitted paper.

`belief_dual_iql_ree` is the uncertainty-aware extension. It causally infers a
categorical belief over local dynamics contexts from the current observation and
the previous action/reward. Action values combine posterior plausibility with
RAC's cooperative optimism:

```text
Q_dec = (1 - alpha_t) * E_{c~b_t}[Q_c] + alpha_t * max_c Q_c
```

`alpha_t` increases with normalized context entropy and context shift, and is
bounded below by `belief_optimism_min > 0`. Consequently, posterior averaging
never removes the optimistic component; a uniform belief with
`belief_optimism_max: 1.0` exactly recovers RAC's maximum over contexts.

Run it with:

```train
python main.py --env-config mate --config belief_dual_iql_ree --max-steps 2000000
```

The implementation logs `context_uncertainty_mean`, `context_shift_mean`,
`context_optimism_weight`, `context_posterior_q_mean`,
`context_optimistic_q_mean`, `context_optimism_gap`, reconstruction/KL losses,
and per-context posterior mass.

Set the maximum number of training environment steps with `--max-steps`:

```train
python main.py --env-config mate --config dual_iql_ree --max-steps 2000000
```

### Weights & Biases

W&B logging is optional and disabled by default. Install/authenticate W&B, then
enable it through Sacred's `with` overrides:

```bash
pip install "wandb>=0.22.3"
wandb login
python main.py --env-config mate --config dual_iql_ree --max-steps 500000 \
  with use_wandb=True wandb_project="rac-mate" use_tensorboard=False
```

Every scalar already sent to Sacred/CSV (returns, coverage, losses, epsilon,
and return-slot occupancy) is sent to W&B with `t_env` as its x-axis. Optional
settings include `wandb_entity`, `wandb_group`, `wandb_run_name`, `wandb_tags`,
and `wandb_mode`. Use `wandb_mode="offline"` when internet access is unavailable.

### Running MATE training on Kaggle

1. Create a Kaggle Notebook, enable a GPU accelerator and internet access, and
   attach this repository as a Dataset (or clone it from GitHub). Copy attached
   code from `/kaggle/input` to the writable `/kaggle/working` directory:

   ```python
   !cp -r /kaggle/input/<dataset-name>/RAC_code /kaggle/working/RAC_code
   %cd /kaggle/working/RAC_code
   ```

   For a GitHub repository, use this instead:

   ```python
   !git clone <repository-url> /kaggle/working/RAC_code
   %cd /kaggle/working/RAC_code
   ```

2. Install the Kaggle-specific dependencies. This uses a known-compatible
   PyTorch 2.5.1 CUDA 12.1 wheel instead of the obsolete torch pin from the
   original `requirements.txt` or a runtime wheel missing older GPU kernels:

   ```python
   %pip install -q -r requirements-kaggle.txt
   ```

3. Add `WANDB_API_KEY` under **Add-ons -> Secrets** in the notebook, enable its
   notebook permission, and authenticate without printing the key:

   ```python
   import os
   from kaggle_secrets import UserSecretsClient

   os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
   ```

4. Check CUDA and start training from `src`:

   ```python
   import torch
   print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))

   %cd /kaggle/working/RAC_code/src
   !python main.py --env-config mate --config dual_iql_ree --max-steps 500000 \
     with use_wandb=True wandb_project="rac-mate" use_tensorboard=False
   ```

   For a no-network run, omit the secret and add `wandb_mode="offline"`. W&B's
   local run files and all model/CSV outputs are written beneath
   `/kaggle/working/RAC_code/src/results`.

## Experiment metrics and plots

Every run writes scalar metrics in long-form CSV format under:

```text
results/metrics/<env>/<scenario>/<algorithm>/<run_id>/metrics.csv
```

Each row contains the run ID, algorithm, environment, scenario, seed, metric,
training environment step, and scalar value. CSV logging is independent of
TensorBoard and is flushed continuously, so completed data remains usable after
an interrupted or failed run.

The run automatically updates aggregate plots for test return and, when
available, test coverage. To create another comparison plot across runs, seeds,
and algorithms:

```powershell
python plot_metrics.py --root results/metrics --env mate --scenario MATE-4v8-9.yaml --metric test_coverage_rate_mean --output results/plots/mate_coverage.png
```

For coverage metrics, the CSV stores values in `[0, 1]` while plots display
percentages. With multiple seeds for the same algorithm, the bold curve is the
mean and the shaded region is one standard deviation; faint lines are individual runs.

## Hyper-parameters

To modify the hyper-parameters of algorithms and environments, refer to:

```
src/config/algs/dual_iql_ree.yaml
src/config/algs/belief_dual_iql_ree.yaml
src/config/default.yaml
```
```
src/config/envs/matrix_game_3.yaml
src/config/envs/pred_prey_punish.yaml
src/config/envs/mate.yaml
src/config/envs/sc2.yaml
```

## Note

This repository is developed based on PyMARL. And we have cited the SMAC paper in our work.
