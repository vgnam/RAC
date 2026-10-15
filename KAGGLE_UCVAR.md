# Run UCVaR-IQL on Kaggle

UCVaR-IQL is IQL whose target is the mean of the top `ucvar_alpha` tail of the
TD target (`src/learners/ucvar_q_learner.py`). `ucvar_alpha=1` is IQL.

Enable a GPU (and Internet) in the notebook, add the ZIP as a dataset, then:

```python
from pathlib import Path
import subprocess, sys, zipfile

archive = next(Path('/kaggle/input').rglob('RAC_code_ucvar_iql_*.zip'))
with zipfile.ZipFile(archive) as z:
    z.extractall('/kaggle/working')
root = Path('/kaggle/working/RAC_code')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(root / 'requirements-kaggle.txt')], check=True)
# LBF only (not in requirements-kaggle.txt):
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'lbforaging==2.0.0', 'gymnasium>=1.0'], check=True)
```

Optional online W&B logging (otherwise runs log offline):

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')
```

## Commands

```python
# LBF 5x5, 2M steps (one run per alpha, sequential)
subprocess.run([sys.executable, str(root / 'kaggle_run_ucvar.py'), '--env', 'lbf',
                '--alphas', '0.5', '0.3', '0.1', '--seeds', '0'], cwd=root, check=True)

# Predator-Prey (pred_prey_punish), 1M steps
subprocess.run([sys.executable, str(root / 'kaggle_run_ucvar.py'), '--env', 'pred_prey',
                '--alphas', '0.5', '0.3', '0.1', '--seeds', '0'], cwd=root, check=True)
```

Lenient schedule (Predator-Prey only): alpha is per (observation, action) and rises from
the `--alphas` value to 1 as the pair is visited, `alpha(n) = 1 - (1 - start) exp(-n / k)`
(`--count-scale` = k, visits of the SimHash bucket of the observation, all agents):

```python
subprocess.run([sys.executable, str(root / 'kaggle_run_ucvar.py'), '--env', 'pred_prey',
                '--lenient', '--alphas', '0.05', '--count-scale', '1000', '--seeds', '0'],
               cwd=root, check=True)
```

Run names look like `LUCVaR0.05-k1000-predprey-1M-eps0.0-seed0`.

Options: `--t-max`, `--wandb-project` (default `rac-marl`), `--tremble-eps`,
`--lbf-key` (default `Foraging-5x5-2p-1f-coop-v3`; set it to the key of the IQL
runs you compare against). Run names look like `UCVaR0.3-lbf5x5-2M-eps0.0-seed0`.

## What to watch in W&B

| Stat | Meaning |
|---|---|
| `test_return_mean` | Performance |
| `q_minus_return` | Q of taken actions minus the discounted return-to-go of the sampled episodes: the overestimation (positive and growing = optimism drift) |
| `tail_frac` | Fraction of targets above eta; near `ucvar_alpha` for continuous targets, jumpy when sparse rewards put an atom at the quantile |
| `target_mean`, `cvar_target_mean`, `eta_mean` | Mean TD target, CVaR target and quantile |
| `alpha_mean`, `alpha_low_frac` | Lenient only: mean alpha of the sampled pairs and fraction below 0.5 |
| `alpha_action5_mean` | Lenient only: mean alpha of 'catch' (stays low while catching is rarely tried) |
| `hash_cells_visited` | Lenient only: occupied (SimHash bucket, action) cells |

Unlike `q_learner` / `hysteretic_q_learner`, this learner's `q_taken_mean` and
`target_mean` are true per-agent means (not divided by `n_agents` twice).
`return_to_go_mean` is truncated at the episode end, so `q_minus_return` is slightly
positive even for an exact Q near the time limit.
