# Run this ZIP on Kaggle

The archive contains `RAC_code/` and the modified DM2 checkout as ordinary files
inside `RAC_code/dm2/`. No Windows junction is needed after extraction.

Enable a GPU and Internet in the notebook settings. Add the ZIP as a dataset.
Extract to the writable working directory:

```python
from pathlib import Path
import zipfile

archive = next(Path('/kaggle/input').rglob('RAC_DM2_Kaggle_20260911.zip'))
with zipfile.ZipFile(archive) as z:
    z.extractall('/kaggle/working')
root = Path('/kaggle/working/RAC_code')
```

Install the Kaggle requirements before importing torch/numpy/mate in the kernel:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(root / 'requirements-kaggle.txt')], check=True)
```

For online W&B logging, add the `WANDB_API_KEY` secret in Kaggle, enable access,
and run this before training:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')
```

Without `WANDB_API_KEY`, the included launcher uses W&B offline mode.

## Belief-RAC command matching the supplied experiment

```python
command = [sys.executable, str(root / 'kaggle_run_belief.py')]
subprocess.run(command, cwd=root, check=True)
```

`kaggle_run_belief.py` contains the full editable `command` list and shared
arguments. It preserves the supplied belief settings, including
`belief_prior_strength=0.0`. Because `shared_args` was not supplied, the run-name
values are made explicit: `seed=2`, `buffer_size=500`, `kl_weight=0.5` and
`target_update_interval_steps=10000`. Other shared settings follow the selected
RAC preset. `kl_weight` and `belief_kl_weight=0.0001` are different losses.
This command trains Belief-RAC, not DM2.

## Optional DM2 baseline

The additional packages below are only needed when launching DM2; the main
Kaggle requirements above already install PyTorch and MATE.

```python
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'stable-baselines3==2.5.0', 'h5py>=3.8,<4',
                'einops>=0.6,<1', 'cloudpickle>=2,<4'], check=True)

command = [
    sys.executable, str(root / 'run_dm2.py'),
    '--env-config=mate',       # or pred_prey_punish
    '--mode=dm2_sil',          # or ippo; dm2 requires expert demonstrations
    '--max-steps', '500000',
    '--seed', '2',
    'with',
    'use_cuda=True',
]
subprocess.run(command, cwd=root, check=True)
```

RAC's belief flags and `shared_args` do not apply to the separate DM2 learner.
See `docs/DM2.md` for the DM2 integration details. Training outputs stay under
`/kaggle/working/RAC_code/results` (DM2 Sacred metadata is in `dm2/results`).

The ZIP excludes local virtual environments, Git metadata, old experiment
outputs and bytecode. Functional tests were run locally; this package has not
been executed on Kaggle's GPU runtime in this session.
