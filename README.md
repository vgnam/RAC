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

Before running, you should set ```SETTINGS['CAPTURE_MODE'] = 'fd'``` for linux or ```SETTINGS['CAPTURE_MODE'] = 'sys'``` for windows.

To train the approach in the paper, run this command:

```train
python main.py
```

You can select the training task between matrix game, predator and prey, MATE, and SMAC by setting ```--env-config='matrix_game_3'
 or 'pred_prey_punish' or 'mate' or 'sc2'```.
Also you can select the training algorithm by setting ```--config='dual_iql_ree' or 'iql' or 'hysteretic_q'```

Here ```dual_iql_ree``` refers to ```RAC``` in the submitted paper.

## Hyper-parameters

To modify the hyper-parameters of algorithms and environments, refer to:

```
src/config/algs/dual_iql_ree.yaml
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