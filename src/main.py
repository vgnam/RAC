import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import collections
import collections.abc
import torch as th
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import yaml
from src.utils.logging import get_logger


from src.run import run
# 'fd' for linux, 'sys' for windows
SETTINGS['CAPTURE_MODE'] = "sys" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def _get_config_from_argparse(args, arg_name, subfolder):
    config_name = None
    if arg_name == "--env-config":
        config_name = args.env_config
    elif arg_name == "--config":
        config_name = args.config

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='the manual setting of params')
    parser.add_argument("--env-config", type=str, default='matrix_game_3', choices=['matrix_game_3', 'pred_prey_punish', 'sc2', 'mate'])
    parser.add_argument("--config", type=str, default='dual_iql_ree',
                        choices=['dual_iql_ree', 'iql', 'hysteretic_q'])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of environment steps used for training (maps to t_max).",
    )
    args, _ = parser.parse_known_args()

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")

    params = deepcopy(sys.argv)

    # Remove the argparse-only flags (and their values) so that sacred's
    # command-line parser does not reject them.
    filtered_params = [params[0]]
    skip_next = False
    for p in params[1:]:
        if skip_next:
            skip_next = False
            continue
        if p in ("--env-config", "--config", "--max-steps") \
                or p.startswith("--env-config=") \
                or p.startswith("--config=") \
                or p.startswith("--max-steps="):
            if "=" not in p:
                skip_next = True
            continue
        filtered_params.append(p)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    # env_config = _get_config(params, "--env-config", "envs")
    # alg_config = _get_config(params, "--config", "algs")
    env_config = _get_config_from_argparse(args, "--env-config", "envs")
    alg_config = _get_config_from_argparse(args, "--config", "algs")
    # Update original params loaded from default.yaml
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    if args.max_steps is not None:
        config_dict["t_max"] = args.max_steps
    # config_dict = {**config_dict, **env_config, **alg_config}

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    algo_name = parse_command(params, "name", config_dict['name'])

    if config_dict["env"] == "sc2" or config_dict["env"] == "mpe":
        map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
        file_obs_path = os.path.join(results_path, "sacred", config_dict["env"], map_name, algo_name)
    else:
        file_obs_path = os.path.join(results_path, "sacred", config_dict["env"], algo_name)

    logger.info("Saving to FileStorageObserver in {}".format(file_obs_path))
    ex.observers.append(FileStorageObserver(file_obs_path))

    # th.backends.cudnn.benchmark = False
    # th.backends.cudnn.deterministic = True

    ex.run_commandline(filtered_params)

    # flush
    sys.stdout.flush()
