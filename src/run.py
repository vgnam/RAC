import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from src.utils.logging import Logger
from src.utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from src.learners import REGISTRY as le_REGISTRY
from src.runners import REGISTRY as r_REGISTRY
from src.controllers import REGISTRY as mac_REGISTRY
from src.components.episode_buffer import ReplayBuffer
from src.components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda:{}".format(args.device) if args.use_cuda else "cpu"
    print("Device", args.device)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # Configure tensorboard logger
    if 'iql_ree' == args.name:
        unique_token = "{}_seed_{}_use_mask_{}_bonus_w_{}_slot_number_{}_tau_{}_rnd_rep_size_{}_{}".format(args.name,
                                                                                                           args.seed,
                                                                                                           args.use_mask,
                                                                                                           args.bonus_weight,
                                                                                                           args.slot_number,
                                                                                                           args.rnd_tau,
                                                                                                           args.rnd_rep_size,
                                                                                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif 'new_iql_ree' == args.name:
        unique_token = "{}_seed_{}_slot_number_{}_{}".format(args.name,
                                                             args.seed,
                                                             args.slot_number,
                                                             datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif 'dual_iql_ree' == args.name:
        unique_token = "{}_seed_{}_slot_number_{}_kl_weight_{}_twin_agent_{}_{}".format(args.name,
                                                                                        args.seed,
                                                                                        args.slot_number,
                                                                                        args.kl_weight,
                                                                                        args.twin_agent,
                                                                                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif 'hysteretic_q' in args.name:
        unique_token = "{}_seed_{}_beta_{}_{}".format(args.name, args.seed, args.beta, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        unique_token = "{}_seed_{}_{}".format(args.name, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    args.unique_token = unique_token

    if args.env == 'sc2':
        scenario = args.env_args['map_name']
    elif args.env == 'mate':
        scenario = args.env_args.get('mate_config', 'mate')
    else:
        scenario = args.env
    scenario_path = str(scenario).replace('/', '_').replace('\\', '_')

    metrics_root = os.path.join(
        args.local_results_path, "metrics", args.env, scenario_path, args.name
    )
    metrics_file = os.path.join(metrics_root, unique_token, "metrics.csv")
    logger.setup_csv(metrics_file, {
        "run_id": unique_token,
        "algorithm": args.name,
        "env": args.env,
        "scenario": scenario,
        "seed": args.seed,
    })

    if args.use_tensorboard:
        if args.env == 'sc2':
            tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env, args.env_args['map_name'], args.name)
        else:
            tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env, args.name)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        wandb_tags = list(getattr(args, "wandb_tags", []) or [])
        wandb_tags.extend([args.env, args.name, str(scenario)])
        logger.setup_wandb(
            config=_config,
            project=args.wandb_project,
            entity=getattr(args, "wandb_entity", None),
            group=getattr(args, "wandb_group", None),
            tags=wandb_tags,
            mode=args.wandb_mode,
            run_name=getattr(args, "wandb_run_name", None) or unique_token,
            directory=os.path.join(args.local_results_path, "wandb"),
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train. Always flush CSV/TensorBoard, including failed runs.
    try:
        run_sequential(args=args, logger=logger)
    finally:
        logger.close()
        try:
            from src.plot_metrics import plot_metric

            plot_root = os.path.join(args.local_results_path, "plots", args.env, scenario_path)
            for metric in ("test_return_mean", "test_coverage_rate_mean"):
                output_path = os.path.join(plot_root, "{}.png".format(metric))
                if plot_metric(
                    root=os.path.join(args.local_results_path, "metrics"),
                    metric=metric,
                    output=output_path,
                    env=args.env,
                    scenario=str(scenario),
                ):
                    _log.info("Updated aggregate plot %s", output_path)
        except Exception as exc:
            _log.warning("Could not update aggregate plots: %s", exc)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

def evaluate_sequential(args, runner):
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    for _ in range(n_test_runs):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "noise": {"vshape": (args.noise_dim,) if "maven" in args.name else (1, 0)}
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multi-agent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
        if "maven" in args.name:
            runner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    # For the fingerprints algo
    if "fingerprints" in args.name:
        mac.training_iter_num = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env < args.t_max:

        # Run for a whole episode at a time
        if "maven" not in args.name:
            with th.no_grad():
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
                # print('cur_buffer_size:', buffer.episodes_in_buffer)
        else:
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            for _ in range(args.training_iters):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)
                del episode_sample

                if "fingerprints" in args.name:
                    mac.training_iter_num += 1

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                test_batch = runner.run(test_mode=True)
            if 'matrix_game' in args.env:
                learner.show_matrix_info(test_batch, runner.t_env)
            # elif 'mmdp_game' in args.env:
            #     learner.show_mmdp_info(test_batch, runner.t_env)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            if args.env == "sc2":
                save_path = os.path.join(args.local_results_path, "models", args.env, args.env_args['map_name'], args.name, args.unique_token, str(runner.t_env))
            else:
                save_path = os.path.join(args.local_results_path, "models", args.env, args.name, args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # Finally save the latest models into disk.
    if args.env == "sc2":
        save_path = os.path.join(args.local_results_path, "models", args.env, args.env_args['map_name'], args.unique_token, str(runner.t_env))
    else:
        save_path = os.path.join(args.local_results_path, "models", args.env, args.unique_token, str(runner.t_env))
    os.makedirs(save_path, exist_ok=True)
    logger.console_logger.info("Finally saving models to {}".format(save_path))
    learner.save_models(save_path)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
