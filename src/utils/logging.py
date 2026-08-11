from collections import defaultdict
import csv
import logging
import os
import numpy as np
import torch as th


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_csv = False
        self.use_hdf = False
        self.use_wandb = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        # from tensorboard_logger import configure, log_value
        # configure(directory_name)
        # self.tb_logger = log_value
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(logdir=directory_name)
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_wandb(
        self,
        config,
        project,
        run_name,
        directory,
        entity=None,
        group=None,
        tags=None,
        mode="online",
    ):
        # Import lazily so W&B remains optional when use_wandb is false.
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "use_wandb=True requires the 'wandb' package. "
                "Install it with: pip install wandb>=0.22.3"
            ) from exc

        os.makedirs(directory, exist_ok=True)
        self.wandb = wandb
        self.wandb_run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name,
            group=group or None,
            tags=list(tags or []),
            config=config,
            dir=directory,
            mode=mode,
            job_type=str(config.get("name", "train")),
        )
        # W&B has its own internal row counter. All charts use the actual MARL
        # environment step instead, even when several metrics share one t_env.
        self.wandb_run.define_metric("t_env")
        self.wandb_run.define_metric("*", step_metric="t_env")
        self.use_wandb = True
        self.console_logger.info(
            "Logging metrics to Weights & Biases project %s (mode=%s)",
            project,
            mode,
        )

    def setup_csv(self, file_path, metadata):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.csv_file = open(file_path, "a", newline="", encoding="utf-8")
        self.csv_fields = [
            "run_id", "algorithm", "env", "scenario", "seed",
            "metric", "t_env", "value",
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
        if self.csv_file.tell() == 0:
            self.csv_writer.writeheader()
        self.csv_metadata = dict(metadata)
        self.use_csv = True
        self.console_logger.info("Logging scalar metrics to %s", file_path)

    @staticmethod
    def _scalar(value):
        if th.is_tensor(value):
            return value.detach().cpu().item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def log_stat(self, key, value, t, to_sacred=True):
        value = self._scalar(value)
        self.stats[key].append((t, value))

        if self.use_tb:
            self.writer.add_scalar(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

        if self.use_csv:
            row = dict(self.csv_metadata)
            row.update({"metric": key, "t_env": int(t), "value": value})
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        if self.use_wandb:
            self.wandb_run.log({"t_env": int(t), key: value})

    def log_histogram(self, key, value, t):
        if self.use_tb:
            self.writer.add_histogram(key, value, t)
        if self.use_wandb:
            if th.is_tensor(value):
                value = value.detach().cpu().numpy()
            self.wandb_run.log({
                "t_env": int(t),
                key: self.wandb.Histogram(value),
            })

    def close(self):
        if self.use_tb:
            self.writer.flush()
            self.writer.close()
            self.use_tb = False
        if self.use_csv:
            self.csv_file.flush()
            self.csv_file.close()
            self.use_csv = False
        if self.use_wandb:
            self.wandb_run.finish()
            self.use_wandb = False

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            mean_value = th.mean(th.tensor([x[1] for x in self.stats[k][-window:]])).item()
            if "coverage_rate" in k:
                item = "{:.4f}%".format(100.0 * mean_value)
            else:
                item = "{:.4f}".format(mean_value)
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
