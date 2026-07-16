import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger
        self.stats = defaultdict(lambda: [])
        self.wandb_run = None
        self.pending_metrics = {}
        self.pending_step = None

    def setup_wandb(self, config, name, project, entity, mode, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            mode=mode,
            dir=directory,
            settings=wandb.Settings(console="off"),
        )

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        if self.pending_step is not None and t != self.pending_step:
            self.flush()
        self.pending_step = t
        self.pending_metrics[key] = value.item() if hasattr(value, "item") else value

    def flush(self):
        if self.pending_metrics:
            self.wandb_run.log(self.pending_metrics, step=self.pending_step)
            self.pending_metrics = {}
            self.pending_step = None

    def finish(self):
        self.flush()
        self.wandb_run.finish()

    def print_recent_stats(self, t_env):
        log_str = f"Recent Stats | t_env: {t_env:>10}\n"
        i = 0
        for k, v in sorted(self.stats.items()):
            i += 1
            values = []
            for _, raw_value in self.stats[k][-5:]:
                if isinstance(raw_value, (int, float)):
                    values.append(raw_value)
                elif hasattr(raw_value, "item"):
                    values.append(raw_value.item())
                else:
                    values.append(raw_value)
            item = f"{np.mean(values):.4f}"
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")
    return logger
