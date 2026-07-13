"""Profile one fixed-length FACMAC run without changing production code."""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from os.path import abspath, dirname

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from components.episode_buffer import EpisodeBatch
from controllers.cqmix_controller import CQMixMAC
from direct_main import config_copy, load_yaml, normalize_scalars, recursive_dict_update
from ea.ssne_base import SSNEBase
from envs.maenv import MGEnv
from learners.facmac_learner import FACMACLearner
from run import run
from runners.episode_runner import EpisodeRunner
from utils.logging import get_logger


class Timer:
    def __init__(self):
        self.calls = defaultdict(int)
        self.seconds = defaultdict(float)

    def record(self, name, elapsed):
        self.calls[name] += 1
        self.seconds[name] += elapsed

    def snapshot(self):
        return dict(self.calls), dict(self.seconds)

    def delta(self, snapshot, name):
        calls, seconds = snapshot
        return self.calls[name] - calls.get(name, 0), self.seconds[name] - seconds.get(
            name, 0.0
        )


class CsvTimingLog:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def write(self, kind, seconds, t_env="", episode="", step=""):
        self.rows.append(
            {
                "kind": kind,
                "t_env": t_env,
                "episode": episode,
                "step": step,
                "seconds": f"{seconds:.9f}",
            }
        )

    def close(self):
        with open(self.path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("kind", "t_env", "episode", "step", "seconds"),
            )
            writer.writeheader()
            writer.writerows(self.rows)


def timed_method(cls, method_name, timer, metric_name, after=None):
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        started = time.perf_counter()
        result = original(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        timer.record(metric_name, elapsed)
        if after is not None:
            after(self, args, kwargs, elapsed)
        return result

    setattr(cls, method_name, wrapped)


def load_config(config_name):
    config_dir = os.path.join(PROJECT_ROOT, "src", "config")
    config = load_yaml(os.path.join(config_dir, "default.yaml"))
    config = recursive_dict_update(
        config, load_yaml(os.path.join(config_dir, "envs", "microgrid.yaml"))
    )
    config = recursive_dict_update(
        config, load_yaml(os.path.join(config_dir, "algs", f"{config_name}.yaml"))
    )
    return normalize_scalars(config_copy(config))


def main():
    parser = argparse.ArgumentParser(description="Profile FACMAC training bottlenecks.")
    parser.add_argument("--config", default="facmac_ea")
    parser.add_argument("--timesteps", type=int, default=12144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/training_timing.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    episode_limit = MGEnv(deepcopy(config["env_args"]), config["name"]).episode_limit
    if args.timesteps % episode_limit:
        raise ValueError(
            f"--timesteps must be divisible by the {episode_limit}-step episode limit."
        )

    # run.py uses <=, so stop one episode early to end at exactly --timesteps.
    config.update(
        {
            "seed": args.seed,
            "use_cuda": False,
            "wandb_mode": "disabled",
            "save_model": False,
            "test_nepisode": 1,
            "t_max": args.timesteps - episode_limit,
            # Keep the profiler output limited to its CSV and final summary.
            "test_interval": args.timesteps + episode_limit,
            "log_interval": args.timesteps + episode_limit,
            "runner_log_interval": args.timesteps + episode_limit,
            "learner_log_interval": args.timesteps + episode_limit,
        }
    )
    config["env_args"]["seed"] = args.seed
    config["env_args"]["run_id"] = "timing_profile"

    timer = Timer()
    csv_log = CsvTimingLog(args.output)
    run_started = time.perf_counter()

    def report_episode(runner, method_args, method_kwargs, elapsed):
        test_mode = method_kwargs.get("test_mode", False)
        if not test_mode:
            csv_log.write("episode_rollout", elapsed, runner.t_env, step=runner.t)

    def report_update(learner, method_args, method_kwargs, elapsed):
        csv_log.write("learner_update", elapsed, method_args[1], method_args[2])

    def report_environment_step(env, method_args, method_kwargs, elapsed):
        csv_log.write("environment_step", elapsed, step=env.idx[1])

    def report_evolution(evolver, method_args, method_kwargs, elapsed):
        csv_log.write("evolution", elapsed)

    def report_action_selection(mac, method_args, method_kwargs, elapsed):
        csv_log.write("action_selection", elapsed)

    def report_batch_update(batch, method_args, method_kwargs, elapsed):
        csv_log.write("batch_update", elapsed)

    timed_method(
        MGEnv, "step", timer, "environment_step", report_environment_step
    )
    timed_method(
        CQMixMAC,
        "select_actions",
        timer,
        "action_selection",
        report_action_selection,
    )
    timed_method(EpisodeBatch, "update", timer, "batch_update", report_batch_update)
    timed_method(EpisodeRunner, "run", timer, "episode_rollout", report_episode)
    timed_method(FACMACLearner, "train", timer, "learner_update", report_update)
    timed_method(SSNEBase, "epoch", timer, "evolution", report_evolution)

    logger = get_logger()
    logger.setLevel("WARNING")
    try:
        run(config, logger)
    finally:
        csv_log.close()

    total = time.perf_counter() - run_started
    print("\nTiming summary")
    print(f"wall_clock: {total:.3f}s")
    print(f"detail_csv: {args.output}")
    for name in (
        "episode_rollout",
        "learner_update",
        "evolution",
        "action_selection",
        "environment_step",
        "batch_update",
    ):
        calls = timer.calls[name]
        seconds = timer.seconds[name]
        share = 100 * seconds / total if total else 0.0
        average = seconds / calls if calls else 0.0
        print(
            f"{name:20} {seconds:8.3f}s {share:5.1f}% "
            f"calls={calls:5d} avg={average:.5f}s"
        )


if __name__ == "__main__":
    main()
