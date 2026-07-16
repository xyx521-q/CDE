import argparse
import collections
import collections.abc
import os
import re
from copy import deepcopy
from os.path import abspath, dirname

import numpy as np
import torch as th
import yaml

from run import run
from utils.logging import get_logger

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence


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
    if isinstance(config, list):
        return [config_copy(v) for v in config]
    return deepcopy(config)


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


INT_PATTERN = re.compile(r"^-?\d+$")
FLOAT_PATTERN = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def normalize_scalars(value):
    if isinstance(value, dict):
        return {k: normalize_scalars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_scalars(v) for v in value]
    if isinstance(value, str):
        if INT_PATTERN.match(value):
            return int(value)
        if FLOAT_PATTERN.match(value):
            parsed = float(value)
            return int(parsed) if parsed.is_integer() else parsed
    return value


def main():
    parser = argparse.ArgumentParser(description="Run FACMAC training.")
    parser.add_argument("--config", default="facmac_ea")
    parser.add_argument("--env-config", default="microgrid")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config entries, e.g. --override t_max=3000000 --override use_cuda=False",
    )
    args = parser.parse_args()

    src_dir = dirname(abspath(__file__))
    config_dir = os.path.join(src_dir, "config")

    config = load_yaml(os.path.join(config_dir, "default.yaml"))
    config = recursive_dict_update(
        config, load_yaml(os.path.join(config_dir, "envs", f"{args.env_config}.yaml"))
    )
    config = recursive_dict_update(
        config, load_yaml(os.path.join(config_dir, "algs", f"{args.config}.yaml"))
    )

    for override in args.override:
        key, raw_value = override.split("=", 1)
        set_nested(config, key, yaml.safe_load(raw_value))

    config = normalize_scalars(config_copy(config))
    config.setdefault("seed", 0)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config.setdefault("env_args", {})
    config["env_args"]["seed"] = config["seed"]
    config["env_args"]["run_id"] = "direct"

    logger = get_logger()
    run(config, logger)


if __name__ == "__main__":
    main()
