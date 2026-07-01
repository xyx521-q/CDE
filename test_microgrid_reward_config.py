import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")

from envs.maenv import MGEnv


class MicrogridRewardConfigTest(unittest.TestCase):
    def test_config_uses_milp_scaled_controllable_resources(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            env_args = yaml.safe_load(f)["env_args"]

        for agent in env_args["data"]:
            self.assertEqual(agent["dg_max"], 80.0)
            self.assertEqual(agent["battery_cap"], 200.0)
            self.assertEqual(agent["low_action"][1], -1.0)
            self.assertEqual(agent["high_action"][1], 1.0)

        self.assertEqual(env_args["buy_reward_mode"], "relative_saving")
        self.assertEqual(env_args["buy_reward_scale"], 100.0)
        self.assertEqual(env_args["battery_action_mode"], "soc_feasible_fraction")
        self.assertEqual(env_args["battery_c_rate_limit"], 0.3)
        self.assertEqual(env_args["generate_cost_weight"], 0.1)
        self.assertEqual(env_args["bess_cost_weight"], 0.1)
        self.assertEqual(env_args["env_reward_weight"], 0.1)

    def test_env_info_exposes_cost_metrics_for_tensorboard_logging(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            env_args = yaml.safe_load(f)["env_args"]
        env_args = copy.deepcopy(env_args)
        env_args["log_trajectories"] = False
        env_args["battery_action_mode"] = "fixed_power"

        env = MGEnv(config=env_args, algo_name="reward_config_test")
        env.reset_test(seed=0)
        terminated = False
        info = {}
        while not terminated:
            actions = np.array([[0.5, 0.0]] * env.n_agents, dtype=np.float32)
            _, terminated, info = env.step(actions)

        for key in [
            "grid_purchase_cost",
            "baseline_grid_purchase_cost",
            "grid_purchase_saving",
            "grid_purchase_saving_ratio",
            "mean_soc",
            "battery_action_clip",
        ]:
            self.assertIn(key, info)

    def test_soc_reserve_reward_penalizes_upper_bound_violation(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            env_args = yaml.safe_load(f)["env_args"]
        env_args = copy.deepcopy(env_args)
        env_args["log_trajectories"] = False

        env = MGEnv(config=env_args, algo_name="reward_config_test")
        env.reset_test(seed=0)
        env.params["socs"] = np.array([0.79] * env.n_agents, dtype=np.float32)

        actions = np.array([[0.0, -1.0]] * env.n_agents, dtype=np.float32)
        _, _, info = env.step(actions)

        self.assertLess(info["soc_reserve_reward"], 0.0)

    def test_soc_feasible_battery_action_avoids_low_soc_clip(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            env_args = yaml.safe_load(f)["env_args"]
        env_args = copy.deepcopy(env_args)
        env_args["log_trajectories"] = False

        env = MGEnv(config=env_args, algo_name="reward_config_test")
        env.reset_test(seed=0)
        env.params["socs"] = np.array([0.21] * env.n_agents, dtype=np.float32)

        actions = np.array([[0.0, 1.0]] * env.n_agents, dtype=np.float32)
        _, _, info = env.step(actions)

        self.assertAlmostEqual(info["battery_action_clip"], 0.0, places=6)
        self.assertAlmostEqual(info["soc_violation"], 0.0, places=6)
        self.assertTrue(np.all(env.params["socs"] >= 0.2))

    def test_soc_feasible_battery_action_avoids_high_soc_clip(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            env_args = yaml.safe_load(f)["env_args"]
        env_args = copy.deepcopy(env_args)
        env_args["log_trajectories"] = False

        env = MGEnv(config=env_args, algo_name="reward_config_test")
        env.reset_test(seed=0)
        env.params["socs"] = np.array([0.79] * env.n_agents, dtype=np.float32)

        actions = np.array([[0.0, -1.0]] * env.n_agents, dtype=np.float32)
        _, _, info = env.step(actions)

        self.assertAlmostEqual(info["battery_action_clip"], 0.0, places=6)
        self.assertAlmostEqual(info["soc_violation"], 0.0, places=6)
        self.assertTrue(np.all(env.params["socs"] <= 0.8))

    def test_unknown_battery_action_mode_fails_fast(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            env_args = yaml.safe_load(f)["env_args"]
        env_args = copy.deepcopy(env_args)
        env_args["battery_action_mode"] = "typo"

        with self.assertRaises(ValueError):
            MGEnv(config=env_args, algo_name="reward_config_test")


if __name__ == "__main__":
    unittest.main()
