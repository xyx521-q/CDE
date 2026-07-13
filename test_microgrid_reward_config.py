import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")

from envs.maenv import MGEnv


class MicrogridRewardConfigTest(unittest.TestCase):
    def _load_env_args(self):
        config_path = Path("src/config/envs/microgrid.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            return copy.deepcopy(yaml.safe_load(f)["env_args"])

    def _make_reward_env(self):
        env_args = self._load_env_args()
        env_args["log_trajectories"] = False
        return MGEnv(config=env_args, algo_name="reward_config_test")

    def test_config_uses_milp_scaled_controllable_resources(self):
        env_args = self._load_env_args()

        for agent in env_args["data"]:
            self.assertEqual(agent["dg_max"], 80.0)
            self.assertEqual(agent["battery_cap"], 200.0)
            self.assertEqual(agent["low_action"][1], -1.0)
            self.assertEqual(agent["high_action"][1], 1.0)
            self.assertAlmostEqual(agent["split_ratio"], 1.0 / 3.0, places=12)

        self.assertAlmostEqual(sum(agent["split_ratio"] for agent in env_args["data"]), 1.0, places=12)
        self.assertEqual(env_args["reward_mode"], "milp_economic")
        self.assertEqual(env_args["buy_reward_mode"], "raw_cost")
        self.assertEqual(env_args["reward_scale"], 1.0)
        self.assertEqual(env_args["generate_cost_weight"], 1.0)
        self.assertEqual(env_args["bess_cost_weight"], 1.0)
        self.assertEqual(env_args["env_reward_weight"], 1.0)
        self.assertEqual(env_args["curtailment_penalty_weight"], 1.0)

    def test_env_info_exposes_cost_metrics_for_tensorboard_logging(self):
        env = self._make_reward_env()
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
            "economic_reward",
            "economic_cost",
            "curtailment_cost",
        ]:
            self.assertIn(key, info)

    def test_milp_economic_reward_prefers_lower_total_cost(self):
        env = self._make_reward_env()
        env.reset_test(seed=0)
        env.idx = [0, 0]
        env.params["socs"] = np.array([0.5] * env.n_agents, dtype=np.float32)
        env.data_dict["load_pv"][0, 0] = 240.0
        env.data_dict["price"][0, 0] = 10.0

        low_actions = np.array([[0.0, 0.0]] * env.n_agents, dtype=np.float32)
        high_actions = np.array([[1.0, 0.0]] * env.n_agents, dtype=np.float32)

        _, _, low_info = env.step(low_actions)

        env = self._make_reward_env()
        env.reset_test(seed=0)
        env.idx = [0, 0]
        env.params["socs"] = np.array([0.5] * env.n_agents, dtype=np.float32)
        env.data_dict["load_pv"][0, 0] = 240.0
        env.data_dict["price"][0, 0] = 10.0
        _, _, high_info = env.step(high_actions)

        self.assertGreater(high_info["economic_reward"], low_info["economic_reward"])
        self.assertLess(high_info["economic_cost"], low_info["economic_cost"])
        self.assertAlmostEqual(high_info["curtailment_cost"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
