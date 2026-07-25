import os
import sys
import unittest
from os.path import abspath, dirname

import numpy as np

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from envs.vmas_transport import VMASTransportEnv


class VMASTransportEnvTest(unittest.TestCase):
    def test_task_termination_takes_precedence_over_truncation(self):
        cases = (
            (False, False, False),
            (False, True, True),
            (True, False, False),
            (True, True, False),
        )
        for terminated, truncated, expected in cases:
            with self.subTest(terminated=terminated, truncated=truncated):
                self.assertEqual(
                    VMASTransportEnv._is_pure_truncation(terminated, truncated),
                    expected,
                )

    def test_three_agent_transport_contract(self):
        env = VMASTransportEnv({"n_agents": 3, "episode_limit": 1, "seed": 0})

        env_info = env.get_env_info()
        self.assertEqual(env_info["n_agents"], 3)
        self.assertEqual(env_info["n_actions"], 2)
        self.assertEqual(env_info["state_shape"], 33)
        self.assertEqual([obs.shape for obs in env.get_obs()], [(11,), (11,), (11,)])

        reward, terminated, info = env.step(np.zeros((3, 2), dtype=np.float32))

        self.assertIsInstance(reward, float)
        self.assertTrue(terminated)
        self.assertEqual(
            set(info),
            {
                "episode_limit",
                "success_rate",
                "reward/package_progress",
                "reward/agent_approach",
                "contact_rate",
                "package_displacement",
                "zero_return",
                "zero_package_progress",
            },
        )
        self.assertTrue(info["episode_limit"])
        self.assertEqual(info["success_rate"], 0.0)
        self.assertEqual(info["reward/agent_approach"], 0.0)
        self.assertEqual(info["package_displacement"], 0.0)
        self.assertEqual(info["zero_return"], 1.0)
        self.assertEqual(info["zero_package_progress"], 1.0)


if __name__ == "__main__":
    unittest.main()
