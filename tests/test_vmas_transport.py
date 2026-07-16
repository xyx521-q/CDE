import os
import sys
import unittest
from os.path import abspath, dirname

import numpy as np

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from envs.vmas_transport import VMASTransportEnv


class VMASTransportEnvTest(unittest.TestCase):
    def test_three_agent_transport_contract(self):
        env = VMASTransportEnv({"n_agents": 3, "episode_limit": 10, "seed": 0})

        env_info = env.get_env_info()
        self.assertEqual(env_info["n_agents"], 3)
        self.assertEqual(env_info["n_actions"], 2)
        self.assertEqual(env_info["state_shape"], 33)
        self.assertEqual([obs.shape for obs in env.get_obs()], [(11,), (11,), (11,)])

        reward, terminated, info = env.step(np.zeros((3, 2), dtype=np.float32))

        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
