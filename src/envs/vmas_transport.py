import numpy as np
import torch as th
from gymnasium import spaces
import vmas


class VMASTransportEnv:
    """Expose VMAS transport through the runner's single-environment API."""

    def __init__(self, config):
        self.n_agents = int(config["n_agents"])
        self.episode_limit = int(config["episode_limit"])
        self.env = vmas.make_env(
            scenario="transport",
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            max_steps=self.episode_limit,
            seed=config.get("seed"),
            n_agents=self.n_agents,
            n_packages=int(config.get("n_packages", 1)),
            package_width=float(config.get("package_width", 0.15)),
            package_length=float(config.get("package_length", 0.15)),
            package_mass=float(config.get("package_mass", 50)),
        )
        action_shape = self.env.agents[0].action_size
        action_low = -np.ones(action_shape, dtype=np.float32)
        action_high = np.ones(action_shape, dtype=np.float32)
        self.action_spaces = [
            spaces.Box(low=action_low, high=action_high, dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self._obs = None
        self.reset(seed=config.get("seed"))

    def get_env_info(self):
        obs_shape = self._obs[0].shape[0]
        return {
            "n_agents": self.n_agents,
            "n_actions": self.action_spaces[0].shape[0],
            "obs_shape": obs_shape,
            "state_shape": obs_shape * self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_spaces,
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }

    def reset(self, seed=None):
        observations = self.env.reset(seed=seed)
        self._obs = [
            observation[0].cpu().numpy().astype(np.float32)
            for observation in observations
        ]
        return self._obs

    def reset_test(self, seed=None):
        return self.reset(seed=seed)

    def get_obs(self):
        return self._obs

    def get_state(self):
        return np.concatenate(self._obs)

    def step(self, actions):
        actions = th.as_tensor(actions, dtype=th.float32).view(
            self.n_agents, self.action_spaces[0].shape[0]
        )
        observations, rewards, done, infos = self.env.step(
            [action.unsqueeze(0) for action in actions]
        )
        self._obs = [
            observation[0].cpu().numpy().astype(np.float32)
            for observation in observations
        ]
        reward_values = th.stack(rewards)
        if not th.allclose(reward_values, reward_values[0]):
            raise ValueError("VMAS transport must provide one shared reward for all agents.")
        return float(reward_values[0, 0]), bool(done[0]), {}

    def close(self):
        pass
