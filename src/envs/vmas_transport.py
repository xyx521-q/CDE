import numpy as np
import torch as th
from gymnasium import spaces
import vmas


class VMASTransportEnv:
    """Expose VMAS transport through the runner's single-environment API."""

    def __init__(self, config):
        self.n_agents = int(config["n_agents"])
        self.episode_limit = int(config["episode_limit"])
        self.agent_approach_reward_weight = float(
            config.get("agent_approach_reward_weight", 1.0)
        )
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
        self._previous_mean_agent_package_distance = (
            self._mean_agent_package_distance()
        )
        self._initial_package_positions = self._package_positions().clone()
        self._episode_steps = 0
        self._episode_package_progress = 0.0
        self._episode_agent_approach = 0.0
        self._episode_return = 0.0
        self._episode_contact_steps = 0
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
        package_progress = float(reward_values[0, 0])
        mean_agent_package_distance = self._mean_agent_package_distance()
        agent_approach = self.agent_approach_reward_weight * (
            self._previous_mean_agent_package_distance - mean_agent_package_distance
        )
        self._previous_mean_agent_package_distance = mean_agent_package_distance

        self._episode_steps += 1
        self._episode_package_progress += package_progress
        self._episode_agent_approach += agent_approach
        self._episode_return += package_progress + agent_approach
        self._episode_contact_steps += int(self._has_agent_package_contact())

        package_displacement = th.linalg.vector_norm(
            self._package_positions() - self._initial_package_positions, dim=1
        ).mean()
        info = {
            # `done` is true only when every package reaches its goal.
            "success_rate": float(done[0]),
            "reward/package_progress": self._episode_package_progress,
            "reward/agent_approach": self._episode_agent_approach,
            "contact_rate": self._episode_contact_steps / self._episode_steps,
            "package_displacement": float(package_displacement),
            "zero_return": float(abs(self._episode_return) < 1e-8),
            "zero_package_progress": float(
                abs(self._episode_package_progress) < 1e-8
            ),
        }
        return package_progress + agent_approach, bool(done[0]), info

    def _package_positions(self):
        return th.stack(
            [package.state.pos[0] for package in self.env.scenario.packages]
        )

    def _agent_package_distances(self):
        agent_positions = th.stack(
            [agent.state.pos[0] for agent in self.env.world.agents]
        )
        return th.linalg.vector_norm(
            agent_positions.unsqueeze(1) - self._package_positions().unsqueeze(0),
            dim=2,
        )

    def _mean_agent_package_distance(self):
        return float(self._agent_package_distances().min(dim=1).values.mean())

    def _has_agent_package_contact(self):
        agent_radii = th.tensor(
            [agent.shape.circumscribed_radius() for agent in self.env.world.agents]
        )
        package_radii = th.tensor(
            [package.shape.circumscribed_radius() for package in self.env.scenario.packages]
        )
        return bool(
            (self._agent_package_distances() <= agent_radii.unsqueeze(1) + package_radii)
            .any()
        )

    def close(self):
        pass
