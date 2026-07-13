import numpy as np
import torch as th
from gymnasium import spaces

from modules.policy import RNNAgent

class CQMixMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent = RNNAgent(self._get_input_shape(scheme), args)
        self.hidden_states = None
        if not all(isinstance(space, spaces.Box) for space in args.action_spaces):
            raise ValueError("Only continuous Box action spaces are supported.")
        if any(space.shape != (args.n_actions,) for space in args.action_spaces):
            raise ValueError("All agents must use the same flat action shape.")
        self._action_lows = th.as_tensor(
            np.stack([space.low for space in args.action_spaces]), dtype=th.float32
        ).unsqueeze(0)
        self._action_highs = th.as_tensor(
            np.stack([space.high for space in args.action_spaces]), dtype=th.float32
        ).unsqueeze(0)
        self._action_bounds_cache = {}

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )

    def parameters(self):
        return self.agent.parameters()

    def named_parameters(self):
        return self.agent.named_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def load_state_from_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)

    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage)
        )

    def _scale_actions_to_space(self, actions):
        low, high = self._get_action_bounds(actions)
        return low + 0.5 * (actions + 1.0) * (high - low)

    def _clamp_actions_to_space(self, actions):
        low, high = self._get_action_bounds(actions)
        return th.maximum(th.minimum(actions, high), low)

    def _get_action_bounds(self, actions):
        cache_key = (str(actions.device), actions.dtype)
        bounds = self._action_bounds_cache.get(cache_key)
        if bounds is None:
            bounds = (
                self._action_lows.to(device=actions.device, dtype=actions.dtype),
                self._action_highs.to(device=actions.device, dtype=actions.dtype),
            )
            self._action_bounds_cache[cache_key] = bounds
        return bounds

    def select_actions(
        self,
        ep_batch,
        t_ep,
        t_env,
        bs=slice(None),
        test_mode=False,
    ):
        chosen_actions = self.forward(ep_batch[bs], t_ep, select_actions=True)[
            "actions"
        ].detach()

        # Now do appropriate noising
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
        # Ornstein-Uhlenbeck:
        if not test_mode:  # do exploration
            if exploration_mode == "ornstein_uhlenbeck":
                x = getattr(self, "ou_noise_state", chosen_actions.clone().zero_())
                mu = 0
                theta = getattr(self.args, "ou_theta", 0.15)
                sigma = getattr(self.args, "ou_sigma", 0.2)

                noise_scale = (
                    getattr(self.args, "ou_noise_scale", 0.3)
                    if t_env
                    < self.args.env_args["episode_limit"] * self.args.ou_stop_episode
                    else 0.0
                )
                dx = theta * (mu - x) + sigma * x.clone().normal_()
                self.ou_noise_state = x + dx
                ou_noise = self.ou_noise_state * noise_scale
                chosen_actions = chosen_actions + ou_noise
            elif exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                if t_env >= start_steps:
                    x = chosen_actions.clone().zero_()
                    chosen_actions += act_noise * x.clone().normal_()
                else:
                    chosen_actions = (
                        th.from_numpy(
                            np.array(
                                [
                                    [
                                        self.args.action_spaces[i].sample()
                                        for i in range(self.n_agents)
                                    ]
                                    for _ in range(ep_batch[bs].batch_size)
                                ]
                            )
                        )
                        .float()
                        .to(device=chosen_actions.device)
                    )

        # For continuous actions, clamp after exploration so the environment and critic see valid actions.
        return self._clamp_actions_to_space(chosen_actions)

    def forward(
        self,
        ep_batch,
        t,
        actions=None,
        select_actions=False,
    ):
        agent_inputs = self._build_inputs(ep_batch, t).to(
            next(self.agent.parameters()).device
        )
        ret = self.agent(agent_inputs, self.hidden_states, actions=actions)
        if "actions" in ret:
            raw_actions = ret["actions"].view(
                ep_batch.batch_size, self.n_agents, self.args.n_actions
            )
            scaled_actions = self._scale_actions_to_space(raw_actions)
            ret = dict(ret)
            ret["raw_actions"] = raw_actions
            ret["actions"] = scaled_actions
        if select_actions:
            self.hidden_states = ret["hidden_state"]
            return ret
        return ret

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
