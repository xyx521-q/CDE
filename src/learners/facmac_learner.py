import copy

import torch as th
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from modules.critic import FACMACCritic
from modules.graph_mixer import GraphDec


class FACMACLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = FACMACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.mixer = None
        if self.args.n_agents > 1:
            self.mixer = GraphDec(args)
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(
                params=self.agent_params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(
                params=self.agent_params,
                lr=args.lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            raise Exception(
                "unknown optimizer {}".format(
                    getattr(self.args, "optimizer", "rmsprop")
                )
            )

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(
                params=self.critic_params,
                lr=args.critic_lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(
                params=self.critic_params,
                lr=args.critic_lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            raise Exception(
                "unknown optimizer {}".format(
                    getattr(self.args, "optimizer", "rmsprop")
                )
            )

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Train the critic batched
        with th.no_grad():
            target_actions = []
            target_hidden_states = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_target_outs = self.target_mac.select_actions(
                    batch,
                    t_ep=t,
                    t_env=None,
                    test_mode=True,
                )
                target_actions.append(agent_target_outs)
                if self.mixer is not None:
                    target_hidden_states.append(
                        self.target_mac.hidden_states.view(
                            batch.batch_size, self.n_agents, -1
                        )
                    )
            target_actions = th.stack(target_actions, dim=1)
            if self.mixer is not None:
                target_hidden_states = th.stack(target_hidden_states, dim=1)

        current_hidden_states = None
        if self.mixer is not None:
            current_hidden_states = self._collect_mac_hidden_states(batch, self.mac)

        self.critic.init_hidden(batch.batch_size)
        critic_out, self.critic.hidden_states = self.critic(
            self._build_inputs_sequence(batch, end=-1),
            actions.detach(),
            self.critic.hidden_states,
        )
        if self.mixer is not None:
            q_taken = self._mix_q_values(
                self.mixer,
                critic_out,
                batch["state"][:, :-1],
                agent_obs=batch["obs"][:, :-1],
                hidden_states=current_hidden_states[:, :-1],
            )
        else:
            q_taken = critic_out.view(
                batch.batch_size, batch.max_seq_length - 1, self.n_agents
            )

        with th.no_grad():
            self.target_critic.init_hidden(batch.batch_size)
            target_critic_out, self.target_critic.hidden_states = self.target_critic(
                self._build_inputs_sequence(batch, start=1),
                target_actions[:, 1:],
                self.target_critic.hidden_states,
            )
            if self.mixer is not None:
                target_vals = self._mix_q_values(
                    self.target_mixer,
                    target_critic_out,
                    batch["state"][:, 1:],
                    agent_obs=batch["obs"][:, 1:],
                    hidden_states=target_hidden_states[:, 1:],
                )
            else:
                target_vals = target_critic_out.view(
                    batch.batch_size, batch.max_seq_length - 1, self.n_agents
                )

        targets = (
            rewards.expand_as(target_vals)
            + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals
        )
        td_error = targets.detach() - q_taken
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        # Train the actor
        # Optimize over the entire joint action space
        mac_out = []
        raw_mac_out = []
        actor_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mac_ret = self.mac.forward(batch, t=t, select_actions=True)
            agent_outs = mac_ret["actions"].view(
                batch.batch_size, self.n_agents, self.n_actions
            )
            raw_agent_outs = mac_ret["raw_actions"].view(
                batch.batch_size, self.n_agents, self.n_actions
            )
            if self.mixer is not None:
                actor_hidden_states.append(
                    self.mac.hidden_states.view(batch.batch_size, self.n_agents, -1)
                )
            mac_out.append(agent_outs)
            raw_mac_out.append(raw_agent_outs)
        mac_out = th.stack(mac_out[:-1], dim=1)
        raw_mac_out = th.stack(raw_mac_out[:-1], dim=1)
        self._set_critic_grad(False)
        chosen_action_qvals, self.critic.hidden_states = self.critic(
            self._build_inputs_sequence(batch, end=-1),
            mac_out,
            self.critic.hidden_states,
        )
        if self.mixer is not None:
            actor_hidden_states = th.stack(actor_hidden_states[:-1], dim=1)
            chosen_action_qvals = self._mix_q_values(
                self.mixer,
                chosen_action_qvals,
                batch["state"][:, :-1],
                agent_obs=batch["obs"][:, :-1],
                hidden_states=actor_hidden_states,
            )
        else:
            chosen_action_qvals = chosen_action_qvals.view(
                batch.batch_size, batch.max_seq_length - 1, self.n_agents
            )
        pi = raw_mac_out

        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad(set_to_none=True)
        pg_loss.backward()
        self._set_critic_grad(True)
        th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.grad_norm_clip
        )
        self.agent_optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            self._update_targets()
        elif getattr(self.args, "target_update_mode", "hard") in [
            "soft",
            "exponential_moving_average",
        ]:
            self._update_targets_soft(
                tau=getattr(self.args, "target_update_tau", 0.001)
            )
        else:
            raise Exception(
                "unknown target update mode: {}!".format(
                    getattr(self.args, "target_update_mode", "hard")
                )
            )

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(
                "loss/critic_loss", loss.item(), t_env
            )
            self.logger.log_stat(
                "loss/policy_loss", pg_loss.item(), t_env
            )
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):
        with th.no_grad():
            for target_param, param in zip(
                self.target_mac.parameters(), self.mac.parameters()
            ):
                target_param.lerp_(param, tau)

            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.lerp_(param, tau)

            if self.mixer is not None:
                for target_param, param in zip(
                    self.target_mixer.parameters(), self.mixer.parameters()
                ):
                    target_param.lerp_(param, tau)

    def _mix_q_values(
        self,
        mixer,
        agent_qs,
        states,
        agent_obs=None,
        hidden_states=None,
        team_rewards=None,
    ):
        if self.mixer is not None:
            q_tot, _, _ = mixer(
                agent_qs,
                states,
                agent_obs=agent_obs,
                team_rewards=team_rewards,
                hidden_states=hidden_states,
            )
            return q_tot
        return mixer(agent_qs, states)

    def _collect_mac_hidden_states(self, batch, mac):
        hidden_states = []
        mac.init_hidden(batch.batch_size)
        with th.no_grad():
            for t in range(batch.max_seq_length):
                mac.forward(batch, t=t, select_actions=True)
                hidden_states.append(
                    mac.hidden_states.view(batch.batch_size, self.n_agents, -1).detach()
                )
        return th.stack(hidden_states, dim=1)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        if self.args.recurrent_critic:
            # The individual Q conditions on the global action-observation history and individual action
            inputs.append(
                batch["obs"][:, t]
                .repeat(1, self.args.n_agents, 1)
                .view(bs, self.args.n_agents, -1)
            )
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(
                        th.zeros_like(
                            batch["actions"][:, t]
                            .repeat(1, self.args.n_agents, 1)
                            .view(bs, self.args.n_agents, -1)
                        )
                    )
                else:
                    inputs.append(
                        batch["actions"][:, t - 1]
                        .repeat(1, self.args.n_agents, 1)
                        .view(bs, self.args.n_agents, -1)
                    )
        else:
            inputs.append(batch["obs"][:, t])

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs_sequence(self, batch, start=0, end=None):
        if self.args.recurrent_critic:
            stop = batch.max_seq_length if end is None else end % batch.max_seq_length
            inputs = [
                self._build_inputs(batch, t).view(
                    batch.batch_size, self.n_agents, -1
                )
                for t in range(start, stop)
            ]
            return th.stack(inputs, dim=1).reshape(-1, inputs[0].shape[-1])
        return batch["obs"][:, start:end].reshape(-1, batch["obs"].shape[-1])

    def _set_critic_grad(self, enabled):
        for param in self.critic_params:
            param.requires_grad_(enabled)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.agent_optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage)
            )
        self.agent_optimiser.load_state_dict(
            th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage)
        )
