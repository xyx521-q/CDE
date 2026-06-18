import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.facmac import FACMACDiscreteCritic, PeVFA_FACMACDiscreteCritic
# from components.action_selectors import multinomial_entropy
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer, PeVFA_QMixer, V_Net
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic
from modules.mixers.graph_Dec import GraphDec
from utils.rl_utils import build_td_lambda_targets
import random
import torch.nn as nn
import time
import math
from torch.nn import functional as F
import torch


import numpy as np


class EA_FACMACDiscreteLearner:
    uses_population_training = True

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)

        self.agent_params = list(mac.parameters())

        self.critic = FACMACDiscreteCritic(scheme, args)
        # self.target_critic = copy.deepcopy(self.critic)
        self.target_critic = FACMACDiscreteCritic(scheme, args)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_params = list(self.critic.parameters())

        if args.EA:
            self.PeVFA_critic = PeVFA_FACMACDiscreteCritic(scheme, args)
            self.PeVFA_params = list(self.PeVFA_critic.parameters())
            # self.target_PeVFA_critic = PeVFA_FACMACDiscreteCritic(scheme, args)
            # self.target_PeVFA_critic.load_state_dict(self.PeVFA_critic.state_dict())
            self.target_PeVFA_critic = copy.deepcopy(self.PeVFA_critic)

            if args.mixer == "graph":
                self.PeVFA_mixer = GraphDec(args)
                self.target_PeVFA_mixer = GraphDec(args)
            else:
                self.PeVFA_mixer = PeVFA_QMixer(args)

            self.PeVFA_params += list(self.PeVFA_mixer.parameters())
            self.target_PeVFA_mixer = copy.deepcopy(self.PeVFA_mixer)
            # self.target_PeVFA_mixer.load_state_dict(self.PeVFA_mixer.state_dict())
            self.PeVFA_optimiser = Adam(params=self.PeVFA_params, lr=args.critic_lr,
                                        eps=getattr(args, "optimizer_epsilon", 10E-8))

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(args)
            elif args.mixer == "graph":
                self.mixer = QMixer(args)
                # self.target_mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            self.critic_params += list(self.mixer.parameters())
            # self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha,
                                           eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr,
                                        eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                            eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr,
                                         eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

    def train(self, batch: EpisodeBatch, all_teams, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        # actions = batch["actions"][:, :]
        actions = batch["actions_onehot"][:, :]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        temp_mask = mask

        start = time.time()

        if self.args.use_cuda:
            self.mac.cuda()
            self.target_mac.cuda()
            self.critic.cuda()
            self.target_critic.cuda()

            if self.mixer is not None:
                self.mixer.cuda()
                self.target_mixer.cuda()

        if self.args.EA:
            # Train the critic batched
            index = random.sample(list(range(self.args.pop_size + 1)), 1)[0]
            selected_team = all_teams[index]

            if self.args.use_cuda:
                selected_team.cuda()
                self.PeVFA_critic.cuda()
                self.target_PeVFA_critic.cuda()
                self.PeVFA_mixer.cuda()
                self.target_PeVFA_mixer.cuda()

            target_mac_out = []
            selected_team.init_hidden(batch.batch_size)
            hidden_states = []
            for t in range(batch.max_seq_length):
                target_act_outs = selected_team.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
                target_mac_out.append(target_act_outs)
                hidden_states.append(selected_team.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

            hidden_states = th.stack(hidden_states, dim=1)

            q_taken_origi_ea, _ = self.PeVFA_critic(batch["obs"][:, :-1], actions[:, :-1])

            if self.mixer is not None:
                if self.args.mixer == "vdn":
                    assert 1 == 2
                elif self.args.mixer == "graph":
                    q_taken, local_rewards, alive_agents_mask = self.PeVFA_mixer(q_taken_origi_ea,
                                                                                 batch["state"][:, :-1],
                                                                                 agent_obs=batch["obs"][:, :-1],
                                                                                 team_rewards=rewards,
                                                                                 hidden_states=hidden_states[:, :-1],
                                                                                 )
                else:
                    q_taken = self.PeVFA_mixer.forward(q_taken_origi_ea.view(batch.batch_size, -1, 1),
                                                       batch["state"][:, :-1])

            target_vals, _ = self.target_PeVFA_critic(batch["obs"][:, :], target_mac_out.detach())
            if self.mixer is not None:
                if self.args.mixer == "graph":
                    target_vals = self.target_PeVFA_mixer(target_vals.view(batch.batch_size, -1, 1),
                                                          batch["state"][:, :],
                                                          agent_obs=batch["obs"][:, :],
                                                          hidden_states=hidden_states
                                                          )[0]
                else:
                    target_vals = self.target_PeVFA_mixer.forward(target_vals.view(batch.batch_size, -1, 1),
                                                                  batch["state"][:, :], )

            if self.mixer is not None:
                q_taken = q_taken.view(batch.batch_size, -1, 1)
                target_vals = target_vals.view(batch.batch_size, -1, 1)
            else:
                q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
                target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

            targets = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals, self.n_agents,
                                              self.args.gamma, self.args.td_lambda)
            mask = temp_mask[:, :-1]
            td_error = (q_taken - targets.detach())
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask
            ea_loss = (masked_td_error ** 2).sum() / mask.sum()

            ########################################local_loss################################################################
            # if self.args.mixer == "graph":
            #     local_targets = local_rewards.detach() + self.args.gamma * (1 - terminated[:, :-1]).repeat(1, 1,
            #                                                                                                self.args.n_agents)
            #
            #     # Td-error
            #     local_td_error = (q_taken_origi_ea.reshape(local_targets.size()) - local_targets)
            #     local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask.float()
            #
            #     # 0-out the targets that came from padded data
            #     local_masked_td_error = local_td_error * local_mask
            #
            #     # Normal L2 loss, take mean over actual data
            #     local_loss = (local_masked_td_error ** 2).sum() / mask.sum()
            #
            #     ea_loss += local_loss
            ########################################local_loss################################################################

            self.PeVFA_optimiser.zero_grad()
            ea_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.PeVFA_params, self.args.grad_norm_clip)
            self.PeVFA_optimiser.step()

        # print("1 ", time.time()-start)
        start = time.time()

        target_mac_out = []
        target_hidden_states = []
        hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_act_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            target_mac_out.append(target_act_outs)
            target_hidden_states.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
            hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time
        target_hidden_states = th.stack(target_hidden_states, dim=1)
        hidden_states = th.stack(hidden_states, dim=1)

        q_taken_origi, _ = self.critic(batch["obs"][:, :-1], actions[:, :-1])
        if self.mixer is not None:
            q_taken = self.mixer(q_taken_origi.view(batch.batch_size, -1, 1), batch["state"][:, :-1])

        target_vals, _ = self.target_critic(batch["obs"][:, :], target_mac_out.detach())
        if self.mixer is not None:
            target_vals = self.target_mixer(target_vals.view(batch.batch_size, -1, 1), batch["state"][:, :])

        if self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        targets_1 = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals, self.n_agents,
                                            self.args.gamma, self.args.td_lambda)
        mask = temp_mask[:, :-1]
        td_error = (q_taken - targets_1.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()


        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        # print("2 ", time.time() - start)
        start = time.time()

        # Train the actor
        # Use gumbel softmax to reparameterize the stochastic policies as deterministic functions of independent
        # noise to compute the policy gradient (one hot action input to the critic)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            act_outs = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)
            mac_out.append(act_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        chosen_action_qvals, _ = self.critic(batch["obs"][:, :-1], mac_out)

        if self.mixer is not None:

            chosen_action_qvals = self.mixer(chosen_action_qvals.view(batch.batch_size, -1, 1),
                                             batch["state"][:, :-1])

        # Compute the actor loss
        pg_loss = - (chosen_action_qvals * mask).sum() / mask.sum()

        # print("3 ", time.time() - start)
        start = time.time()
        MINE_loss = 0
        if self.args.EA:

            mac_out = []
            hidden_states = []
            index = random.sample(list(range(self.args.pop_size + 1)), 1)[0]
            selected_team = all_teams[index]
            selected_team.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                act_outs = selected_team.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)

                mac_out.append(act_outs)
                hidden_states.append(selected_team.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            hidden_states = th.stack(hidden_states, dim=1)

            chosen_action_qvals, _ = self.PeVFA_critic(batch["obs"][:, :-1], mac_out)

            if self.mixer is not None:
                if self.args.mixer == "graph":
                    chosen_action_qvals, local_rewards, alive_agents_mask = self.PeVFA_mixer(
                        chosen_action_qvals.view(batch.batch_size, -1, 1),
                        batch["state"][:, :-1],
                        agent_obs=batch["obs"][:, :-1],
                        team_rewards=rewards,
                        hidden_states=hidden_states
                    )
                else:
                    chosen_action_qvals = self.PeVFA_mixer.forward(chosen_action_qvals.view(batch.batch_size, -1, 1),
                                                                   batch["state"][:, :-1])

            # Compute the actor loss
            # ea_pg_loss = - self.args.EA_alpha * (
            #         chosen_action_qvals * mask).sum() / mask.sum() + self.args.state_alpha * MINE_loss
            ea_pg_loss = -  (
                    chosen_action_qvals * mask).sum() / mask.sum()
        else:
            ea_pg_loss = 0.0
        # ea_pg_loss = 0.0
        # total_loss = pg_loss + self.args.EA_alpha *ea_pg_loss

        total_loss = self.args.Org_alpha * pg_loss + ea_pg_loss
        # Optimise agents
        # self.MINE_optimiser.zero_grad()
        self.agent_optimiser.zero_grad()
        total_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # print("3 ", time.time() - start)
        start = time.time()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("loss/critic_loss", loss.item(), t_env, to_sacred=False)
            self.logger.log_stat("grad/critic_norm", critic_grad_norm, t_env, to_sacred=False)
            self.logger.log_stat("value/td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env, to_sacred=False)
            self.logger.log_stat("value/target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env, to_sacred=False)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):

        if self.args.EA:
            for target_param, param in zip(self.target_PeVFA_critic.parameters(), self.PeVFA_critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

            for target_param, param in zip(self.target_PeVFA_mixer.parameters(), self.PeVFA_mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _update_targets(self):

        if self.args.EA:
            self.target_PeVFA_mixer.load_state_dict(self.PeVFA_mixer.state_dict())
            self.target_PeVFA_critic.load_state_dict(self.PeVFA_critic.state_dict())
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.device = device
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
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
