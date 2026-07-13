from functools import partial

import numpy as np
import torch as th

from components.episode_buffer import EpisodeBatch
from envs.maenv import MGEnv


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = MGEnv(config=self.args.env_args, algo_name=self.args.name)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            # Environment interaction and replay insertion remain CPU-resident.
            device="cpu",
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()
        if test_mode:
            self.env.reset_test()
        else:
            self.env.reset()
        self.t = 0

    def run(self, mac, test_mode=False):
        self.mac = mac
        self.reset(test_mode=test_mode)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            reward, terminated, env_info = self.env.step(actions[0].cpu())
            episode_return += reward

            post_transition_data = {
                # Rollout batches are CPU-resident even when policy inference uses CUDA.
                "actions": actions.cpu(),
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )

        self.batch.update({"actions": actions.cpu()}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self.log_train_stats_t = self.t_env

        return self.batch, episode_return

    def _log(self, returns, stats, prefix):
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        metric_prefix = "test" if prefix == "test_" else "train"

        self.logger.log_stat(
            f"{metric_prefix}/return_mean", return_mean, self.t_env
        )
        self.logger.log_stat(
            f"{metric_prefix}/return_std", return_std, self.t_env
        )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                stat_value = v / stats["n_episodes"]
                self.logger.log_stat(
                    f"{metric_prefix}/{k}", stat_value, self.t_env
                )
        stats.clear()
