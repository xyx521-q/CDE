# `maenv.py` 完整代码与解析

> 对应原文件：`src/envs/maenv.py`。本文件保留了当前原始文件的完整代码副本；代码块后的章节按执行顺序说明 Python 语法、数据流、物理语义和框架接口。这里的代码副本只用于阅读，实际运行仍使用原文件。

## 一、完整代码

```python
import csv
import functools
import pathlib

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


def standardize_data(data: np.ndarray, mean=None, std=None) -> np.ndarray:
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    if np.isclose(std, 0):
        return data - mean
    return (data - mean) / std


class MGEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", None],
        "name": "MGEnv_v0",
    }

    def __init__(
        self,
        config,
        algo_name,
        render_mode="human",
    ):
        self.render_mode = render_mode
        self.n_agents = len(config["data"])
        self.state = None
        self.possible_agents = ["agent" + str(r + 1) for r in range(self.n_agents)]
        agents_config = {agent.pop("agent_name"): agent for agent in config["data"]}
        self.params = self.params_init(agents_config)

        self.action_lows = np.stack(
            [
                np.array(agents_config[agent]["low_action"], dtype=np.float32)
                for agent in self.possible_agents
            ],
            axis=0,
        )
        self.action_highs = np.stack(
            [
                np.array(agents_config[agent]["high_action"], dtype=np.float32)
                for agent in self.possible_agents
            ],
            axis=0,
        )
        self.n_actions = int(self.action_lows.shape[1])

        self.reward_aggregate = config.get("reward_aggregate", "sum")
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.reward_mode = config.get("reward_mode", "legacy")
        self.buy_cost_weight = float(config.get("buy_cost_weight", 1.0))
        self.buy_reward_mode = config.get("buy_reward_mode", "raw_cost")
        self.buy_reward_scale = float(config.get("buy_reward_scale", 100.0))
        self.generate_cost_weight = float(config.get("generate_cost_weight", 1.0))
        self.bess_cost_weight = float(config.get("bess_cost_weight", 1.0))
        self.env_reward_weight = float(config.get("env_reward_weight", 1.0))
        self.battery_punishment_weight = float(
            config.get("battery_punishment_weight", 1.0)
        )
        self.curtailment_penalty_weight = float(
            config.get("curtailment_penalty_weight", 0.0)
        )
        self.soc_reserve_target = float(config.get("soc_reserve_target", 0.35))
        self.soc_reserve_weight = float(config.get("soc_reserve_weight", 0.0))
        self._ep_steps = 0
        self._ep_total_power_loss = 0.0
        self._ep_grid_purchase_cost = 0.0
        self._ep_baseline_grid_purchase_cost = 0.0
        self._ep_grid_purchase_saving = 0.0
        self._ep_soc_violation = 0.0
        self._ep_mean_soc_sum = 0.0
        self._ep_buy_cost_reward = 0.0
        self._ep_generate_cost_reward = 0.0
        self._ep_bess_cost_reward = 0.0
        self._ep_battery_punishment_reward = 0.0
        self._ep_battery_action_clip = 0.0
        self._ep_curtailment_reward = 0.0
        self._ep_soc_reserve_reward = 0.0
        self._ep_economic_reward = 0.0
        self._ep_economic_cost = 0.0
        self._ep_curtailment_cost = 0.0

        env_folder_path = pathlib.Path(__file__ or ".").parent

        # 2. 定位到数据表格所在的 data/ 专属宿舍（即 src/envs/data/）

        data_path = env_folder_path / "data"

        from pandas import read_csv

        price_data = np.array(read_csv(data_path / "price.csv", header=None))
        pv_data = np.array(read_csv(data_path / "pv.csv", header=None))
        load_data = np.array(read_csv(data_path / "load.csv", header=None))
        price_min = float(np.min(price_data))
        price_max = float(np.max(price_data))

        self.spaces = {}
        for agent in self.possible_agents:
            base_low = np.array(agents_config[agent]["low_state"], dtype=np.float32)
            base_high = np.array(agents_config[agent]["high_state"], dtype=np.float32)
            obs_low = np.concatenate(
                [base_low, np.array([price_min, 0.0], dtype=np.float32)]
            )
            obs_high = np.concatenate(
                [base_high, np.array([price_max, 1.0], dtype=np.float32)]
            )
            self.spaces[agent] = {
                "observation": spaces.Box(
                    low=obs_low,
                    high=obs_high,
                    shape=(obs_low.shape[0],),
                ),
                "action": spaces.Box(
                    low=self.action_lows[self.possible_agents.index(agent)],
                    high=self.action_highs[self.possible_agents.index(agent)],
                    shape=(self.n_actions,),
                    dtype=np.float32,
                ),
            }

        self.data_shape = price_data.shape
        self.idx = [0, 0]

        self.seed = config.get("seed", None)
        self.rng = np.random.default_rng(self.seed)
        num_rows = self.data_shape[0]
        perm = self.rng.permutation(num_rows)
        split_point = int(num_rows * 0.8)
        self.train_indices = perm[:split_point]
        self.test_indices = perm[split_point:]

        pv_train = pv_data[self.train_indices]
        load_train = load_data[self.train_indices]
        pv_sd = standardize_data(pv_data, mean=pv_train.mean(), std=pv_train.std())
        load_sd = standardize_data(
            load_data, mean=load_train.mean(), std=load_train.std()
        )

        self.data_dict = {"price": price_data, "load_pv": load_data - pv_data}
        for key in agents_config.keys():
            self.data_dict[key] = {}
            self.data_dict[key]["pv_data"] = pv_sd * agents_config[key]["split_ratio"]
            self.data_dict[key]["load_data"] = (
                load_sd * agents_config[key]["split_ratio"]
            )
            self.data_dict[key]["load"] = load_data * agents_config[key]["split_ratio"]
            self.data_dict[key]["pv"] = pv_data * agents_config[key]["split_ratio"]

        run_id = config.get("run_id", "no_run_id")
        self.log_trajectories = bool(config.get("log_trajectories", False))
        self.log_trajectories_mode = config.get("log_trajectories_mode", "test")
        self.render_dir = (
            pathlib.Path(config["result_dir"])
            / "microgrid_trajs"
            / str(algo_name)
            / str(run_id)
        )
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.ep_counter = 0
        self.ep_buffer = []
        self.is_testing = False
        self.test_source = "test"

        self.bucket_size = 10_000
        self.episode_limit = self.data_shape[1]

    def get_env_info(self):
        """框架通过此函数自动配置神经网络的输入输出维度"""
        first_agent = self.possible_agents[0]
        # 每个智能体的观察维度 (当前代码中是 3 维: load_data, pv_data, soc)
        obs_shape = self.spaces[first_agent]["observation"].shape[0]
        action_shape = self.spaces[first_agent]["action"].shape[0]

        return {
            "n_agents": self.n_agents,  # 智能体数量
            "n_actions": action_shape,  # 每个智能体的动作维数
            "obs_shape": obs_shape,  # 每个智能体的观察维数
            "state_shape": obs_shape
            * self.n_agents,  # 全局状态维数（所有智能体obs拼接）
            "episode_limit": self.data_shape[
                1
            ],  # 一局的最大时序步数（你表格里的总列数，比如24）
            "action_spaces": [
                self.spaces[agent]["action"] for agent in self.possible_agents
            ],
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }

    def get_obs(self):
        """返回一个长度为 n_agents 的列表，每个元素是对应 Agent 的 NumPy 数组"""
        time_frac = self.idx[1] / max(self.episode_limit - 1, 1)
        price = float(self.data_dict["price"][tuple(self.idx)])
        return [
            np.array(
                [
                    self.data_dict[agent_id]["load_data"][tuple(self.idx)],
                    self.data_dict[agent_id]["pv_data"][tuple(self.idx)],
                    self.params["socs"][i],
                    price,
                    time_frac,
                ],
                dtype=np.float32,
            )
            for i, agent_id in enumerate(self.possible_agents)
        ]

    def get_state(self):
        """Return the concatenated observations as the global state."""
        return np.concatenate(self.get_obs())

    def params_init(self, agents_config):
        return {
            "dg_max": np.array(
                [agents_config[a]["dg_max"] for a in self.possible_agents],
                dtype=float,
            ),
            "battery_caps": np.array(
                [agents_config[a]["battery_cap"] for a in self.possible_agents],
                dtype=float,
            ),
            "battery_punishment_params": np.array(
                [
                    agents_config[a]["battery_punishment_param"]
                    for a in self.possible_agents
                ],
                dtype=float,
            ),
            "raw_ch": np.array(
                [agents_config[a]["raw_ch"] for a in self.possible_agents],
                dtype=float,
            ),
            "raw_dis": np.array(
                [agents_config[a]["raw_dis"] for a in self.possible_agents],
                dtype=float,
            ),
            "bessa": np.array(
                [agents_config[a]["bessa"] for a in self.possible_agents],
                dtype=float,
            ),
            "bessb": np.array(
                [agents_config[a]["bessb"] for a in self.possible_agents],
                dtype=float,
            ),
            "bessc": np.array(
                [agents_config[a]["bessc"] for a in self.possible_agents],
                dtype=float,
            ),
            "costa": np.array(
                [agents_config[a]["costa"] for a in self.possible_agents],
                dtype=float,
            ),
            "costb": np.array(
                [agents_config[a]["costb"] for a in self.possible_agents],
                dtype=float,
            ),
            "costc": np.array(
                [agents_config[a]["costc"] for a in self.possible_agents],
                dtype=float,
            ),
            "env_param": np.array(
                [agents_config[a]["env_param"] for a in self.possible_agents],
                dtype=float,
            ),
            "split_ratio": np.array(
                [agents_config[a]["split_ratio"] for a in self.possible_agents],
                dtype=float,
            ),
            "socs": np.full(self.n_agents, 0.2, dtype=float),
        }

    @functools.cache
    def observation_space(self, agent: int):
        agent = "agent" + str(agent)
        return self.spaces[agent]["observation"]

    @functools.cache
    def action_space(self, agent: int):
        agent = "agent" + str(agent)
        return self.spaces[agent]["action"]

    def reset(self, seed=None, options=None):
        """改动：剥离原本的字典外壳，直接返回 get_obs()"""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        row_index = int(rng.choice(self.train_indices))
        self.idx = [row_index, 0]
        self.params["socs"] = rng.uniform(0.2, 0.5, size=self.n_agents)
        self.ep_buffer = []
        self.ep_counter += 1
        self.is_testing = False
        self.test_source = "train"
        self._ep_steps = 0
        self._ep_total_power_loss = 0.0
        self._ep_grid_purchase_cost = 0.0
        self._ep_baseline_grid_purchase_cost = 0.0
        self._ep_grid_purchase_saving = 0.0
        self._ep_soc_violation = 0.0
        self._ep_mean_soc_sum = 0.0
        self._ep_buy_cost_reward = 0.0
        self._ep_generate_cost_reward = 0.0
        self._ep_bess_cost_reward = 0.0
        self._ep_battery_punishment_reward = 0.0
        self._ep_battery_action_clip = 0.0
        self._ep_curtailment_reward = 0.0
        self._ep_soc_reserve_reward = 0.0
        self._ep_economic_reward = 0.0
        self._ep_economic_cost = 0.0
        self._ep_curtailment_cost = 0.0

        # 严格注意：框架的 runner.py 里写的是 self.env.reset()，它只认观察列表
        return self.get_obs()

    def reset_test(self, seed=None, source: str = "test"):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        if source == "test":
            row_index = int(rng.choice(self.test_indices))
        elif source == "train":
            row_index = int(rng.choice(self.train_indices))
        else:
            num_rows = self.data_shape[0]
            row_index = int(rng.integers(num_rows))
        self.idx = [row_index, 0]
        self.params["socs"] = rng.uniform(0.2, 0.5, size=self.n_agents)
        self.ep_buffer = []
        self.ep_counter += 1
        self.is_testing = True
        self.test_source = source
        self._ep_steps = 0
        self._ep_total_power_loss = 0.0
        self._ep_grid_purchase_cost = 0.0
        self._ep_baseline_grid_purchase_cost = 0.0
        self._ep_grid_purchase_saving = 0.0
        self._ep_soc_violation = 0.0
        self._ep_mean_soc_sum = 0.0
        self._ep_buy_cost_reward = 0.0
        self._ep_generate_cost_reward = 0.0
        self._ep_bess_cost_reward = 0.0
        self._ep_battery_punishment_reward = 0.0
        self._ep_battery_action_clip = 0.0
        self._ep_curtailment_reward = 0.0
        self._ep_soc_reserve_reward = 0.0
        self._ep_economic_reward = 0.0
        self._ep_economic_cost = 0.0
        self._ep_curtailment_cost = 0.0
        return self.get_obs()

    def step(self, actions):
        """
        EPyMARL框架传入连续动作，直接映射到发电功率和电池功率
        """
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        actions_raw = np.asarray(actions, dtype=np.float32).reshape(
            self.n_agents, self.n_actions
        )
        actions_raw = np.clip(actions_raw, self.action_lows, self.action_highs)

        pds = actions_raw[:, 0] * self.params["dg_max"]
        pbs_requested = actions_raw[:, 1] * self.params["battery_caps"]

        socs = self.params["socs"].copy()
        battery_caps = self.params["battery_caps"]
        battery_punishment_params = self.params["battery_punishment_params"]

        last_socs = socs
        delta_soc = np.where(
            pbs_requested < 0,
            -self.params["raw_ch"] * pbs_requested / battery_caps,
            -self.params["raw_dis"] * pbs_requested / battery_caps,
        )
        socs_raw = last_socs + delta_soc
        socs = socs + delta_soc

        clamped_low = socs < 0.2
        clamped_high = socs > 0.8
        pb_adj = pbs_requested.copy()
        pb_adj[clamped_low] = (last_socs[clamped_low] - 0.2) * battery_caps[clamped_low]
        pb_adj[clamped_high] = (last_socs[clamped_high] - 0.8) * battery_caps[
            clamped_high
        ]

        # 惩罚按“请求动作与最终执行动作的偏差”计算，避免 reward 与环境实际执行错位。
        pbs = pb_adj
        battery_action_clip = np.abs(pbs_requested - pbs)
        battery_punishment = -battery_punishment_params * battery_action_clip

        delta_soc = np.where(
            pbs < 0,
            -self.params["raw_ch"] * pbs / battery_caps,
            -self.params["raw_dis"] * pbs / battery_caps,
        )
        socs = np.clip(last_socs + delta_soc, 0.2, 0.8)

        self.params["socs"] = socs
        pg = self.data_dict["load_pv"][tuple(self.idx)] - sum(pds) - sum(pbs)

        x = pbs + 3 * battery_caps * (1 - socs)
        bess_cost = np.abs(
            self.params["bessa"] * x**2
            + self.params["bessb"] * x
            + self.params["bessc"]
        )

        generate_costs = (
            self.params["costa"] * pds**2
            + self.params["costb"] * pds
            + self.params["costc"]
        )
        baseline_pg = max(float(self.data_dict["load_pv"][tuple(self.idx)]), 0.0)
        baseline_grid_purchase_cost = (
            baseline_pg * self.data_dict["price"][tuple(self.idx)]
        )
        actual_grid_purchase_cost = (
            max(float(pg), 0.0) * self.data_dict["price"][tuple(self.idx)]
        )
        if self.reward_mode == "milp_economic":
            split_weights = self.params["split_ratio"] / max(
                np.sum(self.params["split_ratio"]), 1e-6
            )
            grid_cost_share = actual_grid_purchase_cost * split_weights
            env_cost = pds * self.params["env_param"]
            curtailment = np.full(self.n_agents, np.maximum(-pg, 0.0), dtype=float)
            curtailment_cost = (
                self.curtailment_penalty_weight * curtailment * split_weights
            )
            economic_cost = (
                grid_cost_share
                + generate_costs
                + bess_cost
                + env_cost
                + curtailment_cost
            )
            economic_reward = -economic_cost
            weighted_generate_costs = -generate_costs
            weighted_bess_cost = -bess_cost
            weighted_buy_cost = -grid_cost_share
            weighted_curtailment_cost = -curtailment_cost
        elif self.buy_reward_mode == "relative_saving":
            buy_signal = self.buy_reward_scale * (
                (baseline_grid_purchase_cost - actual_grid_purchase_cost)
                / max(baseline_grid_purchase_cost, 1e-6)
            )
            buy_cost = float(buy_signal)
            weighted_buy_cost = self.buy_cost_weight * buy_cost
            weighted_generate_costs = self.generate_cost_weight * (-generate_costs)
            weighted_bess_cost = self.bess_cost_weight * (-bess_cost)
            env_cost = pds * self.params["env_param"]
            weighted_curtailment_cost = -self.curtailment_penalty_weight * np.full(
                self.n_agents, np.maximum(-pg, 0.0), dtype=float
            )
            economic_cost = (
                actual_grid_purchase_cost
                + np.sum(generate_costs)
                + np.sum(bess_cost)
                + np.sum(env_cost)
                + np.sum(np.maximum(-pg, 0.0)) * self.curtailment_penalty_weight
            )
            economic_reward = -economic_cost
        else:
            buy_cost = -actual_grid_purchase_cost
            weighted_buy_cost = self.buy_cost_weight * buy_cost
            weighted_generate_costs = self.generate_cost_weight * (-generate_costs)
            weighted_bess_cost = self.bess_cost_weight * (-bess_cost)
            env_cost = pds * self.params["env_param"]
            weighted_curtailment_cost = -self.curtailment_penalty_weight * np.full(
                self.n_agents, np.maximum(-pg, 0.0), dtype=float
            )
            economic_cost = (
                actual_grid_purchase_cost
                + np.sum(generate_costs)
                + np.sum(bess_cost)
                + np.sum(env_cost)
                + np.sum(np.maximum(-pg, 0.0)) * self.curtailment_penalty_weight
            )
            economic_reward = -economic_cost

        env_reward = self.env_reward_weight * (-pds * self.params["env_param"])
        punishment = self.battery_punishment_weight * battery_punishment
        curtailment = np.full(self.n_agents, np.maximum(-pg, 0.0), dtype=float)
        curtailment_penalty = weighted_curtailment_cost
        soc_reserve_penalty = -self.soc_reserve_weight * (
            np.maximum(self.soc_reserve_target - socs, 0.0)
            + np.maximum(socs_raw - 0.8, 0.0)
        )
        eco_reward = (
            weighted_generate_costs
            + weighted_bess_cost
            + weighted_buy_cost
            + curtailment_penalty
        )

        # 每一个 Agent 算出来的单兵奖励列表
        if self.reward_mode == "milp_economic":
            reward_list = economic_reward + punishment + soc_reserve_penalty
        else:
            reward_list = eco_reward + env_reward + punishment + soc_reserve_penalty

        step_total_power_loss = float(np.abs(pg))
        step_grid_purchase_cost = float(actual_grid_purchase_cost)
        step_baseline_grid_purchase_cost = float(baseline_grid_purchase_cost)
        step_grid_purchase_saving = float(
            baseline_grid_purchase_cost - actual_grid_purchase_cost
        )
        step_soc_violation = float(
            np.sum(np.maximum(0.2 - socs_raw, 0) + np.maximum(socs_raw - 0.8, 0))
        )
        step_mean_soc = float(np.mean(self.params["socs"]))
        self._ep_steps += 1
        self._ep_total_power_loss += step_total_power_loss
        self._ep_grid_purchase_cost += step_grid_purchase_cost
        self._ep_baseline_grid_purchase_cost += step_baseline_grid_purchase_cost
        self._ep_grid_purchase_saving += step_grid_purchase_saving
        self._ep_soc_violation += step_soc_violation
        self._ep_mean_soc_sum += step_mean_soc
        self._ep_buy_cost_reward += float(np.sum(weighted_buy_cost))
        self._ep_generate_cost_reward += float(np.sum(weighted_generate_costs))
        self._ep_bess_cost_reward += float(np.sum(weighted_bess_cost))
        self._ep_battery_punishment_reward += float(np.sum(punishment))
        self._ep_battery_action_clip += float(np.sum(battery_action_clip))
        self._ep_curtailment_reward += float(np.sum(curtailment_penalty))
        self._ep_soc_reserve_reward += float(np.sum(soc_reserve_penalty))
        self._ep_economic_reward += float(np.sum(economic_reward))
        self._ep_economic_cost += float(np.sum(-economic_reward))
        self._ep_curtailment_cost += float(np.sum(-curtailment_penalty))

        # --- 记账保存逻辑（保留你原本写 CSV 的行为） ---
        for j, i in enumerate(self.possible_agents):
            self.ep_buffer.append(
                {
                    "episode": int(self.ep_counter),
                    "agent": i,
                    "row_idx": int(self.idx[0]),
                    "col_idx": int(self.idx[1]),
                    "price": float(self.data_dict["price"][tuple(self.idx)]),
                    "load_sd": float(self.data_dict[i]["load_data"][tuple(self.idx)]),
                    "pv_sd": float(self.data_dict[i]["pv_data"][tuple(self.idx)]),
                    "load": float(self.data_dict[i]["load"][tuple(self.idx)]),
                    "pv": float(self.data_dict[i]["pv"][tuple(self.idx)]),
                    "soc_init": float(last_socs[j]),
                    "soc": float(self.params["socs"][j]),
                    "action_pd": float(actions_raw[j, 0]),
                    "action_pb": float(actions_raw[j, 1]),
                    "pd": float(pds[j]),
                    "pb": float(pbs[j]),
                    "pg_total": float(pg),
                    "gen_cost": float(-weighted_generate_costs[j]),
                    "bess_cost": float(-weighted_bess_cost[j]),
                    "buy_cost": float(-weighted_buy_cost[j]),
                    "curtailment_penalty": float(curtailment_penalty[j]),
                    "economic_reward": float(economic_reward[j]),
                    "economic_cost": float(-economic_reward[j]),
                    "baseline_grid_purchase_cost": float(baseline_grid_purchase_cost),
                    "grid_purchase_saving": float(step_grid_purchase_saving),
                    "eco_reward": float(eco_reward[j]),
                    "env_reward": float(env_reward[j]),
                    "punishment": float(punishment[j]),
                    "soc_reserve_penalty": float(soc_reserve_penalty[j]),
                    "reward": float(reward_list[j]),
                }
            )
        # ======================================================================

        # 时钟前进一步
        self.idx[1] += 1
        terminated = False

        # 判断今天 24 小时（或其他步长上限）是否结束
        if self.idx[1] >= self.data_shape[1]:
            terminated = True
            self.idx[1] = 0  # 重置时钟指针供下一轮使用

            write_ok = False
            if self.log_trajectories and self.ep_buffer:
                if self.log_trajectories_mode == "all":
                    write_ok = True
                elif self.log_trajectories_mode == "test" and self.is_testing:
                    write_ok = True
                elif self.log_trajectories_mode == "train" and (not self.is_testing):
                    write_ok = True

            if write_ok:
                file_path = self._current_csv_path()
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_exists = file_path.exists()
                with open(
                    file_path, "a" if file_exists else "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(self.ep_buffer[0].keys())
                    )
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(self.ep_buffer)

        # --- 关键改动：组装 EPyMARL 框架绝对需要的标准返回格式 ---

        # 1. 团队总奖励：框架（如QMIX）默认所有 Agent 协同共享一个标量奖励。
        #    这里支持 sum/mean 两种聚合方式，并支持 reward_scale 缩放，便于稳定训练。
        if self.reward_aggregate == "mean":
            total_reward = float(np.mean(reward_list))
        else:
            total_reward = float(np.sum(reward_list))
        total_reward *= self.reward_scale

        # 2. 打造环境附加信息包 env_info，供框架统计胜率和画网损图
        #    注意：这里加入 "episode_limit"，这样 runner.py 就能区分出是否是时间到了下班。
        env_info = {
            "total_power_loss": float(self._ep_total_power_loss),
            "mean_soc": float(self._ep_mean_soc_sum / max(self._ep_steps, 1)),
            "soc_violation": float(self._ep_soc_violation),
            "grid_purchase_cost": float(self._ep_grid_purchase_cost),
            "baseline_grid_purchase_cost": float(self._ep_baseline_grid_purchase_cost),
            "grid_purchase_saving": float(self._ep_grid_purchase_saving),
            "grid_purchase_saving_ratio": float(
                self._ep_grid_purchase_saving
                / max(self._ep_baseline_grid_purchase_cost, 1e-6)
            ),
            "economic_reward": float(self._ep_economic_reward),
            "economic_cost": float(self._ep_economic_cost),
            "curtailment_cost": float(self._ep_curtailment_cost),
            "buy_cost_reward": float(self._ep_buy_cost_reward),
            "generate_cost_reward": float(self._ep_generate_cost_reward),
            "bess_cost_reward": float(self._ep_bess_cost_reward),
            "battery_punishment_reward": float(self._ep_battery_punishment_reward),
            "battery_action_clip": float(self._ep_battery_action_clip),
            "curtailment_reward": float(self._ep_curtailment_reward),
            "soc_reserve_reward": float(self._ep_soc_reserve_reward),
            "episode_limit": terminated,
        }

        # 严格按照这个顺序返回！
        # 框架的这一行接收它：reward, terminated, env_info = self.env.step(actions[0].cpu())
        return total_reward, terminated, env_info

    def render(self, mode="human"):
        self.render_mode = "human"

    def close(self):
        pass

    def _current_csv_path(self) -> pathlib.Path:
        bucket_idx = (max(self.ep_counter, 1) - 1) // self.bucket_size
        start_ep = bucket_idx * self.bucket_size + 1
        end_ep = (bucket_idx + 1) * self.bucket_size
        src = (
            "test"
            if self.test_source == "test"
            else ("train" if self.test_source == "train" else "all")
        )
        filename = f"episodes_{src}_{start_ep:06d}-{end_ep:06d}.csv"
        return self.render_dir / filename


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 [START] 成功进入独立测试流程...")
    print("=" * 50)

    # 手动制造一个符合你 YAML 格式的虚拟配置字典
    mock_config = {
        "result_dir": "./results",
        "data": [
            {
                "agent_name": "agent1",
                "split_ratio": 0.25,
                "dg_max": 2.0,
                "battery_cap": 5.0,
                "battery_punishment_param": 0.1,
                "raw_ch": 0.95,
                "raw_dis": 0.95,
                "bessa": 0.1,
                "bessb": 0.2,
                "bessc": 0.3,
                "costa": 0.05,
                "costb": 0.1,
                "costc": 0.2,
                "env_param": 0.01,
                "low_state": [0.0, 0.0, 0.2],
                "high_state": [10.0, 10.0, 0.8],
                "low_action": [-1.0, -1.0],
                "high_action": [1.0, 1.0],
            },
            {
                "agent_name": "agent2",
                "split_ratio": 0.25,
                "dg_max": 2.0,
                "battery_cap": 5.0,
                "battery_punishment_param": 0.1,
                "raw_ch": 0.95,
                "raw_dis": 0.95,
                "bessa": 0.1,
                "bessb": 0.2,
                "bessc": 0.3,
                "costa": 0.05,
                "costb": 0.1,
                "costc": 0.2,
                "env_param": 0.01,
                "low_state": [0.0, 0.0, 0.2],
                "high_state": [10.0, 10.0, 0.8],
                "low_action": [-1.0, -1.0],
                "high_action": [1.0, 1.0],
            },
        ],
    }

    # 彻底不用 try-except，让任何一丁点错误直接原形毕露砸在控制台上！
    print("[STEP 1] 正在尝试读取 CSV 并初始化 MGEnv...")
    env = MGEnv(config=mock_config, algo_name="debug_run")
    print("✅ 环境构建成功！")

    print("\n[STEP 2] 正在测试 get_env_info()...")
    env_info = env.get_env_info()
    print(f"✅ 环境信息: {env_info}")

    print("\n[STEP 3] 正在测试 reset()...")
    obs = env.reset()
    print(f"✅ Reset 成功！初始观察矩阵维度: {np.array(obs).shape}")

    print("\n[STEP 4] 正在模拟跑一步 step()...")
    # 模拟两个智能体的连续动作 [n_agents, n_actions]
    mock_actions = np.array([[0.5, -0.2], [0.1, 0.8]], dtype=np.float32)
    total_reward, terminated, env_info = env.step(mock_actions)
    print(f"✅ Step 成功！团队总奖励: {total_reward}, 是否结束: {terminated}")
    print(f"✅ 附加网损信息: {env_info}")

    print("\n" + "=" * 50)
    print("🎉 [SUCCESS] 你的微电网环境所有核心接口完全正常！")
    print("=" * 50 + "\n")
```

## 二、整体职责与调用链

`MGEnv` 是一个 PettingZoo `ParallelEnv` 子类，但为了匹配本项目的 EpisodeRunner，它使用简化接口：`reset()` 返回观测列表，`step(actions)` 返回 `(团队奖励, 是否终止, 统计字典)`。训练时的数据流是：

```text
microgrid.yaml 的 env_args
  -> MGEnv.__init__()
  -> 读取 price/load/pv.csv，构建动作与观测空间
  -> reset() 选择一天并初始化 SOC
  -> step(actions) 将策略动作转换成柴油机/电池功率
  -> 更新 SOC、计算购电与设备成本、合成团队奖励
  -> EpisodeRunner 写入 EpisodeBatch，学习器使用该批数据训练
```

此环境把 CSV 的“行”视为一个日样本，把“列”视为日内时间步。因此 `idx = [row_index, time_index]`，并且 `episode_limit = price_data.shape[1]`。

## 三、导入、函数与类定义

### 第 1--15 行：依赖与标准化函数

- `import csv` 用于可选的轨迹 CSV 输出；`pathlib.Path` 使用跨平台路径拼接；`functools.cache` 缓存动作/观测空间查询。
- `numpy as np` 是环境数值计算主体。Gymnasium 的 `spaces.Box` 表示连续上下界；`ParallelEnv` 提供多智能体环境协议。
- `def standardize_data(...)` 定义函数；参数标注 `data: np.ndarray` 和返回标注 `-> np.ndarray` 是类型提示，不会在运行时自动做检查。
- `mean=None, std=None` 表示调用者可传入训练集统计量。若标准差近似零，`np.isclose` 分支只做中心化，避免除零；其余情况返回 z-score `(data - mean) / std`。

### 第 18--31 行：`MGEnv` 与初始化入口

- `class MGEnv(ParallelEnv):` 是继承语法，环境继承 PettingZoo 的并行环境类型。
- 类变量 `metadata` 是框架元数据，声明可渲染模式和环境名。
- `__init__` 是构造器；每创建一个环境实例时执行一次。参数 `config` 对应 YAML 的 `env_args`，`algo_name` 用于轨迹目录，`render_mode` 仅保存供渲染接口使用。

## 四、构造器：配置、空间与数据

### 智能体与动作（约第 32--68 行）

- `len(config["data"])` 确定智能体数。这里 `config["data"]` 是 YAML 列表。
- 列表推导式 `["agent" + str(r + 1) for r in range(...)]` 生成 `agent1`、`agent2` 等固定 ID。
- 字典推导式 `{agent.pop("agent_name"): agent ...}` 用 `agent_name` 建索引。注意 `pop` 会从传入的每个配置字典中删除 `agent_name`，所以同一个 `config` 对象不应再次用于新环境。
- `np.stack(..., axis=0)` 把各智能体的二维动作上下界堆叠为形状 `[n_agents, 2]` 的数组；`n_actions` 从第二维得到。
- `config.get(key, default)` 在字段缺失时使用默认值；`float(...)` 强制奖励权重为标量浮点数。以 `_ep_` 开头的成员是累计到当前 episode 的指标，不参与状态转移。

### 外部数据与 Gymnasium 空间（约第 89--128 行）

- `Path(__file__).parent / "data"` 以本源码目录为锚点定位 `src/envs/data`，而非依赖当前工作目录。
- `read_csv(..., header=None)` 读取无表头 CSV，并由 `np.array` 转为 NumPy 数组。三个矩阵的形状必须相容：行是可选日样本，列是时间步。
- 观测空间通过 `np.concatenate` 在 YAML 的三项 `low_state/high_state` 后添加价格和归一化时间；所以实际每个 agent 的观测是 5 维，不是注释所称的 3 维。
- `spaces.Box` 声明连续空间边界。动作空间直接采用该 agent 的两项动作下界/上界。

### 训练/测试切分、标准化和运行目录（约第 129--180 行）

- `np.random.default_rng(seed)` 创建局部随机数生成器，不依赖 NumPy 全局随机状态。
- `permutation(num_rows)` 将日样本随机打乱，前 80% 是 `train_indices`，其余是 `test_indices`。
- 负荷与光伏的标准化统计量仅从训练行求得，再应用到全部数据，避免测试数据泄露进训练标准化。
- `data_dict["load_pv"] = load_data - pv_data` 是全局净负荷；每个 agent 的负荷、光伏和标准化值都按 `split_ratio` 分摊。
- `render_dir.mkdir(parents=True, exist_ok=True)` 创建轨迹输出目录；`parents=True` 同时创建不存在的父目录，`exist_ok=True` 允许目录已存在。
- `bucket_size = 10_000` 用于将轨迹 CSV 每 1 万 episode 分桶，`episode_limit` 是一天的时间步数。

## 五、环境接口和状态表示

### `get_env_info`、`get_obs`、`get_state`（约第 182--224 行）

- `get_env_info()` 返回训练框架建网络和缓冲区所需的结构描述。全局状态维度是全部局部观测的串接，即 `obs_shape * n_agents`。
- `get_obs()` 使用列表推导式为每个 agent 构建 `float32` 五维数组：标准化负荷、标准化光伏、该 agent 的 SOC、原始电价、`time_frac`。
- `time_frac = time_index / max(episode_limit - 1, 1)` 将日内位置映射到 `[0, 1]`，并在只有一个时间步时避免除零。
- `get_state()` 使用 `np.concatenate` 按 agent 顺序把局部观测拼成集中式 critic 使用的全局状态。

### `params_init`、空间查询（约第 226--290 行）

- `params_init` 用列表推导式从配置提取每个 agent 的额定功率、效率和成本系数，建立逐 agent 的 NumPy 向量。向量化保证后续 `pds`、`pbs`、SOC 和成本可同时计算。
- `np.full(self.n_agents, 0.2)` 设置构造时的初始 SOC；每个 episode 的真实初值由 `reset` 覆盖。
- `@functools.cache` 是装饰器：对相同 agent 参数的空间查询缓存返回值。方法先将整数转成 `"agent" + str(agent)`，所以调用者需传 1、2、3，而不是零基索引。

### `reset` 与 `reset_test`（约第 292--361 行）

- 两个方法都会选取一行数据、将时间列重置为 0、将 SOC 均匀随机初始化在 `[0.2, 0.5)`，并清空 episode 累计指标。
- `reset` 固定从训练行采样；`reset_test(source="test")` 可从测试行、训练行或所有行采样。
- 若传入 `seed`，方法创建临时 RNG，因此这一次的选日和 SOC 可复现；否则复用环境 RNG。
- 两者最后直接返回 `get_obs()`。这是本项目 Runner 的接口约定，并非 PettingZoo 常见的 `(observations, infos)` 返回格式。

## 六、一步状态转移与电力物理意义

### 动作预处理和 SOC 限制（约第 363--410 行）

- 输入动作若是 PyTorch tensor，`hasattr(actions, "cpu")` 分支先移到 CPU 并转 NumPy；随后 `reshape(n_agents, n_actions)` 固定其二维形状。
- `np.clip(actions_raw, action_lows, action_highs)` 逐元素裁剪策略输出，保证环境从不执行越界动作。
- `pds = a_dg * dg_max`：第 0 维动作是柴油机出力比例。由于 YAML 下界为 0，它不能反向发电。
- `pbs_requested = a_batt * battery_caps`：第 1 维为电池功率。负值表示充电、正值表示放电。
- `np.where(pbs_requested < 0, ...)` 根据充/放电效率计算 SOC 增量：充电时 `-raw_ch * pb / capacity`，负的 `pb` 会提高 SOC；放电时的正 `pb` 会降低 SOC。
- 候选 SOC 低于 0.2 或高于 0.8 时，`pb_adj` 调整为恰好到达边界的可执行功率。请求和可执行功率的绝对差 `battery_action_clip` 乘 `battery_punishment_param` 形成负惩罚。
- 第二次计算 `delta_soc` 采用实际 `pbs`，随后 `np.clip` 作为数值安全边界。

### 功率平衡与成本（约第 412--491 行）

- `pg = load - pv - sum(pds) - sum(pbs)` 是聚合电网功率：正值表示从主网购电，负值表示本地供给过剩。
- 电池成本使用 `abs(a*x**2 + b*x + c)`，其中 `x = pb + 3*capacity*(1-soc)`；`**2` 是幂运算。
- 柴油成本是每 agent 的二次函数 `costa*pd**2 + costb*pd + costc`。
- 基准购电成本假设没有本地柴油机和电池调度；实际购电成本只对 `max(pg, 0)` 收费，环境没有把负 `pg` 作为卖电收入。
- `reward_mode == "milp_economic"` 是当前 YAML 的分支：实际购电成本按 `split_ratio` 分给各 agent，同时加入发电、电池、环境和弃电成本，`economic_reward = -economic_cost`。
- 其他两个分支是旧式奖励：`relative_saving` 以相对购电节约作为信号，默认分支直接以购电成本的相反数作为奖励。它们仍会加权发电/电池/环境分量。

### 奖励、统计、轨迹（约第 493--573 行）

- SOC 储备惩罚计算目标 SOC 以下的缺口，并额外惩罚未裁剪候选 SOC 超过 0.8 的量；权重由 YAML 控制。
- 当前 `milp_economic` 模式下，单 agent 奖励为 `economic_reward + battery clip punishment + SOC reserve penalty`；不额外加 `env_reward`，因为环境成本已包含在 `economic_cost` 中。
- 指标值以 `float(...)` 转为 Python 标量，便于日志系统/CSV 序列化。以 `+=` 累积的值是本 episode 的总量。
- `ep_buffer.append({...})` 给每个 agent、每个时间步保存一条可选轨迹记录；字典键最终成为 CSV 表头。

### 终止、团队奖励和返回值（约第 575--650 行）

- 每步后 `idx[1] += 1` 推进时间。达到列数时设置 `terminated=True`，再将列索引置零供下次 reset 前的内部状态保持一致。
- 仅当 `log_trajectories` 开启且模式匹配训练/测试 episode 时，使用 `csv.DictWriter` 追加轨迹。文件不存在时先写表头。
- 多智能体环境先得到 `reward_list`，再由 `reward_aggregate` 求平均或求和，并乘 `reward_scale`，最终只返回一个团队标量奖励。
- `env_info` 是累计指标字典。这里的 `"episode_limit": terminated` 告诉 Runner 这次终止是否由时间上限导致。
- 返回顺序必须是 `total_reward, terminated, env_info`；Runner 对该三元组有固定解包顺序。

## 七、其余接口与独立测试

- `render` 当前只记录模式，没有实现图形渲染；`close` 是空实现，满足环境生命周期接口。
- `_current_csv_path` 通过整除 `//` 算出 episode 所在的一万集桶，生成稳定的输出文件名。
- `if __name__ == "__main__":` 是仅在直接执行此文件时成立的模块保护。它构造两 agent 的 `mock_config`，依次验证初始化、环境信息、重置和一步交互；被 `direct_main.py` 导入时不会执行。

