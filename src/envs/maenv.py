import csv
import functools
import pathlib

import numpy as np
import torch
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
        self.buy_cost_weight = float(config.get("buy_cost_weight", 1.0))
        self.buy_reward_mode = config.get("buy_reward_mode", "raw_cost")
        self.buy_reward_scale = float(config.get("buy_reward_scale", 100.0))
        self.generate_cost_weight = float(config.get("generate_cost_weight", 1.0))
        self.bess_cost_weight = float(config.get("bess_cost_weight", 1.0))
        self.env_reward_weight = float(config.get("env_reward_weight", 1.0))
        self.battery_punishment_weight = float(config.get("battery_punishment_weight", 1.0))
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
        self._ep_soc_reserve_reward = 0.0

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
            obs_low = np.concatenate([base_low, np.array([price_min, 0.0], dtype=np.float32)])
            obs_high = np.concatenate([base_high, np.array([price_max, 1.0], dtype=np.float32)])
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
        load_sd = standardize_data(load_data, mean=load_train.mean(), std=load_train.std())

        self.data_dict = {"price": price_data, "load_pv": load_data - pv_data}
        for key in agents_config.keys():
            self.data_dict[key] = {}
            self.data_dict[key]["pv_data"] = pv_sd * agents_config[key]["split_ratio"]
            self.data_dict[key]["load_data"] = load_sd * agents_config[key]["split_ratio"]
            self.data_dict[key]["load"] = load_data * agents_config[key]["split_ratio"]
            self.data_dict[key]["pv"] = pv_data * agents_config[key]["split_ratio"]

        run_id = config.get("run_id", "no_run_id")
        self.log_trajectories = bool(config.get("log_trajectories", False))
        self.log_trajectories_mode = config.get("log_trajectories_mode", "test")
        self.render_dir = pathlib.Path(config["result_dir"]) / "microgrid_trajs" / str(algo_name) / str(run_id)
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
            "n_agents": self.n_agents,          # 智能体数量
            "n_actions": action_shape,          # 每个智能体的动作维数
            "obs_shape": obs_shape,            # 每个智能体的观察维数
            "state_shape": obs_shape * self.n_agents, # 全局状态维数（所有智能体obs拼接）
            "episode_limit": self.data_shape[1], # 一局的最大时序步数（你表格里的总列数，比如24）
            "action_spaces": [self.spaces[agent]["action"] for agent in self.possible_agents],
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }
    def get_obs(self):
        """返回一个长度为 n_agents 的列表，每个元素是对应 Agent 的 NumPy 数组"""
        time_frac = self.idx[1] / max(self.episode_limit - 1, 1)
        price = float(self.data_dict["price"][tuple(self.idx)])
        return [
            np.array([
                self.data_dict[agent_id]["load_data"][tuple(self.idx)],
                self.data_dict[agent_id]["pv_data"][tuple(self.idx)],
                self.params["socs"][i],
                price,
                time_frac,
            ], dtype=np.float32)
            for i, agent_id in enumerate(self.possible_agents)
        ]
    def get_state(self):
        """EPyMARL 诸如 QMIX 算法强烈依赖全局状态，这里直接拼接所有人的观察"""
        return np.concatenate(self.get_obs())
    def get_avail_actions(self):
        """电网动作通常是连续的（Box），全返回 1 表示所有动作随时可用"""
        return [[1] * self.get_env_info()["n_actions"] for _ in range(self.n_agents)]
    
    

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
        self._ep_soc_reserve_reward = 0.0
        
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
        self._ep_soc_reserve_reward = 0.0
        return self.get_obs()

    def step(self, actions):
        """
        EPyMARL框架传入连续动作，直接映射到发电功率和电池功率
        """
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        actions_raw = np.asarray(actions, dtype=np.float32).reshape(self.n_agents, self.n_actions)
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
        pb_adj[clamped_high] = (last_socs[clamped_high] - 0.8) * battery_caps[clamped_high]

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
        bess_poly = -np.abs(
            self.params["bessa"] * x**2 + self.params["bessb"] * x + self.params["bessc"]
        )

        generate_costs = -(
            self.params["costa"] * pds**2 + self.params["costb"] * pds + self.params["costc"]
        )
        baseline_pg = max(float(self.data_dict["load_pv"][tuple(self.idx)]), 0.0)
        baseline_grid_purchase_cost = baseline_pg * self.data_dict["price"][tuple(self.idx)]
        actual_grid_purchase_cost = max(float(pg), 0.0) * self.data_dict["price"][tuple(self.idx)]
        if self.buy_reward_mode == "relative_saving":
            buy_signal = self.buy_reward_scale * (
                (baseline_grid_purchase_cost - actual_grid_purchase_cost)
                / max(baseline_grid_purchase_cost, 1e-6)
            )
            buy_cost = float(buy_signal)
        else:
            buy_cost = -actual_grid_purchase_cost
        weighted_buy_cost = self.buy_cost_weight * buy_cost
        weighted_generate_costs = self.generate_cost_weight * generate_costs
        weighted_bess_cost = self.bess_cost_weight * bess_poly
        env_reward = self.env_reward_weight * (-pds * self.params["env_param"])
        punishment = self.battery_punishment_weight * battery_punishment
        soc_reserve_penalty = -self.soc_reserve_weight * (
            np.maximum(self.soc_reserve_target - socs, 0.0) + np.maximum(socs_raw - 0.8, 0.0)
        )
        eco_reward = weighted_generate_costs + weighted_bess_cost + weighted_buy_cost
        
        # 每一个 Agent 算出来的单兵奖励列表
        reward_list = eco_reward + env_reward + punishment + soc_reserve_penalty

        step_total_power_loss = float(np.abs(pg))
        step_grid_purchase_cost = float(actual_grid_purchase_cost)
        step_baseline_grid_purchase_cost = float(baseline_grid_purchase_cost)
        step_grid_purchase_saving = float(baseline_grid_purchase_cost - actual_grid_purchase_cost)
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
        self._ep_soc_reserve_reward += float(np.sum(soc_reserve_penalty))
        
        # --- 记账保存逻辑（保留你原本写 CSV 的行为） ---
        for j, i in enumerate(self.possible_agents):
            self.ep_buffer.append({
                "episode": int(self.ep_counter), "agent": i, "row_idx": int(self.idx[0]), "col_idx": int(self.idx[1]),
                "price": float(self.data_dict["price"][tuple(self.idx)]), "load_sd": float(self.data_dict[i]["load_data"][tuple(self.idx)]),
                "pv_sd": float(self.data_dict[i]["pv_data"][tuple(self.idx)]), "load": float(self.data_dict[i]["load"][tuple(self.idx)]),
                "pv": float(self.data_dict[i]["pv"][tuple(self.idx)]), "soc_init": float(last_socs[j]), "soc": float(self.params["socs"][j]),
                "action_pd": float(actions_raw[j, 0]), "action_pb": float(actions_raw[j, 1]), "pd": float(pds[j]), "pb": float(pbs[j]),
                "pg_total": float(pg), "gen_cost": float(weighted_generate_costs[j]), "bess_cost": float(weighted_bess_cost[j]), "buy_cost": float(weighted_buy_cost),
                "baseline_grid_purchase_cost": float(baseline_grid_purchase_cost), "grid_purchase_saving": float(step_grid_purchase_saving),
                "eco_reward": float(eco_reward[j]), "env_reward": float(env_reward[j]), "punishment": float(punishment[j]),
                "soc_reserve_penalty": float(soc_reserve_penalty[j]), "reward": float(reward_list[j]),
            })
        # ======================================================================

        # 时钟前进一步
        self.idx[1] += 1
        terminated = False
        
        # 判断今天 24 小时（或其他步长上限）是否结束
        if self.idx[1] >= self.data_shape[1]:
            terminated = True
            self.idx[1] = 0 # 重置时钟指针供下一轮使用
            
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
                with open(file_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(self.ep_buffer[0].keys()))
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
                self._ep_grid_purchase_saving / max(self._ep_baseline_grid_purchase_cost, 1e-6)
            ),
            "buy_cost_reward": float(self._ep_buy_cost_reward),
            "generate_cost_reward": float(self._ep_generate_cost_reward),
            "bess_cost_reward": float(self._ep_bess_cost_reward),
            "battery_punishment_reward": float(self._ep_battery_punishment_reward),
            "battery_action_clip": float(self._ep_battery_action_clip),
            "soc_reserve_reward": float(self._ep_soc_reserve_reward),
            "episode_limit": terminated,
            # 设置一个虚拟胜负：只要全天没有任何一个电池 SoC 低于 0.2 违规，就记 battle_won=1（赢了）
            "battle_won": 1.0 if np.all(self.params["socs"] >= 0.2) else 0.0 
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
    import sys
    print("\n" + "="*50)
    print("🚀 [START] 成功进入独立测试流程...")
    print("="*50)
    
    # 手动制造一个符合你 YAML 格式的虚拟配置字典
    mock_config = {
        "result_dir": "./results",
        "data": [
            {
                "agent_name": "agent1", "split_ratio": 0.25, "dg_max": 2.0, "battery_cap": 5.0,
                "battery_punishment_param": 0.1, "raw_ch": 0.95, "raw_dis": 0.95,
                "bessa": 0.1, "bessb": 0.2, "bessc": 0.3, "costa": 0.05, "costb": 0.1, "costc": 0.2, "env_param": 0.01,
                "low_state": [0.0, 0.0, 0.2], "high_state": [10.0, 10.0, 0.8], "low_action": [-1.0, -1.0], "high_action": [1.0, 1.0]
            },
            {
                "agent_name": "agent2", "split_ratio": 0.25, "dg_max": 2.0, "battery_cap": 5.0,
                "battery_punishment_param": 0.1, "raw_ch": 0.95, "raw_dis": 0.95,
                "bessa": 0.1, "bessb": 0.2, "bessc": 0.3, "costa": 0.05, "costb": 0.1, "costc": 0.2, "env_param": 0.01,
                "low_state": [0.0, 0.0, 0.2], "high_state": [10.0, 10.0, 0.8], "low_action": [-1.0, -1.0], "high_action": [1.0, 1.0]
            }
        ]
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
    
    print("\n" + "="*50)
    print("🎉 [SUCCESS] 你的微电网环境所有核心接口完全正常！")
    print("="*50 + "\n")
