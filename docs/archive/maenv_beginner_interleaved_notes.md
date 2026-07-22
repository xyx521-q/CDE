# maenv.py 小白逐段注释版

> 对照原文件：`src/envs/maenv.py`。这是单独新建的学习文档，不会被训练程序导入，也不会改动任何源代码。
>
> 阅读方法：每一段先看“原样代码”，紧接着看解释。所有原文件代码按原有顺序完整保留；解释用白话说明 Python 语法、变量含义和电网逻辑。

## 先建立整体画面

这个环境把一行 CSV 当作“一天”，把这一行里的每一列当作一天中的一个时间步。每一步，三个智能体同时决定：柴油机开多少（动作第 0 维）、电池充还是放多少（动作第 1 维）。环境据此更新电池 SOC，算出还需从大电网买多少电，以及所有成本和奖励。

```text
YAML 配置 + load/pv/price CSV
        -> 创建 MGEnv
        -> reset(): 随机选一天，初始化 SOC
        -> step(actions): 动作转成功率，更新 SOC，计算成本/奖励
        -> 返回团队奖励、结束标志、统计指标
```

## 第 1 段：导入库与数据标准化函数

**原样代码（原文件第 1--18 行）**

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

```

**小白解释**

这一段先准备工具。<code>import</code> 的意思是“把别的库拿进来使用”：<code>csv</code> 负责写轨迹表，<code>pathlib</code> 负责拼文件路径，<code>numpy</code> 负责矩阵和数组计算，Gymnasium/PettingZoo 则定义强化学习环境的“接口规范”。<br><br><code>standardize_data</code> 把原始负荷或光伏数据变成标准化数据：每个值减去平均值，再除以标准差。这样神经网络看到的输入量级更接近，通常更容易训练。<code>mean is None</code> 的意思是“调用时没有提供均值”；<code>np.isclose(std, 0)</code> 是在防止标准差为零时除零。<br><br>这里的 <code>data: np.ndarray</code> 和 <code>-> np.ndarray</code> 是类型提示，帮助阅读者和编辑器理解输入/输出是 NumPy 数组；它们不改变函数本身的计算。

## 第 2 段：定义环境并读取基础配置

**原样代码（原文件第 20--88 行）**

```python
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
```

**小白解释**

<code>class MGEnv(ParallelEnv)</code> 定义名为 <code>MGEnv</code> 的微电网环境；圆括号表示它继承 PettingZoo 的并行多智能体环境规范。<code>metadata</code> 是环境说明信息。<br><br><code>__init__</code> 是构造函数：训练程序创建环境时会自动执行。<code>config</code> 就是 YAML 中的 <code>env_args</code>。<code>self.</code> 开头的变量会保存到这个环境对象里，后续 <code>reset</code>、<code>step</code> 都能使用。<br><br><code>possible_agents</code> 根据配置数量自动得到 <code>agent1</code>、<code>agent2</code>、<code>agent3</code>。字典推导式 <code>{... for ...}</code> 把列表转换成按名称查询的字典。注意：<code>pop("agent_name")</code> 会从传入的配置字典里删除 <code>agent_name</code>，因此不能把同一个 <code>config</code> 对象不加复制地重复传给新环境。<br><br><code>np.stack(..., axis=0)</code> 将每个智能体的动作上下界叠成二维数组，形状是 <code>[智能体数, 动作维度]</code>。这一项目中动作有两维：柴油机比例和电池充/放电比例。后半部分将 YAML 的奖励字段保存为浮点数；<code>config.get("字段", 默认值)</code> 表示 YAML 没写该字段时用默认值。以 <code>_ep_</code> 开头的变量是“当前 episode 的累计统计量”。

## 第 3 段：读取 CSV、声明观测/动作空间、划分训练测试集

**原样代码（原文件第 90--170 行）**

```python
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

```

**小白解释**

<code>Path(__file__).parent</code> 找到当前 <code>maenv.py</code> 所在目录；再用 <code>/ "data"</code> 拼出 <code>src/envs/data</code>，所以程序不会依赖你从哪个目录启动。<br><br>三个 <code>read_csv</code> 分别读电价、光伏、负荷。每个 CSV 的“行”是一条可供选择的日样本，“列”是这一天的连续时间步。它们的形状必须匹配。<br><br><code>spaces.Box</code> 表示连续数值空间。YAML 提供的 <code>low_state/high_state</code> 只有负荷、光伏、SOC 三项；代码额外拼上电价和时间比例，所以实际观测是 5 维。<code>dtype=np.float32</code> 指明单精度浮点格式，深度学习中常用。<br><br><code>permutation</code> 随机打乱“天”的行编号，前 80% 用于训练，后 20% 用于测试。标准化均值和标准差只由训练行计算，再作用到所有行，避免测试数据提前泄露给训练。<code>load_pv = load - pv</code> 是整个微电网的净负荷；每个 agent 的数据按 <code>split_ratio</code> 分摊。

## 第 4 段：训练框架看到的环境信息、局部观测和全局状态

**原样代码（原文件第 172--223 行）**

```python
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
```

**小白解释**

<code>get_env_info()</code> 向训练框架说明数据形状：有多少 agent、每个动作多少维、每个观测多少维，以及一个 episode 有多少步。<code>state_shape = obs_shape * n_agents</code> 表示“全局状态”由所有智能体的局部观测直接拼接而成。<br><br><code>get_obs()</code> 返回一个 Python 列表，列表中每个元素对应一个智能体。每个元素含 5 个值，顺序非常重要：标准化负荷、标准化光伏、该智能体 SOC、当前原始电价、日内时间比例。<br><br><code>tuple(self.idx)</code> 把列表 <code>[行, 列]</code> 变为元组，以便 NumPy 用二维索引取当前时刻数据。<code>enumerate</code> 同时给出序号 <code>i</code> 和 agent 名称。<code>get_state()</code> 调用 <code>np.concatenate</code>，把三个局部观测按顺序拼成一个集中式状态向量。

## 第 5 段：把 YAML 的逐智能体参数变成可并行计算的数组

**原样代码（原文件第 225--292 行）**

```python
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
```

**小白解释**

<code>params_init</code> 接收按 agent 名称索引的配置字典，返回一个新的字典 <code>self.params</code>。每个键对应一个长度为 <code>n_agents</code> 的数组，例如 <code>dg_max</code> 可能是 <code>[80, 80, 80]</code>。<br><br>方括号中的 <code>[agents_config[a]["dg_max"] for a in self.possible_agents]</code> 是列表推导式：按 agent 顺序取出所有人的同一字段。随后 <code>np.array</code> 使它能与动作向量逐元素同时运算，而不需要写三次相同的计算。<br><br><code>socs</code> 是当前电池荷电状态，构造时设为 0.2；每次 reset 会重新随机初始化。<br><br><code>@functools.cache</code> 是装饰器，表示相同参数的 <code>observation_space</code>/<code>action_space</code> 查询可直接复用先前结果。这里的 <code>agent: int</code> 要传 1、2、3，函数内部将它转为 <code>"agent1"</code> 等名称。

## 第 6 段：开始一局训练或测试

**原样代码（原文件第 294--361 行）**

```python
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
```

**小白解释**

<code>reset()</code> 是训练开始一局时调用的方法。三元表达式 <code>A if 条件 else B</code> 的含义是：如果调用者传了种子，临时创建一个可复现的随机数生成器；否则使用环境自己的随机数生成器。<br><br><code>rng.choice(self.train_indices)</code> 随机选训练集中的一天，<code>self.idx = [row_index, 0]</code> 将时钟定位到这一天的第一列。SOC 在 0.2 到 0.5 之间随机初始化；随后清空轨迹和所有 episode 累计量。<br><br><code>reset_test()</code> 逻辑类似，但默认从测试行选择一天。<code>source</code> 是 <code>"test"</code> 时选测试集、是 <code>"train"</code> 时选训练集，其他值会从全部行随机选。<br><br>两者最后直接返回 <code>self.get_obs()</code>。这是本项目 Runner 要求的简化格式，不能随意替换成一些 Gymnasium 示例中的 <code>(obs, info)</code>。

## 第 7 段：接收动作、换算功率并约束 SOC

**原样代码（原文件第 363--412 行）**

```python
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
```

**小白解释**

<code>step(actions)</code> 是环境每走一个时间步调用的核心函数。训练器可能传 PyTorch Tensor；如果对象有 <code>cpu</code> 方法，就先移到 CPU 再转成 NumPy。<code>reshape(n_agents, n_actions)</code> 把输入固定为“每个 agent 一行、每个动作一列”。<br><br><code>np.clip</code> 将任何越界动作压回 YAML 声明的范围。第 0 个动作 <code>actions_raw[:, 0]</code> 乘 <code>dg_max</code> 得柴油机功率 <code>pds</code>；第 1 个动作乘电池容量得到请求电池功率 <code>pbs_requested</code>。本代码中负电池功率代表充电，正值代表放电。<br><br><code>np.where(条件, 值1, 值2)</code> 是逐元素条件选择：按充/放电效率计算 SOC 改变量。先得到“如果完全执行请求动作”的 <code>socs_raw</code>，再发现低于 0.2 或高于 0.8 的 agent，修改对应的实际功率 <code>pb_adj</code>，让 SOC 恰好停在边界。<br><br>最后 <code>battery_action_clip</code> 是请求功率与实际功率的差；越界的动作没有被静默接受，而会受到负惩罚。第二次 SOC 计算使用的是已调整的真实功率。

## 第 8 段：功率平衡、成本和奖励模式分支

**原样代码（原文件第 413--485 行）**

```python
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
```

**小白解释**

<code>pg</code> 是总电网交换功率：<code>净负荷 - 所有柴油发电 - 所有电池放电</code>。<code>pg &gt; 0</code> 表示本地供给不足，需要从大电网买电；<code>pg &lt; 0</code> 表示本地供给过剩。当前实现不把过剩视作售电收入，而会用于弃电惩罚。<br><br><code>bess_cost</code> 和 <code>generate_costs</code> 分别是电池成本和柴油机成本。<code>**2</code> 表示平方，因而两者都含二次成本项。<code>baseline_grid_purchase_cost</code> 是“不调度柴油机和电池”时的购电基准；<code>actual_grid_purchase_cost</code> 是执行动作后的实际购电成本。<br><br>当前 YAML 的 <code>reward_mode: "milp_economic"</code> 会进入第一个 <code>if</code> 分支。<code>split_weights</code> 将共享购电和弃电成本按 <code>split_ratio</code> 分给各 agent，且用 <code>max(sum, 1e-6)</code> 防止除以零。每个 agent 的经济成本由购电分摊、柴油成本、电池成本、环境成本和弃电成本相加；经济奖励就是成本取负。<br><br><code>elif</code> 和 <code>else</code> 是保留的旧式奖励路径：一种按相对购电节省给信号，另一种直接把购电成本取负。只要 <code>reward_mode</code> 保持 <code>milp_economic</code>，训练不会走到它们。

## 第 9 段：合成单智能体奖励并累计指标

**原样代码（原文件第 487--539 行）**

```python
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
```

**小白解释**

<code>env_reward</code> 是旧式路径的环境惩罚；当前经济模式已经把 <code>env_cost</code> 放入经济成本，所以最终奖励不会额外使用它。<br><br><code>soc_reserve_penalty</code> 不是硬边界惩罚：SOC 虽然允许在 0.2 到 0.8 内，但如果低于 YAML 的储备目标 0.35，还会产生额外负奖励。这样策略会倾向留出应急电量。<br><br>当前模式下 <code>reward_list = economic_reward + punishment + soc_reserve_penalty</code>，它是长度为 agent 数的数组。每一名 agent 都有自己的经济成本份额和电池/SOC 惩罚。<br><br>下方 <code>step_</code> 变量计算当前一步的统计量；<code>+=</code> 表示在当前 episode 总计中继续加。它们主要用于日志和结果分析，通常不影响下一步的电网物理状态。

## 第 10 段：为可选 CSV 轨迹记录逐智能体明细

**原样代码（原文件第 541--573 行）**

```python
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

```

**小白解释**

<code>for j, i in enumerate(self.possible_agents)</code> 对每个 agent 生成一条记录。<code>j</code> 是 0、1、2 这样的数组下标，<code>i</code> 是 <code>"agent1"</code> 等名称。<br><br><code>self.ep_buffer.append({...})</code> 将一个字典加入列表。字典的键将来成为 CSV 的列名，值是本时刻的输入数据、动作、实际功率、SOC、分项成本与最终奖励。<br><br>大量 <code>float(...)</code>/<code>int(...)</code> 转换是为了把 NumPy 标量改为普通 Python 数值，方便 <code>csv</code> 和日志系统写入。这段仅记录数据；是否真的写文件要等 episode 结束时由 <code>log_trajectories</code> 决定。

## 第 11 段：结束判断、写轨迹、团队奖励与返回值

**原样代码（原文件第 575--650 行）**

```python
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
```

**小白解释**

每次 <code>step</code> 后，<code>self.idx[1] += 1</code> 将日内时间列前进一格。到达 CSV 的最后一列时 <code>terminated=True</code>，说明这个“一天”的 episode 结束。<br><br>如果 YAML 打开 <code>log_trajectories</code> 且记录模式与当前训练/测试来源相符，代码会用 <code>csv.DictWriter</code> 写入缓存。<code>"a" if file_exists else "w"</code> 意为：文件已存在就追加，否则新建；首次创建时还会写表头。<br><br>训练器接收的是一个团队标量奖励，不是每个 agent 的奖励数组。因此 <code>reward_aggregate == "mean"</code> 时取平均，否则求和，再乘 <code>reward_scale</code>。当前 YAML 使用 <code>sum</code> 和 1.0。<br><br><code>env_info</code> 不作为策略输入，而是返回给 Runner 供记录和画图，例如累计购电成本、SOC 违规量、弃电成本。最后一行的三个返回值顺序固定：团队奖励、是否终止、附加指标。

## 第 12 段：辅助接口与轨迹文件命名

**原样代码（原文件第 652--674 行）**

```python
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
```

**小白解释**

<code>render</code> 和 <code>close</code> 是环境接口的占位实现：当前没有图形渲染或需要关闭的外部资源。<br><br><code>_current_csv_path</code> 生成当前 episode 应写入的 CSV 路径。<code>//</code> 是整除，用来每 10,000 个 episode 分一个文件桶。<code>f"..."</code> 是 f-string，会将花括号中的变量填入文件名。<code>:06d</code> 表示整数至少 6 位，不足时前补 0，例如 <code>1</code> 变为 <code>000001</code>。

## 第 13 段：直接运行本文件时的独立自测程序

**原样代码（原文件第 676--735 行）**

```python
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

**小白解释**

<code>if __name__ == "__main__":</code> 是 Python 常见写法：只有直接运行 <code>python src/envs/maenv.py</code> 时才执行此段；被训练入口 import 时不会运行。<br><br>这里的 <code>mock_config</code> 是一个两 agent 的最小配置，用于脱离训练器检查环境能否读取 CSV、构建空间、reset 并完成一步 step。它的数值不等同于正式 YAML 配置。<br><br>测试动作 <code>[[0.5, -0.2], [0.1, 0.8]]</code> 的含义是：第一个 agent 以 50% 柴油机功率充电，第二个 agent 以 10% 柴油机功率放电。末尾 <code>print</code> 仅把检查结果打印到终端。

## 最重要的变量速查

| 变量 | 简单理解 |
|---|---|
| `idx = [行, 列]` | 当前选中的“哪一天、当天第几个时间步”。 |
| `pds` | 每个 agent 的柴油机实际发电功率。 |
| `pbs` | 每个 agent 实际执行的电池功率；负数充电，正数放电。 |
| `socs` | 每个 agent 电池的荷电状态，硬范围是 0.2 到 0.8。 |
| `pg` | 整个微电网与大电网的功率差；正数要买电，负数本地过剩。 |
| `economic_cost` | 当前经济模式下每个 agent 的综合成本。 |
| `reward_list` | 每个 agent 的单独奖励；随后会合成为一个团队奖励。 |

## 代码完整性补充

以下几行位于前后代码段的分界处，单独列出以保证原文件的所有非空代码行都保留在本学习文档中。

```python
        self.bucket_size = 10_000
                    "col_idx": int(self.idx[1]),
        # 判断今天 24 小时（或其他步长上限）是否结束
        end_ep = (bucket_idx + 1) * self.bucket_size
    # 手动制造一个符合你 YAML 格式的虚拟配置字典
```

这些行仍然分别属于前面介绍的构造器、轨迹字典、episode 结束判断、文件名计算和独立自测代码；这里只是为保证完整保留而重复展示，不应把它们当作一段可以单独运行的 Python 代码。
