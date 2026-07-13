# microgrid.yaml 详细解析

本文解释 `src/config/envs/microgrid.yaml` 中每个配置项的作用，以及它们在电网环境 `src/envs/maenv.py` 中如何参与观测、动作、状态更新和奖励计算。

## 1. 文件定位

`microgrid.yaml` 是微电网环境配置文件。运行命令中使用：

```powershell
uv run python src/main.py --config=facmac_smac --env-config=microgrid with use_cuda=False
```

其中 `--env-config=microgrid` 会加载：

```text
src/config/envs/microgrid.yaml
```

该文件主要提供两层配置：

```yaml
env: "microgrid"

env_args:
  ...
```

`env: "microgrid"` 决定使用 `src/envs/__init__.py` 中注册的 `MGEnv` 环境。`env_args` 会作为 `config` 传入 `MGEnv(config=..., algo_name=...)`。

## 2. 顶层字段

```yaml
env: "microgrid"
```

含义：选择微电网环境。

代码对应：

```python
REGISTRY["microgrid"] = partial(env_fn, env=MGEnv)
```

因此，只要配置中写 `env: "microgrid"`，runner 创建环境时就会实例化 `MGEnv`。

## 3. env_args 总体作用

`env_args` 是微电网环境本体使用的参数集合，主要控制：

- 结果目录与轨迹记录。
- 动作维度和动作范围。
- 奖励聚合方式和奖励缩放。
- 购电、发电、电池、弃电、SOC 约束等奖励权重。
- 每个智能体的发电机、电池、成本函数和数据分摊比例。

代码中读取位置主要在 `MGEnv.__init__()` 和 `MGEnv.step()`。

## 4. 结果与轨迹记录

```yaml
result_dir: "./results"
```

含义：实验结果根目录。环境会把轨迹 CSV 写到：

```text
<result_dir>/microgrid_trajs/<algo_name>/<run_id>/
```

当前配置下，默认根目录是 `./results`。

```yaml
log_trajectories: false
```

含义：是否把每一步的微电网调度轨迹写入 CSV。

- `false`：不写轨迹文件。
- `true`：按 `log_trajectories_mode` 决定写哪些 episode。

```yaml
log_trajectories_mode: "test"
```

含义：轨迹记录模式。只有 `log_trajectories: true` 时生效。

可用语义：

- `"test"`：只记录测试 episode。
- `"train"`：只记录训练 episode。
- `"all"`：训练和测试都记录。

每条轨迹记录包括价格、负荷、光伏、SOC、动作、发电功率、电池功率、购电成本、奖励分量等字段。

## 5. 动作相关配置

```yaml
action_bins: 7
```

含义：离散动作分箱数量。

当前主算法 `facmac_smac.yaml` 中：

```yaml
discretize_actions: False
agent_output_type: "continuous"
```

因此主线训练使用连续动作，`action_bins` 在当前连续动作流程中不是核心参数。

每个 agent 中有：

```yaml
low_action: [0.0, -1.0]
high_action: [1.0, 1.0]
```

含义：每个智能体动作是二维连续向量：

```text
action = [a_dg, a_batt]
```

其中：

- `a_dg` 范围 `[0.0, 1.0]`，控制柴油发电机输出比例。
- `a_batt` 范围 `[-1.0, 1.0]`，控制电池充放电比例。

在 `MGEnv.step()` 中：

```python
pds = actions_raw[:, 0] * self.params["dg_max"]
pbs_requested = actions_raw[:, 1] * self.params["battery_caps"]
```

所以：

- `pds` 是每个 agent 的发电功率。
- `pbs_requested` 是每个 agent 请求的电池功率。

按当前配置：

```yaml
dg_max: 80.0
battery_cap: 200.0
```

则每个 agent：

- `a_dg = 1.0` 表示发电 `80.0`。
- `a_dg = 0.5` 表示发电 `40.0`。
- `a_batt = 1.0` 表示请求电池功率 `200.0`。
- `a_batt = -1.0` 表示请求电池功率 `-200.0`。

注意：代码中 `pbs_requested < 0` 按充电处理，`pbs_requested > 0` 按放电处理。

## 6. 奖励聚合与缩放

```yaml
reward_aggregate: "mean"
reward_scale: 0.1
```

环境内部会先为每个 agent 计算一个 `reward_list`。然后把多个 agent 的奖励合成为一个团队奖励，返回给强化学习框架。

如果：

```yaml
reward_aggregate: "mean"
```

则：

```python
total_reward = mean(reward_list)
```

如果不是 `"mean"`，代码走求和逻辑：

```python
total_reward = sum(reward_list)
```

最后都会乘：

```python
total_reward *= reward_scale
```

当前配置表示：使用三个 agent 奖励的平均值，再乘 `0.1`。这样可以降低奖励量级，让训练更稳定。

## 7. 购电奖励配置

```yaml
buy_cost_weight: 1.0
buy_reward_mode: "relative_saving"
buy_reward_scale: 100.0
```

环境中先计算净负荷：

```python
load_pv = load - pv
```

然后根据 agent 发电和电池动作计算电网购电功率：

```python
pg = load_pv - sum(pds) - sum(pbs)
```

其中：

- `pg > 0` 表示仍需从电网购电。
- `pg < 0` 表示发电和放电超过当前负荷，出现反送或弃电语义。

基准购电成本：

```python
baseline_pg = max(load_pv, 0.0)
baseline_grid_purchase_cost = baseline_pg * price
```

实际购电成本：

```python
actual_grid_purchase_cost = max(pg, 0.0) * price
```

当前模式：

```yaml
buy_reward_mode: "relative_saving"
```

表示购电奖励按相对节省比例计算：

```python
buy_signal = buy_reward_scale * (
    (baseline_grid_purchase_cost - actual_grid_purchase_cost)
    / max(baseline_grid_purchase_cost, 1e-6)
)
```

再乘：

```python
weighted_buy_cost = buy_cost_weight * buy_signal
```

当前含义：相比“什么都不调度、直接按净负荷购电”的基准，实际购电越省，奖励越高。

如果 `buy_reward_mode` 不是 `"relative_saving"`，代码会使用：

```python
buy_cost = -actual_grid_purchase_cost
```

即直接把购电成本作为负奖励。

## 8. 发电成本配置

```yaml
generate_cost_weight: 0.1
```

每个 agent 的发电成本参数在 `data` 中：

```yaml
costa: 0.02
costb: 0.5
costc: 0.1
```

代码中的发电奖励分量：

```python
generate_costs = -(costa * pds**2 + costb * pds + costc)
weighted_generate_costs = generate_cost_weight * generate_costs
```

这是一个负奖励。发电功率 `pds` 越高，成本惩罚通常越大。

当前 `generate_cost_weight: 0.1` 表示把发电成本惩罚缩小到 10%。

## 9. 电池成本配置

```yaml
bess_cost_weight: 0.1
```

每个 agent 的电池成本参数：

```yaml
bessa: 0.01
bessb: 0.1
bessc: 0.05
```

代码中先构造：

```python
x = pbs + 3 * battery_caps * (1 - socs)
```

再计算电池成本：

```python
bess_poly = -abs(bessa * x**2 + bessb * x + bessc)
weighted_bess_cost = bess_cost_weight * bess_poly
```

这是负奖励。它惩罚电池使用与 SOC 状态相关的成本。

当前 `bess_cost_weight: 0.1` 表示把该成本分量缩小到 10%。

## 10. 环境排放或运行惩罚

```yaml
env_reward_weight: 0.1
```

每个 agent 还有：

```yaml
env_param: 0.05
```

代码中的环境惩罚：

```python
env_reward = env_reward_weight * (-pds * env_param)
```

含义：发电越多，环境惩罚越大。当前配置中该惩罚权重是 `0.1`。

## 11. 电池动作裁剪惩罚

```yaml
battery_punishment_weight: 5.0
```

每个 agent 还有：

```yaml
battery_punishment_param: 1.0
```

环境会根据 SOC 约束裁剪电池动作，保证 SOC 最终被限制在 `[0.2, 0.8]`。

如果智能体请求的电池功率导致 SOC 越界，代码会调整实际执行的电池功率：

```python
pbs = pb_adj
battery_action_clip = abs(pbs_requested - pbs)
battery_punishment = -battery_punishment_param * battery_action_clip
punishment = battery_punishment_weight * battery_punishment
```

含义：请求动作越不符合电池可行范围，惩罚越大。

当前 `battery_punishment_weight: 5.0` 较大，说明配置强烈鼓励智能体输出可执行的电池动作。

## 12. 弃电惩罚

```yaml
curtailment_penalty_weight: 1.0
```

代码中：

```python
curtailment = max(-pg, 0.0)
curtailment_penalty = -curtailment_penalty_weight * curtailment
```

含义：

- 如果 `pg < 0`，说明发电和电池放电超过当前净负荷。
- 超出的部分被视作弃电或过剩功率。
- 过剩越大，惩罚越大。

当前权重 `1.0` 表示直接按弃电量给负奖励。

## 13. SOC 储备惩罚

```yaml
soc_reserve_target: 0.35
soc_reserve_weight: 2.0
```

代码中：

```python
soc_reserve_penalty = -soc_reserve_weight * (
    max(soc_reserve_target - socs, 0.0)
    + max(socs_raw - 0.8, 0.0)
)
```

含义：

- 如果最终 SOC 低于 `0.35`，会受到惩罚。
- 如果原始 SOC 更新值 `socs_raw` 高于 `0.8`，也会受到惩罚。

这不仅要求 SOC 不越过硬边界 `[0.2, 0.8]`，还鼓励电池保留一定电量，不要长期贴近下限。

当前 `soc_reserve_target: 0.35` 表示希望 SOC 至少保持在 35% 左右。

## 14. 每个 agent 的观测范围

每个 agent 都有：

```yaml
low_state: [0.0, 0.0, 0.2]
high_state: [5.0, 5.0, 0.8]
```

这三个值对应基础观测中的：

```text
[load_data, pv_data, soc]
```

代码会额外拼接两个观测量：

```text
price
time_frac
```

所以实际每个 agent 的观测是 5 维：

```text
[标准化负荷, 标准化光伏, 当前SOC, 当前电价, 当前时间比例]
```

对应代码：

```python
return [
    np.array([
        load_data,
        pv_data,
        soc,
        price,
        time_frac,
    ], dtype=np.float32)
]
```

全局状态 `state` 是所有 agent 的观测拼接：

```python
state = concat(all_agent_obs)
```

当前 3 个 agent，每个观测 5 维，所以：

```text
state_shape = 5 * 3 = 15
```

## 15. 每个 agent 的设备参数

当前有 3 个 agent，参数完全相同：

```yaml
agent_name: "agent1"
...
agent_name: "agent2"
...
agent_name: "agent3"
...
```

### agent_name

```yaml
agent_name: "agent1"
```

含义：智能体名称。代码中会把 `data` 列表转换成以 agent 名称为 key 的字典。

注意：`MGEnv` 内部默认可能的 agent 名称是：

```text
agent1, agent2, agent3, ...
```

所以这里的名字要和这个格式匹配。

### split_ratio

```yaml
split_ratio: 0.25
```

含义：该 agent 分摊总负荷和总光伏数据的比例。

代码中：

```python
agent_load = load_data * split_ratio
agent_pv = pv_data * split_ratio
```

当前 3 个 agent 都是 `0.25`，总共只覆盖 `0.75`。这不一定是错误，因为环境的系统净负荷 `load_pv` 仍然使用全局 `load - pv`；每个 agent 的 `split_ratio` 主要影响局部观测和轨迹记录。

### dg_max

```yaml
dg_max: 80.0
```

含义：该 agent 柴油发电机最大出力。

动作映射：

```python
pd = action[0] * dg_max
```

当前 `action[0]` 范围 `[0, 1]`，所以发电功率范围是 `[0, 80]`。

### battery_cap

```yaml
battery_cap: 200.0
```

含义：该 agent 电池容量，也是动作映射中的电池功率尺度。

动作映射：

```python
pb_requested = action[1] * battery_cap
```

当前 `action[1]` 范围 `[-1, 1]`，所以请求电池功率范围是 `[-200, 200]`。

### raw_ch 和 raw_dis

```yaml
raw_ch: 0.95
raw_dis: 0.95
```

含义：电池充电、放电效率参数。

代码中 SOC 更新：

```python
delta_soc = where(
    pb < 0,
    -raw_ch * pb / battery_cap,
    -raw_dis * pb / battery_cap,
)
soc_next = soc + delta_soc
```

由于 `pb < 0` 表示充电，`-raw_ch * pb` 为正，SOC 上升。

由于 `pb > 0` 表示放电，`-raw_dis * pb` 为负，SOC 下降。

### battery_punishment_param

```yaml
battery_punishment_param: 1.0
```

含义：单个 agent 的电池动作裁剪惩罚系数。

它先作用在 agent 自身的裁剪偏差上，然后再乘全局的 `battery_punishment_weight`。

### bessa, bessb, bessc

```yaml
bessa: 0.01
bessb: 0.1
bessc: 0.05
```

含义：电池成本二次函数参数。

用于：

```python
-(abs(bessa * x**2 + bessb * x + bessc))
```

再乘 `bess_cost_weight`。

### costa, costb, costc

```yaml
costa: 0.02
costb: 0.5
costc: 0.1
```

含义：发电成本二次函数参数。

用于：

```python
-(costa * pd**2 + costb * pd + costc)
```

再乘 `generate_cost_weight`。

### env_param

```yaml
env_param: 0.05
```

含义：发电相关环境惩罚系数。

用于：

```python
env_reward = env_reward_weight * (-pd * env_param)
```

## 16. 当前奖励结构总览

每一步中，每个 agent 的奖励大致由以下部分组成：

```text
agent_reward =
    发电成本惩罚
  + 电池成本惩罚
  + 购电节省奖励或购电成本惩罚
  + 弃电惩罚
  + 发电环境惩罚
  + 电池动作裁剪惩罚
  + SOC 储备惩罚
```

对应代码结构：

```python
eco_reward = (
    weighted_generate_costs
    + weighted_bess_cost
    + weighted_buy_cost
    + curtailment_penalty
)

reward_list = (
    eco_reward
    + env_reward
    + punishment
    + soc_reserve_penalty
)
```

最后：

```python
total_reward = mean(reward_list) * reward_scale
```

因为当前配置是：

```yaml
reward_aggregate: "mean"
reward_scale: 0.1
```

## 17. 当前配置的直观含义

这份配置描述了一个 3 智能体协同微电网环境：

- 每个 agent 有一个最大出力 `80.0` 的发电机。
- 每个 agent 有一个容量尺度 `200.0` 的电池。
- 每个 agent 输出二维连续动作：发电比例和电池充放电比例。
- 环境使用真实数据表中的价格、负荷和光伏序列。
- 训练目标不是简单最小化购电成本，而是综合考虑购电节省、发电成本、电池使用成本、弃电、SOC 储备和动作可行性。
- 返回给算法的是团队标量奖励，因此这是协作式多智能体强化学习问题。

## 18. 调参时最常改的字段

如果想改变训练目标，通常优先看：

- `buy_cost_weight`
- `buy_reward_mode`
- `buy_reward_scale`
- `generate_cost_weight`
- `bess_cost_weight`
- `battery_punishment_weight`
- `curtailment_penalty_weight`
- `soc_reserve_target`
- `soc_reserve_weight`
- `reward_scale`

如果想改变物理设备能力，通常改：

- `dg_max`
- `battery_cap`
- `raw_ch`
- `raw_dis`
- `low_action`
- `high_action`

如果想改变多 agent 分摊局部观测，通常改：

- `split_ratio`

## 19. 阅读代码时的对应路线

建议按下面顺序对照阅读：

1. `src/config/envs/microgrid.yaml`：看配置值。
2. `src/envs/__init__.py`：看 `microgrid` 如何注册到 `MGEnv`。
3. `src/envs/maenv.py` 的 `__init__()`：看配置如何变成环境属性。
4. `src/envs/maenv.py` 的 `get_obs()` 和 `get_state()`：看观测和全局状态。
5. `src/envs/maenv.py` 的 `step()`：看动作、SOC、电网购电和奖励。
6. `src/runners/episode_runner.py`：看环境返回值如何进入 batch。
7. `src/learners/facmac_learner.py`：看 `reward` 如何参与 FACMAC 更新。

