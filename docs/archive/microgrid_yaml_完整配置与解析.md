# `microgrid.yaml` 完整配置与解析

> 对应原文件：`src/config/envs/microgrid.yaml`。本文件保留当前 YAML 的完整副本，并解释 YAML 语法、配置层级，以及每个值如何传给 `MGEnv`。实际训练仍读取原 YAML 文件。

## 一、完整配置

```yaml
# microgrid.yaml
env_args:
  result_dir: "./results"
  log_trajectories: false
  log_trajectories_mode: "test"
  reward_mode: "milp_economic"
  reward_aggregate: "sum"
  reward_scale: 1.0
  buy_cost_weight: 1.0
  buy_reward_mode: "raw_cost"
  buy_reward_scale: 100.0
  generate_cost_weight: 1.0
  bess_cost_weight: 1.0
  env_reward_weight: 1.0
  battery_punishment_weight: 1.0
  curtailment_penalty_weight: 1.0
  soc_reserve_target: 0.35
  soc_reserve_weight: 2.0
  data:
    - agent_name: "agent1"
      low_state: [0.0, 0.0, 0.2]
      high_state: [5.0, 5.0, 0.8]
      low_action: [0.0, -1.0]
      high_action: [1.0, 1.0]
      dg_max: 80.0
      battery_cap: 200.0
      battery_punishment_param: 1.0
      raw_ch: 0.95
      raw_dis: 0.95
      split_ratio: 0.3333333333333333
      bessa: 0.01
      bessb: 0.1
      bessc: 0.05
      costa: 0.02
      costb: 0.5
      costc: 0.1
      env_param: 0.05
    - agent_name: "agent2"
      low_state: [0.0, 0.0, 0.2]
      high_state: [5.0, 5.0, 0.8]
      low_action: [0.0, -1.0]
      high_action: [1.0, 1.0]
      dg_max: 80.0
      battery_cap: 200.0
      battery_punishment_param: 1.0
      raw_ch: 0.95
      raw_dis: 0.95
      split_ratio: 0.3333333333333333
      bessa: 0.01
      bessb: 0.1
      bessc: 0.05
      costa: 0.02
      costb: 0.5
      costc: 0.1
      env_param: 0.05
    - agent_name: "agent3"
      low_state: [0.0, 0.0, 0.2]
      high_state: [5.0, 5.0, 0.8]
      low_action: [0.0, -1.0]
      high_action: [1.0, 1.0]
      dg_max: 80.0
      battery_cap: 200.0
      battery_punishment_param: 1.0
      raw_ch: 0.95
      raw_dis: 0.95
      split_ratio: 0.3333333333333333
      bessa: 0.01
      bessb: 0.1
      bessc: 0.05
      costa: 0.02
      costb: 0.5
      costc: 0.1
      env_param: 0.05
    
    # 如果有多个 agent，依此类推在下面贴上 agent2, agent3, agent4...
```

## 二、YAML 语法与加载关系

YAML 用缩进表示层级，而不是花括号。冒号 `:` 分隔键和值；`#` 开头是注释；`[...]` 是内联列表；`-` 表示列表中的一个元素；字符串可用双引号标识。这里的层级为：

```text
env_args (映射/dict)
  ├─ 环境级标量：奖励、轨迹、SOC 等设置
  └─ data (列表/list)
       ├─ 第 1 个 agent 的映射
       ├─ 第 2 个 agent 的映射
       └─ 第 3 个 agent 的映射
```

训练入口从此文件取出 `env_args`，作为 `MGEnv(config=env_args, ...)` 的 `config` 参数。每一个 `data` 列表项必须有 `agent_name`，且名称必须与环境依次生成的 `agent1`、`agent2`、`agent3` 对应。

## 三、环境级字段

| 字段 | 当前值 | 代码位置与含义 |
|---|---:|---|
| `result_dir` | `"./results"` | 轨迹根目录；环境在其下创建 `microgrid_trajs/<algo>/<run_id>`。 |
| `log_trajectories` | `false` | 是否将逐 agent、逐时间步数据写为 CSV。关闭时只保留内存中的当前 episode 缓冲。 |
| `log_trajectories_mode` | `"test"` | 开启轨迹时选择 `test`、`train` 或 `all` episode。 |
| `reward_mode` | `"milp_economic"` | 当前使用经济成本奖励分支：购电分摊、柴油、电池、环境与弃电成本相加后取负。 |
| `reward_aggregate` | `"sum"` | 将三个 agent 的奖励求和，作为学习器得到的单一团队奖励；只有值正好为 `"mean"` 时才求平均。 |
| `reward_scale` | `1.0` | 团队奖励最后的统一乘数。 |
| `buy_cost_weight` | `1.0` | 旧式奖励分支中的购电奖励权重；当前 `milp_economic` 分支不读取该权重。 |
| `buy_reward_mode` | `"raw_cost"` | 旧式奖励分支的购电信号选择；当前 `milp_economic` 分支不读取。 |
| `buy_reward_scale` | `100.0` | 旧式 `relative_saving` 购电信号的放大系数；当前分支不读取。 |
| `generate_cost_weight` | `1.0` | 旧式奖励分支的柴油成本权重；当前经济分支直接计入完整柴油成本。 |
| `bess_cost_weight` | `1.0` | 旧式奖励分支的电池成本权重；当前经济分支直接计入完整电池成本。 |
| `env_reward_weight` | `1.0` | 旧式环境排放/运行惩罚权重；当前经济分支通过 `env_param` 直接计入成本。 |
| `battery_punishment_weight` | `1.0` | 放大 SOC 边界导致的“请求功率与实际执行功率”差额惩罚。 |
| `curtailment_penalty_weight` | `1.0` | 当 `pg < 0`、本地供给过剩时的弃电惩罚权重；当前经济分支按分摊比例分给 agent。 |
| `soc_reserve_target` | `0.35` | SOC 储备目标；低于它会产生额外负奖励，即使仍高于绝对下限 0.2。 |
| `soc_reserve_weight` | `2.0` | SOC 储备缺口惩罚的系数。 |

“当前分支不读取”不等于字段无效：如果把 `reward_mode` 改为非 `milp_economic`，相关字段会重新参与 `step()` 的旧式奖励路径。

## 四、`data` 列表与每个智能体的字段

三个配置块的数值和字段相同，差别只有 `agent_name`。因此当前环境为三个参数相同、各分到三分之一负荷和光伏数据的 agent。三个 `split_ratio` 之和约为 1；当前经济模式也用这些比例分摊电网购电和弃电成本。

| agent 字段 | 当前值 | 语法/物理含义 |
|---|---:|---|
| `agent_name` | `"agent1"` 等 | 字符串 ID；构造器用 `pop("agent_name")` 将它作为字典键。 |
| `low_state` | `[0.0, 0.0, 0.2]` | 三项原始观测下界：标准化负荷、标准化光伏、SOC。环境额外追加电价和时间两维。 |
| `high_state` | `[5.0, 5.0, 0.8]` | 对应三项原始观测上界；SOC 可行范围是 0.2 到 0.8。 |
| `low_action` | `[0.0, -1.0]` | 二维连续动作下界：柴油机比例不小于 0，电池比例可到 -1（充电）。 |
| `high_action` | `[1.0, 1.0]` | 二维连续动作上界：柴油机最大额定功率，电池最大放电比例。 |
| `dg_max` | `80.0` | 柴油发电机额定功率。实际 `pd = action[0] * 80`。 |
| `battery_cap` | `200.0` | 电池容量，同时代码以它将电池动作比例换算为功率：`pb = action[1] * 200`。 |
| `battery_punishment_param` | `1.0` | 单 agent 的裁剪差额惩罚系数。 |
| `raw_ch` | `0.95` | 充电效率；负电池功率增大 SOC 时使用。 |
| `raw_dis` | `0.95` | 放电效率；正电池功率减小 SOC 时使用。 |
| `split_ratio` | `0.3333333333333333` | 该 agent 分摊的负荷/光伏比例；同时用于分摊共享电网成本。 |
| `bessa` | `0.01` | 电池成本二次项系数。 |
| `bessb` | `0.1` | 电池成本一次项系数。 |
| `bessc` | `0.05` | 电池成本常数项。 |
| `costa` | `0.02` | 柴油成本二次项系数。 |
| `costb` | `0.5` | 柴油成本一次项系数。 |
| `costc` | `0.1` | 柴油成本常数项。 |
| `env_param` | `0.05` | 每单位柴油功率的环境成本系数。 |

## 五、当前配置下的一步计算

设第 i 个 agent 的动作是 `[a_dg_i, a_batt_i]`，则动作范围保证 `0 <= a_dg_i <= 1` 且 `-1 <= a_batt_i <= 1`。当前配置把它转换为：

```text
pd_i = 80 * a_dg_i
pb_requested_i = 200 * a_batt_i
```

负 `pb` 是充电，正 `pb` 是放电。环境先按效率计算候选 SOC，若越过 [0.2, 0.8]，就削减实际电池功率到边界，并对削减量施加惩罚。总电网功率是：

```text
pg = (load - pv) - sum(pd_i) - sum(pb_i)
```

当 `pg > 0` 时按 `price * pg` 购电；当 `pg < 0` 时不产生卖电收入，而是以 `-pg` 的量作为弃电惩罚来源。当前 `milp_economic` 奖励对每个 agent 的核心形式是：

```text
reward_i =
  - [分摊购电成本_i + 柴油成本_i + 电池成本_i
     + 环境成本_i + 分摊弃电成本_i]
  - 电池越界动作惩罚_i
  - SOC 储备不足惩罚_i
```

最后 `reward_aggregate: "sum"` 将三个 `reward_i` 求和，`reward_scale: 1.0` 保持该量级不变。

## 六、配置修改时必须保持的约束

- `data` 中的 agent 数会决定环境的 `n_agents`；名称必须连续对应 `agent1` 到 `agentN`，否则环境按 `possible_agents` 查找配置会失败。
- 三份 CSV 的行数和列数必须一致；`episode_limit` 由价格 CSV 的列数决定。
- `low_state/high_state` 必须各有 3 项，因为环境会再追加价格和时间两项；`low_action/high_action` 必须各有 2 项。
- `split_ratio` 应为非负且总和为 1，才能使负荷/光伏分配和共享购电成本分配保持物理一致。
- `battery_cap`、`dg_max` 应为正数；充放电效率通常应落在 (0, 1]；SOC 储备目标应处于绝对 SOC 边界 [0.2, 0.8] 内。

