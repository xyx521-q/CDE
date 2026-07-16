# microgrid.yaml 小白逐段注释版

> 对照原文件：`src/config/envs/microgrid.yaml`。这是额外的新学习文件，不替代、不修改原 YAML；训练仍从原路径读取配置。
>
> 阅读方法：每个代码块都是原 YAML 中连续的一段，代码后面紧接着解释每一个配置在环境中做什么。

## 先认识 YAML

YAML 是“靠缩进表示层级”的配置格式。冒号左边是键名，右边是值；两个空格的缩进表示“属于上一级”；短横线 <code>-</code> 表示列表中的一项；<code>#</code> 后面是给人看的注释。加载后，下面的内容会变成 Python 字典和列表，并传入 <code>MGEnv(config=env_args, ...)</code>。

## 第 1 段：文件开头和顶层容器

**原样配置（原文件第 1--2 行）**

```yaml
# microgrid.yaml
env_args:
```

**小白解释**

第 1 行以 <code>#</code> 开头，是注释，YAML 加载器会忽略它。<code>env_args:</code> 后面没有写具体数值，而是换行并缩进两个空格，因此它表示一个“字典/映射”的开始。下面缩进两格的所有字段，都会成为 Python 中的 <code>config</code> 字典内容。

## 第 2 段：结果目录与轨迹记录开关

**原样配置（原文件第 3--5 行）**

```yaml
  result_dir: "./results"
  log_trajectories: false
  log_trajectories_mode: "test"
```

**小白解释**

<code>result_dir</code> 是结果根目录。环境按 <code>results/microgrid_trajs/算法名/run_id</code> 创建轨迹目录。<br><br><code>false</code> 是 YAML 布尔值，对应 Python 的 <code>False</code>；所以目前不会把每个时间步的调度信息写成 CSV。<code>log_trajectories_mode: "test"</code> 只有在打开记录开关后才生效，表示只记录测试 episode。

## 第 3 段：奖励模式与环境级权重

**原样配置（原文件第 6--18 行）**

```yaml
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
```

**小白解释**

<code>reward_mode: "milp_economic"</code> 是最关键的开关：它让 <code>MGEnv.step()</code> 使用综合经济成本的负数作为主要奖励。该模式合并了购电、柴油机、电池、环境与弃电成本。<br><br><code>reward_aggregate: "sum"</code> 表示三个 agent 的奖励会相加，生成训练器看到的一个团队奖励；<code>reward_scale: 1.0</code> 不再额外缩放。<br><br><code>buy_cost_weight</code>、<code>buy_reward_mode</code>、<code>buy_reward_scale</code>、<code>generate_cost_weight</code>、<code>bess_cost_weight</code>、<code>env_reward_weight</code> 是旧式奖励路径的参数。当前 <code>milp_economic</code> 分支不会把这些权重乘进主经济成本，但保留它们使切换奖励模式时仍有配置。<br><br><code>battery_punishment_weight</code> 放大电池动作被 SOC 边界截断时的惩罚。<code>curtailment_penalty_weight</code> 惩罚本地供给超过负荷的量。<code>soc_reserve_target</code> 是期望预留的 SOC，<code>soc_reserve_weight</code> 决定低于它后扣分多重；它们不同于不可越过的 0.2/0.8 硬边界。

## 第 4 段：agent1 的完整参数

**原样配置（原文件第 19--37 行）**

```yaml
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
```

**小白解释**

<code>data:</code> 下面缩进四格的 <code>-</code> 表示“列表中的第一项”。每一个列表项就是一个智能体的完整设备与成本参数。<br><br><code>low_state/high_state</code> 是三个基础观测量的上下界：标准化负荷、标准化光伏、SOC。请注意环境代码还会自动追加“电价”和“时间比例”两维，所以模型最终收到的是 5 维观察。<br><br><code>low_action/high_action</code> 是两个连续动作的范围。动作 0 的 [0, 1] 对应柴油机从关闭到最大功率；动作 1 的 [-1, 1] 对应电池从最大充电到最大放电。<br><br><code>dg_max: 80.0</code> 意味着动作 0 为 1 时柴油机出力 80。<code>battery_cap: 200.0</code> 同时用于电池动作功率换算和 SOC 变化计算。<code>raw_ch/raw_dis</code> 是充/放电效率，0.95 表示大约 95%。<br><br><code>split_ratio</code> 是分摊全局负荷、光伏以及共享电网成本的比例。三位 agent 都是约 1/3。<code>bessa/bessb/bessc</code> 定义电池成本多项式；<code>costa/costb/costc</code> 定义柴油机成本多项式；<code>env_param</code> 是每单位柴油机功率的环境成本系数。

## 第 5 段：agent2 的参数

**原样配置（原文件第 38--55 行）**

```yaml
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
```

**小白解释**

这是列表中的第二个 agent。YAML 中同一缩进层级的 <code>- agent_name</code> 表示一个新的字典，而不是 agent1 的子字段。<br><br>当前 agent2 的设备、动作范围、成本和分摊比例全部与 agent1 相同，只有名称不同。因此环境会把它作为独立决策者，但它拥有同样的物理能力，并分到相同的 1/3 数据份额。

## 第 6 段：agent3 的参数和文件末尾注释

**原样配置（原文件第 56--75 行）**

```yaml
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

**小白解释**

这是第三个 agent，参数同样与前两者一致。三份 <code>split_ratio: 0.3333333333333333</code> 相加约等于 1，使每个 agent 的分摊总和对应整个微电网。<br><br>最后一行的 <code>#</code> 注释只是提示如何增加 agent。实际增加时，不能只复制 YAML：名称必须连续为 <code>agent1</code>、<code>agent2</code>、...，因为环境代码会按此名称查找配置；同时各个 <code>split_ratio</code> 通常应重新调整，使总和保持 1。

## 当前配置会如何运行

1. 环境创建三个 agent，并从 CSV 随机选取一行作为一天。
2. 每个时间步，每个 agent 输出两个数字：柴油机比例和电池比例。
3. 柴油机最大为 80；电池动作最大按 200 换算，负数是充电、正数是放电。
4. 若电池 SOC 想越过 0.2 或 0.8，环境会削减实际动作并扣分。
5. 三人的负荷、光伏和共享购电/弃电成本各按约三分之一分配。
6. 经济成本取负后形成各自奖励，最后三个奖励相加为团队奖励。

## 修改配置前的四条安全规则

- <code>low_action</code> 和 <code>high_action</code> 必须都有两个数；<code>low_state</code> 和 <code>high_state</code> 必须都有三个数。
- 每一位 agent 都必须保留环境读取的所有字段；缺任何一个，初始化时会直接报错。
- 新增 agent 时，名称必须连续，且所有 <code>split_ratio</code> 最好加总为 1。
- 修改成本或奖励权重会直接改变训练目标；先用很短的训练运行确认环境能正常完成至少一个 episode。

