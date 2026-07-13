# GDE-Microgrid 多智能体强化学习项目

## 🚀 快速开始指南

### 第一步：安装 Python（如果还没有）

1. **下载地址**：https://www.python.org/downloads/
2. **版本要求**：Python 3.8 或更高版本
3. **重要**：安装时勾选 ✅ "Add Python to PATH"

---

### 第二步：安装依赖包

打开**命令提示符（cmd）**或 **PowerShell**，运行：

```bash
cd c:\Users\xyx\Desktop\GDE-main
pip install -r requirements.txt
```

如果遇到权限问题，使用管理员身份运行命令提示符。

---

### 第三步：验证安装

创建一个测试文件 `test_install.py`：

```python
import sys
sys.path.insert(0, 'src')

from envs.maenv import MGEnv
import numpy as np

config = {
    "result_dir": "./results",
    "data": [
        {
            "agent_name": "agent1", "split_ratio": 0.25, "dg_max": 2.0, "battery_cap": 5.0,
            "battery_punishment_param": 0.1, "raw_ch": 0.95, "raw_dis": 0.95,
            "bessa": 0.1, "bessb": 0.2, "bessc": 0.3, "costa": 0.05, "costb": 0.1, "costc": 0.2, "env_param": 0.01,
            "low_state": [0.0, 0.0, 0.2], "high_state": [10.0, 10.0, 0.8]
        },
        {
            "agent_name": "agent2", "split_ratio": 0.25, "dg_max": 2.0, "battery_cap": 5.0,
            "battery_punishment_param": 0.1, "raw_ch": 0.95, "raw_dis": 0.95,
            "bessa": 0.1, "bessb": 0.2, "bessc": 0.3, "costa": 0.05, "costb": 0.1, "costc": 0.2, "env_param": 0.01,
            "low_state": [0.0, 0.0, 0.2], "high_state": [10.0, 10.0, 0.8]
        }
    ]
}

env = MGEnv(config=config, algo_name="test")
obs = env.reset()
reward, terminated, info = env.step(np.array([5, 60], dtype=np.int64))
print(f"✅ 安装成功！Reward: {reward:.2f}")
```

运行测试：
```bash
python test_install.py
```

如果看到 `✅ 安装成功！` 说明环境配置正确。

---

### 第四步：运行主程序

#### 方式1：使用批处理文件
双击 `run_microgrid.bat`

#### 方式2：手动运行
```bash
cd c:\Users\xyx\Desktop\GDE-main\src
uv run python src/direct_main.py --config facmac_ea --env-config microgrid --override use_cuda=False --override t_max=5000
```

CUDA is enabled by default when PyTorch detects a compatible GPU. The environment
and replay buffer remain on CPU; policy inference and learner updates use CUDA.
To force CPU execution, add `--override use_cuda=False`.

For a GPU throughput-oriented run, use `facmac_ea_gpu`:

```powershell
uv run python src/direct_main.py --config facmac_ea_gpu --env-config microgrid --override seed=0
```

---

## 📊 运行后会看到什么

### 终端输出示例：
```
[INFO 14:30:25] Experiment Parameters:
{
    "env": "microgrid",
    "n_agents": 3,
    "n_actions": 121,
    ...
}

[INFO 14:30:26] Beginning training for 5000 timesteps
t_env: 0 / 5000

[INFO 14:30:30] Estimated time left: 0:05:00

t_env: 1000 / 5000
RL eval  -125.34  0.85
EA eval  -130.21  0.82
```

### 生成的文件：
- `results/wandb/` - W&B 实验记录
- W&B 在线模式需要先执行 `uv run wandb login`；本地验证可添加 `--override wandb_mode=offline`。
- `results/models/` - 保存的模型

---

## 🔧 常见问题

### 问题1：pip 不是内部或外部命令
**解决方案**：
1. 重新安装Python，确保勾选"Add Python to PATH"
2. 或者使用完整路径：`c:\Users\你的用户名\AppData\Local\Programs\Python\Python3x\Scripts\pip.exe`

### 问题2：缺少某个包
```bash
pip install numpy torch pandas pyyaml
```

### 问题3：权限不足
以**管理员身份**运行命令提示符：
1. 右键点击"命令提示符"
2. 选择"以管理员身份运行"

### 问题4：CUDA相关错误
如果你没有NVIDIA显卡，在命令后加 `use_cuda=False`：
```bash
uv run python src/direct_main.py --config facmac_ea --env-config microgrid --override use_cuda=False
```

---

## 🎯 推荐的操作步骤

1. **先安装Python**（如果需要）
2. **安装依赖**：`pip install -r requirements.txt`
3. **验证安装**：`python test_install.py`
4. **运行主程序**：`uv run python src/direct_main.py ...`

---

## 📞 获取帮助

如果遇到其他问题，请记录：
- 完整的错误信息
- 你执行的命令
- Python版本：`python --version`

---

**祝你玩得开心！🎉**
