import sys

sys.path.insert(0, "src")

import numpy as np

from envs.maenv import MGEnv


def main():
    mock_config = {
        "result_dir": "./results",
        "action_bins": 11,
        "reward_aggregate": "mean",
        "reward_scale": 1.0,
        "buy_cost_weight": 0.0,
        "buy_reward_mode": "raw_cost",
        "buy_reward_scale": 0.0,
        "generate_cost_weight": 0.0,
        "bess_cost_weight": 0.0,
        "env_reward_weight": 0.0,
        "battery_punishment_weight": 0.0,
        "soc_reserve_target": 0.35,
        "soc_reserve_weight": 0.0,
        "curtailment_penalty_weight": 1.0,
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
                "low_action": [0.0, -1.0],
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
                "low_action": [0.0, -1.0],
                "high_action": [1.0, 1.0],
            },
        ],
    }

    env = MGEnv(config=mock_config, algo_name="test", render_mode=None)
    env.reset(seed=0)

    env.idx = [0, 0]
    env.params["socs"] = np.array([0.8, 0.8], dtype=float)
    env.data_dict["load_pv"][0, 0] = -10.0
    env.data_dict["price"][0, 0] = 5.0

    actions = np.zeros((env.n_agents, env.n_actions), dtype=np.float32)
    reward, terminated, env_info = env.step(actions)

    assert env_info["curtailment_reward"] == -20.0
    assert env_info["curtailment_reward"] < 0.0
    assert env.ep_buffer[-1]["curtailment_penalty"] == -10.0
    assert reward < 0.0
    assert terminated is False

    print("curtailment handling ok")


if __name__ == "__main__":
    main()
