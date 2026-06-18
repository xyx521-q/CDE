import sys
sys.path.insert(0, 'src')

import traceback
import numpy as np

try:
    print("=" * 60)
    print("Testing MGEnv Environment")
    print("=" * 60)

    from envs.maenv import MGEnv

    mock_config = {
        "result_dir": "./results",
        "action_bins": 11,
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

    print("\n[1/6] Initializing MGEnv...")
    env = MGEnv(config=mock_config, algo_name="test")
    print(f"   ✅ n_agents: {env.n_agents}")
    print(f"   ✅ n_actions: {env.n_actions}")
    print(f"   ✅ action_bins: {env.action_bins}")

    print("\n[2/6] Testing get_env_info()...")
    env_info = env.get_env_info()
    print(f"   ✅ n_agents: {env_info['n_agents']}")
    print(f"   ✅ n_actions: {env_info['n_actions']}")
    print(f"   ✅ obs_shape: {env_info['obs_shape']}")
    print(f"   ✅ state_shape: {env_info['state_shape']}")
    print(f"   ✅ episode_limit: {env_info['episode_limit']}")

    print("\n[3/6] Testing reset()...")
    obs = env.reset()
    print(f"   ✅ Obs type: {type(obs)}")
    print(f"   ✅ Obs length: {len(obs)}")
    print(f"   ✅ First obs shape: {np.array(obs[0]).shape}")

    print("\n[4/6] Testing get_obs()...")
    obs = env.get_obs()
    print(f"   ✅ Obs length: {len(obs)}")
    print(f"   ✅ Sample obs: {obs[0]}")

    print("\n[5/6] Testing get_state()...")
    state = env.get_state()
    print(f"   ✅ State shape: {state.shape}")

    print("\n[6/6] Testing step() with discrete actions...")
    for action_idx in [0, 60, 120]:
        mock_actions = np.array([action_idx, action_idx], dtype=np.int64)
        reward, terminated, env_info = env.step(mock_actions)
        print(f"   Action {action_idx}: reward={reward:.2f}, terminated={terminated}")

    print("\n" + "=" * 60)
    print("🎉 All tests passed! Environment is working correctly.")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
