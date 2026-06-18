import sys
sys.path.insert(0, 'src')

try:
    print("Step 1: Importing modules...")
    import numpy as np
    print(f"NumPy version: {np.__version__}")

    print("\nStep 2: Importing MGEnv...")
    from envs.maenv import MGEnv
    print("MGEnv imported successfully")

    print("\nStep 3: Creating config...")
    config = {
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

    print("\nStep 4: Initializing environment...")
    env = MGEnv(config=config, algo_name="test")
    print(f"Environment created! n_agents={env.n_agents}, n_actions={env.n_actions}")

    print("\nStep 5: Testing reset...")
    obs = env.reset()
    print(f"Reset successful! Obs length: {len(obs)}")

    print("\nStep 6: Testing step...")
    import numpy as np
    actions = np.array([5, 60], dtype=np.int64)
    reward, terminated, info = env.step(actions)
    print(f"Step successful! Reward: {reward:.2f}, Terminated: {terminated}")

    print("\n✅ SUCCESS: Environment is working!")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
