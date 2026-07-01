@echo off
set PYTHONUNBUFFERED=1
cd /d C:\Users\xyx\Desktop\GDE-main
C:\Users\xyx\Desktop\GDE-main\.venv\Scripts\python.exe src/main.py --config=facmac_smac --env-config=microgrid with name=facmac_graph_milp_reward_scaled_3m_200_1000 use_cuda=False use_tensorboard=True use_buffer_compression=False save_model=False t_max=3000000 buffer_size=500000 lr=0.0003 critic_lr=0.0001
