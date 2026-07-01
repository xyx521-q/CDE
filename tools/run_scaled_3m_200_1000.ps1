$ErrorActionPreference = "Stop"
Set-Location "C:\Users\xyx\Desktop\GDE-main"
$env:UV_CACHE_DIR = "C:\Users\xyx\Desktop\GDE-main\.uv-cache"
uv run --no-project --python 3.10 `
  --with numpy==1.24.2 `
  --with scipy `
  --with pyyaml `
  --with pandas `
  --with gym==0.26.2 `
  --with gymnasium `
  --with pettingzoo `
  --with sacred `
  --with tensorboard_logger `
  --with tensorboardX `
  --with "protobuf<3.21" `
  --with torch `
  --with fastrand `
  python src/main.py --config=facmac_smac --env-config=microgrid with `
  name=facmac_graph_milp_reward_scaled_3m_200_1000 `
  use_cuda=False `
  use_tensorboard=True `
  use_buffer_compression=False `
  save_model=False `
  t_max=3000000 `
  buffer_size=500000 `
  lr=0.0003 `
  critic_lr=0.0001
