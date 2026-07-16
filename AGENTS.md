# AGENTS.md

## Project Scope

- This is a Python 3.12 multi-agent reinforcement-learning project for microgrid experiments. The fixed training pipeline is continuous-action FACMAC with an optional neuroevolution (EA) population.
- Source code lives in `src/`; runtime YAML lives in `src/config/`; versioned environment inputs live in `src/envs/data/`.
- `src/direct_main.py` is the only training entry point. It merges `default.yaml`, one environment YAML, and one algorithm YAML; `--override key=value` supports dotted keys.
- The main experiment is `facmac_ea` + `microgrid`. `facmac_no_ea` and `facmac_ea_no_*` are controlled comparisons. `facmac_ea_gpu` is a throughput preset and is not directly comparable because it changes batch and logging settings.

## Architecture Conventions

- Keep explicit imports and direct construction of `MGEnv`, `EpisodeRunner`, `CQMixMAC`, `RNNAgent`, and FACMAC learners. Do not introduce registries or configuration-selected implementations.
- Only continuous Gymnasium `Box` action spaces are supported. Do not restore discrete-action, particle-environment, or StarCraft II paths.
- Preserve the runner -> `EpisodeBatch`/`ReplayBuffer` -> controller -> learner tensor contract. Changes to observations, actions, state shape, or agent count must be checked across all four boundaries.
- Keep environment interaction, episode batches, and replay storage on CPU. When CUDA is enabled, move policy inference, critic/mixer training, sampled learner batches, and EA population networks to CUDA explicitly.
- EA implementations share behavior through `src/ea/ssne_base.py`; retain the existing ablation implementations and keep mutation/crossover operations valid for both CPU and CUDA tensors.
- Source directories are namespace packages. Do not add empty `__init__.py` files.

## Configuration And Dependencies

- Use `uv` exclusively. Run project commands through `uv run`; update `pyproject.toml` and `uv.lock` together when dependencies change.
- Windows and Linux use the official PyTorch CUDA 12.8 source declared in `pyproject.toml`, currently pinned for RTX 5070 Ti/Blackwell support. Preserve a working CPU fallback through `use_cuda=False`.
- Keep shared algorithm parameters in `src/config/default.yaml`; algorithm YAML files should contain only intentional experiment differences.
- Training intensity must remain configuration-controlled. `learner_update_interval` controls how often normal episodes trigger training, `learner_updates_per_episode` controls the number of updates at that trigger, `ea_learner_updates` controls learner updates during an EA cycle, and `ea_generations_per_cycle` controls population generations per EA cycle.
- CPU and GPU experiments share batch 512, buffer warmup 512, buffer size 100000, `learner_update_interval: 2`, `learner_updates_per_episode: 1`, `ea_learner_updates: 3`, and `ea_generations_per_cycle: 1`. `facmac_ea.yaml` selects CPU with `use_cuda: false`; `facmac_ea_gpu.yaml` selects CUDA and may differ only in numerical precision plus evaluation, logging, and checkpoint cadence.
- Do not change microgrid reward definitions, agent constraints, or source CSV files during unrelated code work.

## Performance Conventions

- Keep the vectorized feed-forward critic and graph-mixer time dimension. The actor and target actor remain sequential across episode time because `RNNAgent` uses a GRU hidden state.
- `CQMixMAC` has one shared policy network for all agents. Run each configured EA generation once over that complete policy; do not loop over `n_agents` and evolve the same shared network repeatedly.
- Keep continuous action scaling and clamping as batched tensor operations with device-local cached bounds. Actions must still return to CPU before environment interaction and `EpisodeBatch` insertion.
- Keep target-policy and target-critic inference under `torch.no_grad()`. During actor optimization, allow gradients through critic operations to the actions but do not accumulate gradients for critic or mixer parameters.
- Keep agent-level EA mutation batched on the tensor device. Do not reintroduce per-weight CUDA `.item()` calls or other CPU/GPU synchronization inside mutation loops.
- `float32_matmul_precision: high` is a throughput-preset choice, not a universal experiment default. Preserve the stricter default unless an algorithm config opts in.
- Evaluate throughput using environment steps per wall-clock second and phase timing, not GPU utilization alone. Larger batches may increase utilization while reducing overall training throughput.

## Failure And Data Rules

- Let training, tensor-shape, environment, and file-loading errors surface. Do not add broad exception handlers, silent fallbacks, or warning suppression.
- `MGEnv` mutates its received `env_args["data"]` entries by removing `agent_name`; do not reuse the same configuration object after environment construction without copying it.
- Generated data belongs under `results/`, including W&B runs, checkpoints, timing CSVs, and trajectories. Do not edit or delete existing artifacts unless explicitly requested.
- Keep logging aggregated. High-frequency learner, target-update, and EA-cycle messages must use configurable intervals rather than print every update.

## Validation

- Do not continuously monitor a running training job unless the user explicitly requests real-time supervision.
- Sync dependencies with `uv sync` and verify lock consistency with `uv lock --check` after dependency changes.
- CPU smoke test: `uv run python src/direct_main.py --config facmac_ea --env-config microgrid --override use_cuda=False --override wandb_mode=disabled --override save_model=False --override t_max=12500`.
- GPU preset: `uv run python src/direct_main.py --config facmac_ea_gpu --env-config microgrid`.
- Configuration changes must be loaded through `src/direct_main.py`. Training-loop or buffer changes must cross replay warmup and complete at least one learner update.
- EA scheduling or mutation changes must complete at least one EA cycle on CPU and CUDA. Exercise both `ea_generations_per_cycle: 1` and the comparable value `3` when changing generation control.
- Time-vectorization changes must compare batched and per-timestep outputs with float32 tolerances before relying on performance results.
- The fixed timing harness is `uv run python tests/test_training_timing.py --config facmac_ea --timesteps 12144 --output results/<new-file>.csv`. Write a new result file; do not overwrite existing timing artifacts.
- Before committing, inspect `git status` and confirm the intended Git author name and email. Do not amend or force-push without explicit approval.
