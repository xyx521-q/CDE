# FACMAC Experiment Configurations

All configurations inherit the same continuous-action FACMAC, graph mixer,
microgrid environment, optimization, and evaluation settings from
`src/config/default.yaml`. Run each configuration with the same set of seeds
and compare the W&B metrics `eval/best_reward` and `eval/rl_reward` at equal
`train/t_env` values.

| Configuration | Purpose | Override |
| --- | --- | --- |
| `facmac_ea` | Main FACMAC with evolutionary augmentation. | None. |
| `facmac_ea_gpu` | RTX 5070 Ti 12 GB preset using CUDA 12.8, a 512-episode learner batch, bounded CPU replay, and less frequent I/O. | Not for strict comparisons. |
| `facmac_no_ea` | RL-only baseline without population rollouts, evolution, or RL-to-EA synchronization. | `EA: false` |
| `facmac_ea_no_mutation` | EA ablation retaining population selection and crossover while disabling mutation. | `mutation_prob: 0.0` |
| `facmac_ea_no_crossover` | EA ablation retaining population selection and mutation while disabling crossover. | `crossover_prob: 0.0` |
| `facmac_ea_no_variation` | EA ablation retaining only selection and elite cloning. | `mutation_prob: 0.0`, `crossover_prob: 0.0` |

For every condition, keep `t_max`, `seed`, and the microgrid configuration
identical. Example CPU run:

```powershell
uv run python src/direct_main.py --config facmac_ea --env-config microgrid --override use_cuda=False --override seed=0
```

For a CUDA-capable GPU, use the throughput preset:

```powershell
uv run python src/direct_main.py --config facmac_ea_gpu --env-config microgrid --override seed=0
```

The environment and replay buffer remain on CPU. If another GPU has less memory,
lower `batch_size` and `buffer_warmup` together to `256`. A GPU with at least
16 GB can try `1024` for both values after checking memory use.
