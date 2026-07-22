# FACMAC Experiment Configurations

All configurations inherit the same continuous-action FACMAC, graph mixer,
microgrid environment, optimization, and evaluation settings from
`src/config/default.yaml`. Run each configuration with the same set of seeds
and compare the W&B metrics `eval/best_reward` and `eval/rl_reward` at equal
`train/t_env` values.

| Configuration | Purpose | Override |
| --- | --- | --- |
| `facmac_ea` | Main FACMAC with evolutionary augmentation. | None. |
| `facmac_no_ea` | RL-only baseline without population rollouts, evolution, or RL-to-EA synchronization. | `EA: false` |
| `facmac_ea_no_mutation` | EA ablation retaining population selection and crossover while disabling mutation. | `mutation_prob: 0.0` |
| `facmac_ea_no_crossover` | EA ablation retaining population selection and mutation while disabling crossover. | `crossover_prob: 0.0` |
| `facmac_ea_no_variation` | EA ablation retaining only selection and elite cloning. | `mutation_prob: 0.0`, `crossover_prob: 0.0` |

For every condition, keep `t_max`, `seed`, and the microgrid configuration
identical. Example run:

```powershell
uv run python src/direct_main.py --config facmac_ea --env-config microgrid --override seed=0
