# VMAS FACMAC-EA tuning state

## Reference runs

| Run | Algorithm override | Final `t_env` | Best | RL | EA |
| --- | --- | ---: | ---: | ---: | ---: |
| `be6ag6le` | Baseline: `lr=3e-4`, `critic_lr=1e-4`, `act_noise=0.05` | 2,999,200 | 0.879783 | -0.290201 | 0.879783 |
| `t19cl9g6` | Low LR: `lr=1e-4`, `critic_lr=3e-5`, `act_noise=0.05` | 2,998,924 | 0.636877 | 0.636877 | 0.581628 |

`be6ag6le` is the current optimization baseline because it has the higher best and EA evaluation rewards at a comparable training budget. `t19cl9g6` has the stronger final RL evaluation, so the current evidence also indicates a stability tradeoff rather than dominance on every metric. Both results use seed 0; tuning decisions require repeat seeds before treating reward differences as general.

## Reusable decisions

- Keep the VMAS transport environment, FACMAC-EA structure, seed, 3M-step budget, replay settings, and EA schedule fixed while testing optimizer or exploration changes.
- Test `facmac_ea_lr_mid` at `lr=2e-4` and `critic_lr=7e-5` to interpolate between the stronger baseline and the lower-rate condition.
- Test `facmac_ea_noise_low` at baseline learning rates with `act_noise=0.025` to isolate whether lower persistent action noise improves final policy stability.
- Compare best, RL, and EA evaluation rewards at equal `train/t_env`; also inspect late-run variability and do not select from best reward alone.
