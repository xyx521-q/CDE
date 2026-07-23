# VMAS FACMAC-EA tuning state

## Reference runs

| Run | Algorithm override | Final `t_env` | Best | RL | EA |
| --- | --- | ---: | ---: | ---: | ---: |
| `be6ag6le` | Baseline: `lr=3e-4`, `critic_lr=1e-4`, `act_noise=0.05` | 2,999,200 | 0.879783 | -0.290201 | 0.879783 |
| `t19cl9g6` | Low LR: `lr=1e-4`, `critic_lr=3e-5`, `act_noise=0.05` | 2,998,924 | 0.636877 | 0.636877 | 0.581628 |
| `lrzqbcuw` | Mid LR: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.05` | 2,999,005 | 0.890246 | 0.324661 | 0.890246 |
| `tvc2z07r` | Low noise: `lr=3e-4`, `critic_lr=1e-4`, `act_noise=0.025` | 2,999,010 | -0.173671 | -0.173671 | -0.314839 |

`be6ag6le` is the reference baseline because it has the higher best and EA evaluation rewards than `t19cl9g6` at a comparable training budget. `t19cl9g6` has the stronger final RL evaluation, so that comparison indicates a stability tradeoff rather than dominance on every metric. Both results use seed 0; tuning decisions require repeat seeds before treating reward differences as general.

## Reusable decisions

- Keep the VMAS transport environment, FACMAC-EA structure, seed, 3M-step budget, replay settings, and EA schedule fixed while testing optimizer or exploration changes.
- Test `facmac_ea_lr_mid` at `lr=2e-4` and `critic_lr=7e-5` to interpolate between the stronger baseline and the lower-rate condition.
- Test `facmac_ea_noise_low` at baseline learning rates with `act_noise=0.025` to isolate whether lower persistent action noise improves final policy stability.
- Compare best, RL, and EA evaluation rewards at equal `train/t_env`; also inspect late-run variability and do not select from best reward alone.

## Current selection

`lrzqbcuw` is selected over `tvc2z07r`. At the shared 3M-step budget it has the stronger final best, RL, and EA rewards. In a 600-point W&B history sample, their final 500k-step best-reward means are close (`0.708442` for `lrzqbcuw` and `0.711970` for `tvc2z07r`), while `lrzqbcuw` has the higher EA mean (`0.385419` versus `0.351047`) and lower RL and EA variability. The final-point advantage is therefore meaningful for selection but not enough to claim general dominance from seed 0 alone.

The next runs keep the selected mid learning rates and default action noise fixed, changing only the seed:

- `facmac_ea_lr_mid_seed1`: repeat with `seed=1`.
- `facmac_ea_lr_mid_seed2`: repeat with `seed=2`.

Use these replications to decide whether the mid-learning-rate result is stable before introducing another optimizer or exploration change.
