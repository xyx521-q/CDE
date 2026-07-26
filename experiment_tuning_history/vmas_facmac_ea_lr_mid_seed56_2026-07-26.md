# VMAS FACMAC-EA mid-LR seed5/6 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `97fb00bcb8b99f0116688b1987bbcbbfb498f112`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.05`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,999,158`, the lower final budget of the two runs.

| Run | Config | Seed | W&B state | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zs3mdrr3` | `facmac_ea_lr_mid_seed6` | 6 | finished | 2,999,158 | 1.315526 | 0.724973 | 1.315526 | 0.0000 | 1.0000 |
| `19l0b8k0` | `facmac_ea_lr_mid_seed5` | 5 | finished | 2,999,939 | 0.796241 | 0.796241 | 0.793395 | 0.0000 | 1.0000 |

The W&B history query returned 1,494 seed6 and 1,493 seed5 evaluation points. The equal-budget late windows contain 250 points over the final 500k steps and 50 points over the final 100k steps for each run.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `zs3mdrr3` | 0.940425 +/- 0.301920 | 0.794715 +/- 0.314833 | 0.673426 +/- 0.452690 | 0.0010 | 0.9990 | 0.185712 | 0.005744 |
| Final 500k | `19l0b8k0` | 1.021657 +/- 0.275704 | 0.910019 +/- 0.303807 | 0.764078 +/- 0.394344 | 0.0000 | 1.0000 | 0.121895 | 0.003636 |
| Final 100k | `zs3mdrr3` | 1.017079 +/- 0.358150 | 0.808038 +/- 0.288668 | 0.843317 +/- 0.437324 | 0.0025 | 0.9975 | 0.194654 | 0.006331 |
| Final 100k | `19l0b8k0` | 0.985687 +/- 0.257486 | 0.893371 +/- 0.262013 | 0.723036 +/- 0.438790 | 0.0000 | 1.0000 | 0.085975 | 0.003096 |

## Selection and next runs

`zs3mdrr3` (seed6) is selected as the better run. It is the only run with non-zero corrected success in the late windows, has substantially higher contact rate and package displacement, and leads on final best/EA reward and final-100k best/EA means. These task-progress signals take precedence over reward alone.

`19l0b8k0` (seed5) is the worse run for this selection, but it has higher and less variable final-500k best, RL, and EA rewards and a higher final-100k RL mean. The selection is therefore directional rather than evidence that seed6 dominates every reward metric. Both success rates remain effectively zero.

The next pair keeps the selected mid learning rates and all training settings fixed, changes only action noise from `0.05` to `0.075`, and uses new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_high_seed7` (`seed=7`).
- Local: `facmac_ea_lr_mid_noise_high_seed8` (`seed=8`).

This tests whether modestly broader exploration can turn seed6's contact and displacement signal into repeatable task termination. The historical low-noise condition is only a provisional pre-correction prior, but it does not support reducing noise for this follow-up.
