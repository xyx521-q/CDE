# VMAS FACMAC-EA higher-noise seed11/12 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `bc87a9f5554b07b72b3c2ffca97362d25fd0960c`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.1`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,998,899`, the lower final budget of the two runs.
- W&B reports both runs as `finished`. The bounded full-history API query timed out, so late-window values come from the complete local and remote logs and are cross-checked against the W&B final summaries.

| Run | Name | Seed | W&B path | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ovfbhunz` | `facmac_ea_lr_mid_noise_higher_seed12__2026-07-27_12-21-13` | 12 | https://wandb.ai/blessingbrandro/gde-vmas/runs/ovfbhunz | 2,998,899 | 0.690597 | 0.236898 | 0.690597 | 0.0000 | 1.0000 |
| `kk75h9km` | `facmac_ea_lr_mid_noise_higher_seed11__2026-07-27_12-20-58` | 11 | https://wandb.ai/blessingbrandro/gde-vmas/runs/kk75h9km | 2,999,125 | 0.808008 | 0.808008 | 0.530546 | 0.0000 | 1.0000 |

The equal-budget logs contain 1,498 seed12 and 1,498 seed11 evaluation records through the ceiling. Each final-500k and final-100k window contains 250 and 50 records per run, respectively.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `ovfbhunz` | 1.163792 +/- 0.167835 | 0.967434 +/- 0.197624 | 0.848139 +/- 0.195945 | 0.0013 | 0.9987 | 0.271948 | 0.008732 | 0.2140 |
| Final 500k | `kk75h9km` | 1.000388 +/- 0.128724 | 0.849224 +/- 0.141008 | 0.745948 +/- 0.214883 | 0.0008 | 0.9992 | 0.237719 | 0.006364 | 0.4765 |
| Final 100k | `ovfbhunz` | 1.132750 +/- 0.170275 | 0.951810 +/- 0.226914 | 0.806774 +/- 0.173480 | 0.0015 | 0.9985 | 0.272206 | 0.008598 | 0.2305 |
| Final 100k | `kk75h9km` | 0.960620 +/- 0.150829 | 0.828984 +/- 0.158360 | 0.624342 +/- 0.305094 | 0.0015 | 0.9985 | 0.255864 | 0.007216 | 0.4425 |

## Selection and next runs

`ovfbhunz` (seed12) is selected as directionally better with a **High** approval rating. It leads every final-500k and final-100k reward mean, has higher final-500k corrected success, and has stronger contact, package displacement, and non-zero package progress in both windows.

`kk75h9km` (seed11) is the worse run for this selection. Its final single-point best and RL rewards are higher, and its best/RL reward variability is lower, so the comparison does not establish dominance. Corrected success remains rare for both runs and exactly zero at the final point; the result is still seed-sensitive.

The next pair keeps the selected `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.1`, and all shared training settings fixed, changing only the seed:

- Remote 715: `facmac_ea_lr_mid_noise_higher_seed13` (`seed=13`).
- Local: `facmac_ea_lr_mid_noise_higher_seed14` (`seed=14`).

These replications test whether seed12's sustained reward and package-progress gains persist before changing another hyperparameter dimension.
