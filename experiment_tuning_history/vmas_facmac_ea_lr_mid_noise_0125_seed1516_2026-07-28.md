# VMAS FACMAC-EA action-noise 0.125 seed15/16 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `4d37497865d1ef3c4a5aa7f18b7dd0cae1c29df2`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.125`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,998,886`, the lower final W&B training budget.
- Fresh read-only W&B API checks report both runs as `finished`. Complete local and remote `.wandb` records supplied the late-window values and were cross-checked against the final summaries and complete logs. The remote `scan_history` endpoint did not return within the bounded wait and was terminated.

| Run | Name | Seed | W&B path | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gxu28q2q` | `facmac_ea_lr_mid_noise_0125_seed16__2026-07-28_09-19-56` | 16 | https://wandb.ai/blessingbrandro/gde-vmas/runs/gxu28q2q | 3,000,006 | 0.861401 | 0.861401 | 0.681068 | 0.0000 | 1.0000 |
| `ydb3k9vk` | `facmac_ea_lr_mid_noise_0125_seed15__2026-07-28_09-19-24` | 15 | https://wandb.ai/blessingbrandro/gde-vmas/runs/ydb3k9vk | 2,998,886 | 1.040753 | 1.040753 | 0.765565 | 0.0000 | 1.0000 |

The equal-budget records contain 1,499 complete evaluation points for each run. Their logging grids differ slightly, so the final-500k windows contain 249 seed16 and 250 seed15 points; both final-100k windows contain 50 points.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `gxu28q2q` | 1.093239 +/- 0.330519 | 0.902664 +/- 0.354814 | 0.860771 +/- 0.449311 | 0.0000 | 1.0000 | 0.272545 | 0.008393 | 0.2892 |
| Final 500k | `ydb3k9vk` | 1.170559 +/- 0.330119 | 0.982541 +/- 0.376145 | 0.889623 +/- 0.458953 | 0.0005 | 0.9995 | 0.199231 | 0.006498 | 0.4215 |
| Final 100k | `gxu28q2q` | 1.065370 +/- 0.310462 | 0.895255 +/- 0.278910 | 0.829479 +/- 0.550671 | 0.0000 | 1.0000 | 0.224350 | 0.006636 | 0.3800 |
| Final 100k | `ydb3k9vk` | 1.175029 +/- 0.313157 | 0.953351 +/- 0.382185 | 0.885017 +/- 0.527994 | 0.0025 | 0.9975 | 0.184206 | 0.006718 | 0.4300 |

## Selection, evidence, and limitations

`ydb3k9vk` (seed15, Run B) is selected as directionally better with a **Medium** approval rating. It leads all final-500k and final-100k best/RL/EA reward means, records the pair's only corrected late-window success, and has the stronger final summary. The parent task's claim that both runs are finished has **High** approval because the state, config, and summaries were independently reproduced through the current read-only W&B API.

`gxu28q2q` (seed16, Run A) is the worse run for this selection, but it has higher contact in both windows, higher final-500k package displacement, and less zero package progress. Seed15's final-100k success mean of 0.0025 represents only one successful evaluation episode in that window, while both final points still have zero success. The comparison is therefore not dominance, differs by seed, and cannot establish that `act_noise=0.125` itself caused the result. No historical or parent-agent claim is accepted beyond the independently verified finished state and run identity; no material claim was rejected, and causal superiority remains unresolved.

## Next runs

The next pair starts from the selected seed15 settings, keeps `lr=2e-4`, `critic_lr=7e-5`, and all shared training settings fixed, and changes only `act_noise` from `0.125` to `0.15` plus new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_015_seed17` (`seed=17`).
- Local: `facmac_ea_lr_mid_noise_015_seed18` (`seed=18`).

The 0.025 noise increment is a narrow test of whether the rare terminated-only success and stronger reward profile can repeat with slightly more exploration. The paired seeds are still required because the task-progress metrics remain seed-sensitive.
