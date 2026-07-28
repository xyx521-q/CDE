# VMAS FACMAC-EA higher-noise seed13/14 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `20e6a173a27e37b711d574a0cc4c4f67ce0b432f`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.1`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,999,070`, the lower final W&B budget of the two runs.
- Fresh read-only W&B API checks report both runs as `finished`. Late-window values come from the complete local and remote logs and are cross-checked against the W&B final summaries.

| Run | Name | Seed | W&B path | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `z9uxn4xu` | `facmac_ea_lr_mid_noise_higher_seed14__2026-07-27_23-13-52` | 14 | https://wandb.ai/blessingbrandro/gde-vmas/runs/z9uxn4xu | 2,999,912 | 1.327710 | 0.681861 | 1.327710 | 0.0000 | 1.0000 |
| `02vjz18a` | `facmac_ea_lr_mid_noise_higher_seed13__2026-07-27_23-13-01` | 13 | https://wandb.ai/blessingbrandro/gde-vmas/runs/02vjz18a | 2,999,070 | 0.702520 | 0.702520 | 0.517456 | 0.0000 | 1.0000 |

The equal-budget logs contain 1,499 seed14 and 1,498 seed13 evaluation records through the ceiling. Their logging grids differ slightly, so the final-500k windows contain 249 seed14 and 248 seed13 records, while the final-100k windows contain 49 and 48 records.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `z9uxn4xu` | 0.934671 +/- 0.125731 | 0.757267 +/- 0.152217 | 0.678195 +/- 0.209573 | 0.0012 | 0.9988 | 0.311145 | 0.008837 | 0.2381 |
| Final 500k | `02vjz18a` | 1.059975 +/- 0.173719 | 0.893673 +/- 0.179527 | 0.760249 +/- 0.213079 | 0.0005 | 0.9995 | 0.202729 | 0.006933 | 0.4838 |
| Final 100k | `z9uxn4xu` | 0.943169 +/- 0.114237 | 0.701396 +/- 0.195822 | 0.726431 +/- 0.202886 | 0.0015 | 0.9985 | 0.328786 | 0.009637 | 0.1816 |
| Final 100k | `02vjz18a` | 1.002694 +/- 0.100202 | 0.929283 +/- 0.115807 | 0.707088 +/- 0.148771 | 0.0010 | 0.9990 | 0.208010 | 0.006704 | 0.4510 |

## Selection, evidence, and limitations

`z9uxn4xu` (seed14) is selected as directionally better with a **Medium** approval rating under the corrected task-completion objective. It has higher corrected success in both late windows, substantially stronger contact and package displacement, and much less zero package progress. It also leads final-100k EA reward and the final single-point best/EA values.

`02vjz18a` (seed13) is the worse run for this selection, but it leads all three final-500k reward means and the final-100k best/RL means, with lower final-100k best/RL/EA variability. The result is therefore not dominance and would reverse if sustained reward alone were the objective.

Corrected success remains rare for both runs and is zero at the final point. The two runs differ by seed, their logging grids are not identical, and one pair cannot isolate a causal seed effect. The parent task's `finished` claim has **High** approval because both states and final summaries were independently reproduced through the read-only W&B API; the directional run selection remains **Medium** because reward and task-progress evidence disagree.

## Next runs

Across the existing `act_noise=0.1` replications, task-progress signals are directionally stronger than the preceding `act_noise=0.075` runs, while reward quality remains seed-sensitive. The next pair keeps `lr=2e-4`, `critic_lr=7e-5`, and all shared training settings fixed, changes only `act_noise` from `0.1` to `0.125`, and uses new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_0125_seed15` (`seed=15`).
- Local: `facmac_ea_lr_mid_noise_0125_seed16` (`seed=16`).

This is a narrow exploration test of whether the stronger contact and package-progress signal can be made more repeatable without changing the learning-rate or EA schedule dimensions.
