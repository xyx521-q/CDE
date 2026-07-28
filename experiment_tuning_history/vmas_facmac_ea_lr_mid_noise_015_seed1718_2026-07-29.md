# VMAS FACMAC-EA action-noise 0.15 seed17/18 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `d3a13abcada4973520de5a684a41f26c17c74f1c`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.15`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,998,632`, the lower final W&B training budget.
- Fresh read-only W&B API checks report both runs as `finished`. The online `scan_history` query did not finish within the bounded wait, so complete local and remote `.wandb` records supplied the late-window values and were cross-checked against the W&B final summaries.

| Run | Name | Seed | W&B path | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `58amhoag` | `facmac_ea_lr_mid_noise_015_seed17__2026-07-28_20-14-36` | 17 | https://wandb.ai/blessingbrandro/gde-vmas/runs/58amhoag | 2,999,959 | 0.984220 | 0.984220 | 0.360373 | 0.0000 | 1.0000 |
| `gwd2ikao` | `facmac_ea_lr_mid_noise_015_seed18__2026-07-28_20-15-26` | 18 | https://wandb.ai/blessingbrandro/gde-vmas/runs/gwd2ikao | 2,998,632 | 0.617921 | 0.350064 | 0.617921 | 0.0000 | 1.0000 |

The equal-budget records contain 1,498 complete evaluation points for each run. Their logging grids differ slightly, so both final-500k windows contain 250 points; the final-100k windows contain 50 seed17 and 51 seed18 points.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `58amhoag` | 1.112145 +/- 0.365412 | 0.936370 +/- 0.385992 | 0.857188 +/- 0.444628 | 0.0000 | 1.0000 | 0.246865 | 0.007224 | 0.3645 |
| Final 500k | `gwd2ikao` | 0.912270 +/- 0.331197 | 0.754536 +/- 0.343745 | 0.670814 +/- 0.456422 | 0.0010 | 0.9990 | 0.204536 | 0.006172 | 0.3330 |
| Final 100k | `58amhoag` | 0.923432 +/- 0.331096 | 0.775092 +/- 0.401999 | 0.624269 +/- 0.387948 | 0.0000 | 1.0000 | 0.282750 | 0.009156 | 0.3050 |
| Final 100k | `gwd2ikao` | 0.860240 +/- 0.284189 | 0.681874 +/- 0.239063 | 0.690281 +/- 0.487354 | 0.0025 | 0.9975 | 0.201184 | 0.006276 | 0.3578 |

## Selection, evidence, and limitations

`58amhoag` (seed17) is selected as directionally better with a **Medium** approval rating. It leads all three final-500k reward means, final-100k best/RL means, contact and package displacement in both windows, and zero-package-progress frequency in the final 100k. Its final best/RL values are also higher.

`gwd2ikao` (seed18) is the worse run for this selection, but it records the pair's only corrected late-window success, has higher final-100k EA reward, lower final-100k best/RL variability, and slightly less zero package progress over the final 500k. The corrected success signal remains rare and both final points have zero success. The comparison is therefore not dominance, differs by seed, and does not establish a causal `act_noise=0.15` advantage. The finished-state and run-identity claims have **High** approval from the current W&B API; the directional selection remains **Medium** because reward/progress and rare success evidence disagree.

## Next runs

The next pair starts from the selected seed17 settings, keeps `lr=2e-4`, `critic_lr=7e-5`, and all shared training settings fixed, and changes only `act_noise` from `0.15` to `0.175` plus new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_0175_seed19` (`seed=19`).
- Local: `facmac_ea_lr_mid_noise_0175_seed20` (`seed=20`).

The 0.025 increment continues the single-dimension exploration test while retaining paired seeds. It tests whether seed17's stronger sustained reward and physical-progress profile can coexist with a repeatable terminated-only success signal.
