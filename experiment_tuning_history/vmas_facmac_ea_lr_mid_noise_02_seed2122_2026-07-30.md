# VMAS FACMAC-EA action-noise 0.2 seed21/22 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `03c371444f18b1e59f0a4ad1794ae6d8abffa413`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.2`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,999,810`, the lower final W&B training budget.
- Fresh read-only W&B API checks report both runs as `finished`. The online `scan_history` query did not finish within the bounded wait and was terminated. Complete local and remote `.wandb` records supplied the equal-budget values and were cross-checked against current W&B summaries, run configs, and complete logs.

| Run | Name | Seed | W&B path | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1uw68kwj` | `facmac_ea_lr_mid_noise_02_seed22__2026-07-29_18-21-46` | 22 | https://wandb.ai/blessingbrandro/gde-vmas/runs/1uw68kwj | 3,000,037 | 1.050458 | 1.050458 | 0.973248 | 0.0000 | 1.0000 |
| `2jr7vq0q` | `facmac_ea_lr_mid_noise_02_seed21__2026-07-29_18-21-28` | 21 | https://wandb.ai/blessingbrandro/gde-vmas/runs/2jr7vq0q | 2,999,810 | 0.906286 | 0.743598 | 0.906286 | 0.0000 | 1.0000 |

The equal-budget records contain 1,493 complete evaluation points for seed22 and 1,494 for seed21. The final-500k windows contain 249 and 250 points; the final-100k windows contain 50 and 51 points, respectively.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `1uw68kwj` | 1.074015 +/- 0.343570 | 0.905243 +/- 0.320863 | 0.840057 +/- 0.467703 | 0.000502 | 0.999498 | 0.175668 | 0.005352 | 0.473394 |
| Final 500k | `2jr7vq0q` | 0.780941 +/- 0.402893 | 0.572255 +/- 0.465544 | 0.484090 +/- 0.543868 | 0.000500 | 0.999500 | 0.265037 | 0.009971 | 0.364000 |
| Final 100k | `1uw68kwj` | 1.097752 +/- 0.334110 | 0.918406 +/- 0.359307 | 0.767525 +/- 0.502484 | 0.000000 | 1.000000 | 0.191675 | 0.005820 | 0.442500 |
| Final 100k | `2jr7vq0q` | 0.838413 +/- 0.288926 | 0.687797 +/- 0.243279 | 0.624084 +/- 0.488862 | 0.000000 | 1.000000 | 0.228922 | 0.006970 | 0.446078 |

## Selection, evidence, and limitations

`1uw68kwj` (seed22) is selected as directionally better with a **Medium** approval rating. It leads all three reward means in both equal-budget late windows, with lower final-500k variability for all three reward metrics. Its final W&B best, RL, and EA values are also higher.

`2jr7vq0q` (seed21) is the worse run for this selection, but it has substantially stronger final-500k contact, package displacement, and non-zero package progress, plus slightly stronger final-100k contact and displacement. Each run records only one non-zero corrected success evaluation in the final 500k, and neither records corrected success in the final 100k or final point. The result is not dominance, differs only by seed, and cannot establish a causal `act_noise=0.2` advantage.

The current W&B API, run configs, complete local/remote records, and logs give the run identity, explicit `finished` states, shared settings, and numerical window calculations **High** approval. The seed22 selection remains **Medium** because reward evidence conflicts with seed21's stronger physical-progress signals and reliable task completion is absent. A universal or causal claim about action noise has **Low** approval.

## Next runs

The next pair starts from the selected seed22 settings, keeps `lr=2e-4`, `critic_lr=7e-5`, and all shared training settings fixed, and changes only `act_noise` from `0.2` to `0.225` plus new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_0225_seed23` (`seed=23`).
- Local: `facmac_ea_lr_mid_noise_0225_seed24` (`seed=24`).

The 0.025 increment is the smallest established step in the current single-dimension action-noise sequence. It tests whether seed22's stronger sustained reward profile can retain physical progress and produce repeatable terminated-only success across a new paired replication.
