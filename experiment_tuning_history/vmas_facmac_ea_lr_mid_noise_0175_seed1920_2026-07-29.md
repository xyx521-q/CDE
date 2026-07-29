# VMAS FACMAC-EA action-noise 0.175 seed19/20 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `41b21af88e9e9e1ea83e808a2c8da2261fb04acf`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.175`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,998,752`, the lower final W&B training budget.
- Fresh read-only W&B API checks report both runs as `finished`. The online `scan_history` query did not finish within the bounded wait and was terminated. Complete local and remote `.wandb` records supplied the late-window values and were cross-checked against the current W&B summaries, run configs, and complete logs.

| Run | Name | Seed | W&B path | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `f3s1yl6z` | `facmac_ea_lr_mid_noise_0175_seed20__2026-07-29_07-14-40` | 20 | https://wandb.ai/blessingbrandro/gde-vmas/runs/f3s1yl6z | 2,998,752 | 0.825873 | 0.806952 | 0.825873 | 0.0000 | 1.0000 |
| `6jgyqxmw` | `facmac_ea_lr_mid_noise_0175_seed19__2026-07-29_07-13-31` | 19 | https://wandb.ai/blessingbrandro/gde-vmas/runs/6jgyqxmw | 2,999,730 | 1.201313 | 0.774660 | 1.201313 | 0.0000 | 1.0000 |

The equal-budget records contain 1,493 complete evaluation points for each run. Both final-500k and final-100k windows contain 250 and 50 points per run, respectively.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `f3s1yl6z` | 1.078740 +/- 0.284489 | 0.927213 +/- 0.304225 | 0.861565 +/- 0.381823 | 0.0015 | 0.9985 | 0.181100 | 0.004910 | 0.4620 |
| Final 500k | `6jgyqxmw` | 1.140952 +/- 0.316485 | 0.986907 +/- 0.329572 | 0.885204 +/- 0.444411 | 0.0000 | 1.0000 | 0.198070 | 0.005770 | 0.3580 |
| Final 100k | `f3s1yl6z` | 1.075911 +/- 0.284996 | 0.933301 +/- 0.325168 | 0.826658 +/- 0.341892 | 0.0000 | 1.0000 | 0.187725 | 0.005375 | 0.4425 |
| Final 100k | `6jgyqxmw` | 1.132984 +/- 0.323270 | 0.980281 +/- 0.302129 | 0.875784 +/- 0.460163 | 0.0000 | 1.0000 | 0.172375 | 0.005020 | 0.3775 |

## Selection, evidence, and limitations

`6jgyqxmw` (seed19) is selected as directionally better with a **Medium** approval rating. It leads all three reward means in both equal-budget late windows. It also has stronger final-500k contact, package displacement, and non-zero package progress, plus less zero package progress in the final 100k. Its final best and EA values are higher.

`f3s1yl6z` (seed20) is the worse run for this selection, but it records the pair's only corrected late-window success: three non-zero success evaluations in the final 500k. It also has higher final-100k contact and displacement, lower best/EA variability in both windows, and a higher final RL point. Both final-100k success rates and both final points are zero, so the rare seed20 success signal does not establish reliable task completion. The result is not dominance, differs by seed, and cannot establish a causal `act_noise=0.175` advantage.

The current W&B API, run configs, local/remote records, and logs give the run identity, explicit `finished` states, shared settings, and numerical window calculations **High** approval. The seed19 selection remains **Medium** because sustained reward/progress evidence conflicts with seed20's rare terminated-only success. A universal or causal claim about action noise has **Low** approval. No old-record or other-agent conclusion was accepted without current evidence; no material current-data claim was rejected, and causal superiority remains unresolved.

## Next runs

The next pair starts from the selected seed19 settings, keeps `lr=2e-4`, `critic_lr=7e-5`, and all shared training settings fixed, and changes only `act_noise` from `0.175` to `0.2` plus new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_02_seed21` (`seed=21`).
- Local: `facmac_ea_lr_mid_noise_02_seed22` (`seed=22`).

The 0.025 increment continues the existing single-dimension exploration sequence and tests whether seed19's stronger sustained reward/progress profile can retain or improve terminated-only success across a new paired replication.
