# VMAS FACMAC-EA higher-noise seed7/8 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `6dbb93c8c63c89340d19a05862146c0f49d8071b`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.075`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,998,162`, the lower final budget of the two runs.
- The W&B history pagination query timed out. Late-window values below come from the complete local and remote logs and are cross-checked against the explicit W&B `finished` states and final summaries.

| Run | Config | Seed | W&B state | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `awp34v9x` | `facmac_ea_lr_mid_noise_high_seed8` | 8 | finished | 2,998,162 | 1.208732 | 0.813094 | 1.208732 | 0.0000 | 1.0000 |
| `wpbuobx7` | `facmac_ea_lr_mid_noise_high_seed7` | 7 | finished | 2,999,830 | 1.410818 | 1.410818 | 0.806456 | 0.0000 | 1.0000 |

The equal-budget logs contain 1,473 evaluation records per run. The final-500k windows contain 250 records per run; the final-100k windows contain 51 seed8 records and 50 seed7 records because their logging grids differ slightly.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `awp34v9x` | 1.183060 +/- 0.167388 | 1.030434 +/- 0.181729 | 0.889076 +/- 0.201283 | 0.0010 | 0.9990 | 0.184239 | 0.006261 | 0.5048 |
| Final 500k | `wpbuobx7` | 1.077118 +/- 0.172089 | 0.914990 +/- 0.185954 | 0.762674 +/- 0.227607 | 0.0009 | 0.9991 | 0.206067 | 0.008014 | 0.3710 |
| Final 100k | `awp34v9x` | 1.217382 +/- 0.169693 | 1.095896 +/- 0.185652 | 0.889210 +/- 0.149034 | 0.0000 | 1.0000 | 0.196029 | 0.006553 | 0.4824 |
| Final 100k | `wpbuobx7` | 1.212888 +/- 0.132827 | 1.065834 +/- 0.121470 | 0.883730 +/- 0.206857 | 0.0000 | 1.0000 | 0.182976 | 0.007092 | 0.4465 |

## Selection and next runs

`awp34v9x` (seed8) is selected as the better run. It leads all final-500k reward means, has lower reward variability in that window, and narrowly leads all final-100k reward means. Its final best and RL points are lower than seed7, so the selection favors sustained equal-budget late-window performance rather than one final evaluation.

`wpbuobx7` (seed7) is the worse run for this selection, but it has stronger final-500k contact rate and package displacement, less zero package progress, and higher final best/RL points. The two runs therefore do not establish dominance. Corrected success remains effectively zero for both: about `0.001` in the final 500k and exactly zero in the final 100k, with pure truncation near one.

Compared with the preceding `act_noise=0.05` seed5/6 pair, the `act_noise=0.075` pair is directionally stronger on sustained late reward and task-progress signals, but the comparison is confounded by new seeds and does not establish a causal noise effect. The next pair retains the selected learning rates, `act_noise=0.075`, and all training settings, changing only the seed:

- Remote 715: `facmac_ea_lr_mid_noise_high_seed9` (`seed=9`).
- Local: `facmac_ea_lr_mid_noise_high_seed10` (`seed=10`).

These replications test whether the higher-noise condition's reward and package-progress gains persist before changing another hyperparameter dimension.
