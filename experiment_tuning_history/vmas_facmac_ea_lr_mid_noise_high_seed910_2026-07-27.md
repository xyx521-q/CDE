# VMAS FACMAC-EA higher-noise seed9/10 comparison

## Comparable runs

- Corrected metric contract: success means VMAS `terminated`; a pure time-limit `truncated` episode is not success.
- Launch revision: `32d8c834981b0efeeefef8be0432b06c0aae9d29`.
- Shared settings: `lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.075`, `t_max=3e6`; only the seed differs.
- Equal-budget ceiling: `train/t_env=2,998,810`, the lower final W&B budget of the two runs.
- The W&B history pagination query did not finish within the bounded check. Late-window values below come from the complete local and remote logs and are cross-checked against the explicit W&B `finished` states and final summaries.

| Run | Config | Seed | W&B state | Final `t_env` | Final best | Final RL | Final EA | Final success | Final episode limit |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yisli9vg` | `facmac_ea_lr_mid_noise_high_seed10` | 10 | finished | 2,998,810 | 0.940152 | 0.940152 | 0.868907 | 0.0000 | 1.0000 |
| `9sfwn2cq` | `facmac_ea_lr_mid_noise_high_seed9` | 9 | finished | 3,000,023 | 1.220532 | 0.867527 | 1.220532 | 0.0000 | 1.0000 |

The equal-budget logs contain 1,498 seed10 and 1,499 seed9 evaluation records. Because their logging grids differ slightly, the final-500k windows contain 248 seed10 and 249 seed9 records, and the final-100k windows contain 49 records per run.

| Window | Run | Best mean +/- std | RL mean +/- std | EA mean +/- std | Success | Episode limit | Contact rate | Package displacement | Zero package progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final 500k | `yisli9vg` | 1.028351 +/- 0.171819 | 0.847646 +/- 0.184856 | 0.708506 +/- 0.217181 | 0.0019 | 0.9981 | 0.223021 | 0.007665 | 0.3841 |
| Final 500k | `9sfwn2cq` | 1.047850 +/- 0.153431 | 0.851904 +/- 0.173775 | 0.770391 +/- 0.208130 | 0.0010 | 0.9990 | 0.230559 | 0.007337 | 0.4052 |
| Final 100k | `yisli9vg` | 0.956892 +/- 0.181133 | 0.747737 +/- 0.162047 | 0.662269 +/- 0.230561 | 0.0015 | 0.9985 | 0.275824 | 0.009392 | 0.2883 |
| Final 100k | `9sfwn2cq` | 1.104469 +/- 0.130531 | 0.855812 +/- 0.126659 | 0.881300 +/- 0.188029 | 0.0000 | 1.0000 | 0.223524 | 0.007418 | 0.3704 |

## Selection and next runs

`yisli9vg` (seed10) is selected as the directionally better run because it retains non-zero corrected success in both late windows and has substantially stronger final-100k contact rate, package displacement, and non-zero package progress. These task-completion and task-progress signals take precedence over reward alone.

`9sfwn2cq` (seed9) is the worse run for this selection, but it leads every final-100k reward mean, is less variable there, and has slightly stronger final-500k reward and contact means. The comparison therefore does not establish dominance. Corrected success remains effectively zero for both and no final-100k seed9 evaluation terminates successfully.

The next pair keeps the selected learning rates and all training settings fixed, changes only action noise from `0.075` to `0.1`, and uses new seeds:

- Remote 715: `facmac_ea_lr_mid_noise_higher_seed11` (`seed=11`).
- Local: `facmac_ea_lr_mid_noise_higher_seed12` (`seed=12`).

This is a narrow exploration test: it checks whether slightly broader action noise can make seed10's contact, displacement, and rare termination signal more repeatable before changing another hyperparameter dimension.
