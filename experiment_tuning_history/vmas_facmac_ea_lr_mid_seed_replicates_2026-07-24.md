# VMAS FACMAC-EA mid-LR seed replications

## Metric scope

- Runs `8bs2laau` and `ftcyl8br` use the pre-correction `success_rate` semantics, so their reported success values are not evidence of transport success.
- This comparison uses reward metrics at equal training budgets. Corrected-metric runs belong in `vmas_facmac_ea_success_rate_corrected.md` and must not be compared directly on `success_rate` with these runs.
- Both runs use the same mid-learning-rate configuration (`lr=2e-4`, `critic_lr=7e-5`, default `act_noise=0.05`) and differ only by seed.

## Comparable results

| Run | Seed | Final `t_env` | Final best | Final RL | Final EA |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ftcyl8br` | 1 | 2,998,981 | 0.873241 | 0.873241 | 0.448192 |
| `8bs2laau` | 2 | 2,999,129 | 1.098854 | 0.761414 | 1.098854 |

Over the final 500k environment steps (250 logged evaluation points per run):

| Run | Best mean +/- std | RL mean +/- std | EA mean +/- std |
| --- | ---: | ---: | ---: |
| `ftcyl8br` | 0.785684 +/- 0.115092 | 0.659863 +/- 0.132417 | 0.576474 +/- 0.161364 |
| `8bs2laau` | 0.741491 +/- 0.150505 | 0.613089 +/- 0.143900 | 0.489825 +/- 0.203474 |

## Selection and uncertainty

`ftcyl8br` is selected as the stronger replication because all three final-500k reward means are higher and less variable. `8bs2laau` has the stronger final best and EA points, and its final-100k best and RL means are higher, so the selection favors sustained late-window stability rather than a final-point peak. The two-run sample is insufficient to claim seed-independent superiority.

The next runs retain the selected mid learning rates and default action noise while adding two independent replications:

- Remote 715: `facmac_ea_lr_mid_seed3` (`seed=3`).
- Local: `facmac_ea_lr_mid_seed4` (`seed=4`).
