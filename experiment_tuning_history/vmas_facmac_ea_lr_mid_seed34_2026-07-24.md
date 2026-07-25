# VMAS FACMAC-EA mid-LR seed3/4 comparison

## Metric scope

- Runs `9grzlz7x` and `kmb8m6iu` use the same mid-learning-rate algorithm settings (`lr=2e-4`, `critic_lr=7e-5`, `act_noise=0.05`) and differ by seed.
- Local run `9grzlz7x` uses the corrected VMAS success semantics, while remote run `kmb8m6iu` uses the pre-correction semantics. Their `success_rate` values are not comparable and are excluded from selection.
- Reward comparison uses equal approximately 3M-step budgets. Final-point values come from W&B summaries; late-window values come from the complete local and remote logs because the full W&B history query timed out.

## Comparable results

| Run | Seed | Final `t_env` | Final best | Final RL | Final EA |
| --- | ---: | ---: | ---: | ---: | ---: |
| `9grzlz7x` | 4 | 2,998,833 | 1.103158 | 0.400351 | 1.103158 |
| `kmb8m6iu` | 3 | 2,998,986 | 0.928536 | 0.882603 | 0.928536 |

Over the final 500k environment steps (250 logged evaluation points per run):

| Run | Best mean +/- std | RL mean +/- std | EA mean +/- std |
| --- | ---: | ---: | ---: |
| `9grzlz7x` | 1.099728 +/- 0.145648 | 0.939625 +/- 0.147047 | 0.821494 +/- 0.209275 |
| `kmb8m6iu` | 0.666306 +/- 0.130108 | 0.486856 +/- 0.145109 | 0.430104 +/- 0.202616 |

Over the final 100k environment steps (50 logged evaluation points per run):

| Run | Best mean +/- std | RL mean +/- std | EA mean +/- std |
| --- | ---: | ---: | ---: |
| `9grzlz7x` | 1.049286 +/- 0.115575 | 0.911726 +/- 0.136488 | 0.701722 +/- 0.202292 |
| `kmb8m6iu` | 0.780810 +/- 0.106080 | 0.564678 +/- 0.107803 | 0.565222 +/- 0.186124 |

## Selection and follow-up

`9grzlz7x` is selected as the better run because its final-500k and final-100k best, RL, and EA reward means are all higher. `kmb8m6iu` has the stronger final RL point, so the conclusion favors sustained late-window performance rather than one final evaluation.

The large seed spread and the single corrected-metric result do not support changing another hyperparameter yet. The next runs retain the selected mid learning rates and action noise while adding two replications:

- Remote 715: `facmac_ea_lr_mid_seed5` (`seed=5`).
- Local: `facmac_ea_lr_mid_seed6` (`seed=6`).
