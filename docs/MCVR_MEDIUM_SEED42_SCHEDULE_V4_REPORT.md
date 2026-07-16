# MCVR Medium Seed42 Schedule V4 Report

Decision: **MEDIUM_SEED42_SCHEDULE_V4_FAIL**

Schedule V4 trained from step 0 with 500-step warmup from `2e-5` to `2e-4`, then cosine decay to `2e-5` at step 10000.

## Completion and timing

| Item | Value |
|---|---:|
| Training status | COMPLETED |
| Completed optimizer steps | 10000 / 10000 |
| Pipeline wall seconds | 1572.171 |
| Training wall seconds | 1466.468 |
| Active optimizer seconds | 752.220 |
| Validation seconds | 360.047 |
| Checkpoint I/O seconds | 0.625 |
| Optimizer steps/s | 13.2940 |
| Samples/s | 106.2987 |
| Mean GPU utilization | 13.701% |
| Peak GPU memory used | 3700.0 MiB |

## Checkpoint comparison

| Step | LR | Qualified | Validity delta | Core improvement | Displacement | High-flex validity | Unseen validity |
|---:|---:|---|---:|---:|---:|---:|---:|
| 500 | 0.00020000 | True | -0.045385 | 0.070986 | 0.001392 | -0.046173 | -0.091605 |
| 1000 | 0.00019877 | True | -0.038393 | 0.054944 | 0.000988 | -0.038788 | -0.076863 |
| 1500 | 0.00019512 | True | -0.084689 | 0.099977 | 0.001952 | -0.085599 | -0.188295 |
| 2000 | 0.00018915 | True | -0.063828 | 0.065553 | 0.001426 | -0.063758 | -0.141785 |
| 3000 | 0.00017096 | True | -0.008720 | 0.017010 | 0.000528 | -0.007760 | -0.001733 |
| 5000 | 0.00011743 | True | -0.054249 | 0.047953 | 0.001171 | -0.053845 | -0.127737 |
| 7500 | 0.00004904 | True | -0.065021 | 0.061465 | 0.001265 | -0.066154 | -0.157063 |
| 10000 | 0.00002000 | True | -0.053583 | 0.044598 | 0.001193 | -0.056606 | -0.128833 |

Selected checkpoint: step **1500** (`f94c317f4e12c559058e26f9842317770179ed3e9cbc07c0a21ec681fed94197`).

## Final Gate

| Metric | Upstream | V4 accepted | Delta |
|---|---:|---:|---:|
| Total validity | 0.793989 | 0.709341 | -0.084648 |
| RMSD | 1.321787 | 1.322533 | 0.000745 |
| MAT-P | 1.321787 | 1.322533 | 0.000745 |
| MAT-R | 2.375986 | 2.376936 | 0.000950 |

Conditions: **26/27**. Failed: `['02_one_core_metric_relative_improvement_ge_10pct']`.

LR history contains 200 measurements at 50-step intervals.

Seed43/44 were not executed. 100k and test evaluation were not run.

模型有统计显著且精度非劣的中等有效性，但未达到预注册10%核心改善门槛。

No Rescue V5 or Gate adjustment is permitted.
