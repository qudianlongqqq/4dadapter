# LSGO GFN2-xTB single-point report

All values are paired `(E_out-E_source) × 627.509474` kcal/mol. GFN2-xTB 6.7.1 was used in strict single-point mode; no geometry optimization was run.

| method      | paired_success | mean_dE   | median_dE | improved | p75       | p90       | p95      | p99      | maximum   | positive_tail_mean | failure |
| ----------- | -------------- | --------- | --------- | -------- | --------- | --------- | -------- | -------- | --------- | ------------------ | ------- |
| A-G         | 600            | -0.658777 | -0.47517  | 0.868333 | -0.240638 | 0         | 0.090186 | 0.310344 | 0.616318  | 0.172552           | 0       |
| B-G_seed173 | 600            | -0.961755 | -0.79888  | 0.938333 | -0.571994 | -0.244024 | 0        | 0        | 0         | 0                  | 0       |
| B-G_seed181 | 600            | -0.974126 | -0.80828  | 0.938333 | -0.575126 | -0.280009 | 0        | 0        | 0.0288835 | 0.0288835          | 0       |
| B-G_seed193 | 600            | -0.966903 | -0.806417 | 0.931667 | -0.567182 | -0.240859 | 0        | 0        | 0.0948245 | 0.0423798          | 0       |

B-G across seeds: median ΔE = -0.804526 ± 0.004977; mean ΔE = -0.967595 ± 0.006215; improved fraction = 93.611% ± 0.385% (sample SD). Worst p95/p99/max = 0.000000/0.000000/0.094825 kcal/mol. All 3,000 jobs succeeded; failure/timeout/nonfinite=0/0/0.
