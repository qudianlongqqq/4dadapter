# Internal-coordinate subspace audit

Jacobians are analytic autograd derivatives checked by central finite differences; periodic wrap is used for torsions. Non-finite and norm≤1e-12 rows are excluded before fixed-tolerance SVD (`rtol=1e-8`, `atol=1e-10`). Raw B/A/T projections are descriptive because they overlap; BA and BAT are joint row-space SVD unions.

| stage   | flex_bin   |   records |   force_norm_median |   B_fraction_median |   A_fraction_median |   T_fraction_median |   BA_union_fraction_median |   BAT_union_fraction_median |   BAT_incremental_fraction_median |   BA_projection_norm_median |   T_projection_norm_median |
|:--------|:-----------|----------:|--------------------:|--------------------:|--------------------:|--------------------:|---------------------------:|----------------------------:|----------------------------------:|----------------------------:|---------------------------:|
| after   | all        |        18 |             0.04764 |             0.95022 |             0.81580 |             0.11766 |                    0.99996 |                     1.00000 |                           0.00003 |                     0.04760 |                    0.00604 |
| before  | all        |        18 |             0.06394 |             0.97339 |             0.82335 |             0.09744 |                    0.99998 |                     1.00000 |                           0.00001 |                     0.06394 |                    0.00740 |
| after   | high_ge_5  |         6 |             0.04764 |             0.95022 |             0.78327 |             0.19432 |                    0.99958 |                     0.99999 |                           0.00038 |                     0.04760 |                    0.00915 |
| after   | low_0_2    |         6 |             0.06119 |             0.97451 |             0.81440 |             0.05658 |                    0.99999 |                     1.00000 |                           0.00000 |                     0.06118 |                    0.00335 |
| after   | medium_3_4 |         6 |             0.03994 |             0.93955 |             0.83374 |             0.11766 |                    0.99999 |                     1.00000 |                           0.00001 |                     0.03994 |                    0.00523 |
| before  | high_ge_5  |         6 |             0.06822 |             0.97332 |             0.78531 |             0.20555 |                    0.99980 |                     1.00000 |                           0.00020 |                     0.06813 |                    0.01308 |
| before  | low_0_2    |         6 |             0.07092 |             0.97604 |             0.83325 |             0.05028 |                    0.99999 |                     1.00000 |                           0.00000 |                     0.07092 |                    0.00381 |
| before  | medium_3_4 |         6 |             0.05689 |             0.96964 |             0.82640 |             0.09734 |                    1.00000 |                     1.00000 |                           0.00000 |                     0.05689 |                    0.00584 |

Finite-difference maximum absolute errors:

| family   |   count |   median |      max |
|:---------|--------:|---------:|---------:|
| A        |      18 | 7.16e-11 | 1.3e-10  |
| B        |      18 | 1.88e-11 | 4.79e-11 |
| T        |      18 | 7.38e-11 | 1.1e-10  |
