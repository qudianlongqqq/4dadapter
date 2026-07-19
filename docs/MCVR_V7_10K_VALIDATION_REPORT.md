# MCVR V7 10K Development Validation Report

## Decision

**V7_READY_FOR_FORMAL_LARGE**

This validation used the frozen D1 checkpoint, V5-B comparator, V7 method,
evaluator, thresholds, and seed. No training, target materialization, test,
formal test, or frozen-holdout record access occurred.

## Frozen V7 method

V7 is the constraint-specific hybrid validated in the 512-molecule study. It
retains the frozen D1 Cartesian prior as the Bond operator, applies the fixed
damped/truncated-SVD analytic Jacobian only to active Angle residuals, and uses
the fixed spatial repulsion operator for Clash. Their corrections are combined
by non-learned constraint-aware normalized fusion and passed through the same
BAC safety/backtracking evaluator. This run did not change the architecture,
checkpoint, thresholds, hidden size, layers, loss, or fusion rule.

## Configuration and data identity

- Seed: `43018`
- Bootstrap draws: `10000` molecule-level paired resamples
- Cohort policy: `train-derived-unseen-development`
- Molecules: `10000`
- Records: `30000`
- Manifest identity: `764d7a19fe40d6795553b37291efebd0e62ad604c54ba4c89a9dcdd0bb5705fc`
- Manifest file SHA256: `a01571daf0cf337f105a19e32632bdae11d6b1840b6a6f4d0d2e05baa001c435`
- Ordered molecule identity: `17f19269598d7985b16bd0beb82f8e00f0401b2a44ba91c42b631bdc8489bf78`
- Ordered sample identity: `880c68ced3e8f3e74b9aa44a207ea1abbc0715776e542298d06514840695c0a3`
- Paired cohort identity: `4f3f4dcaf398aed513a1c74e4548f65c1689556d979e672d284eb9c91d8ff1f0`
- D1 checkpoint SHA256: `9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`
- D1-training overlap: `0` molecules
- Validation-tune overlap: `0` molecules
- Frozen-holdout overlap: `0` molecules

The formal validation split has only 5,000 molecules, including the protected
1,000-molecule holdout. The 10K cohort was therefore selected deterministically
from the existing train source/target pool after excluding every D1-training,
validation-tune, and frozen-holdout molecule. This is an unseen development
evaluation, not a formal-large full scan.

## Results

| Method | Bond delta | Angle delta | Active Angle delta | Clash delta | Ring delta | RMSD delta | MAT-P delta | MAT-R delta | COV-P delta | COV-R delta | Acceptance | Rollback | Mean displacement (A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | -0.093574 | -0.002481 | -0.004572 | -3.916e-10 | -0.010890 | 0.000393 | 0.000393 | 0.000652 | -0.000200 | -0.000154 | 96.98% | 3.02% | 0.006691 |
| V5-B | -0.095549 | -0.003184 | -0.005860 | -1.111e-08 | -0.012985 | 0.000389 | 0.000389 | 0.000643 | -0.000233 | -0.000153 | 97.39% | 2.61% | 0.007349 |
| V7 | -0.092312 | -0.004105 | -0.007362 | -1.885e-08 | -0.011430 | 0.000386 | 0.000386 | 0.000642 | -0.000200 | -0.000145 | 96.86% | 3.14% | 0.007062 |

## Paired bootstrap

- V7-minus-D1 Active Angle: -0.002790 (95% CI [-0.002915, -0.002671])
- V7-minus-D1 Bond: 0.001262 (95% CI [0.001081, 0.001441])
- V7-minus-D1 Acceptance: -0.001200 (95% CI [-0.002467, 0.000033])
- V7-minus-D1 displacement: 0.000371 (95% CI [0.000345, 0.000397])
- V7-minus-V5-B Active Angle: -0.001502 (95% CI [-0.001590, -0.001416])
- V7-minus-V5-B Bond: 0.003237 (95% CI [0.003024, 0.003452])
- V7-minus-V5-B Acceptance: -0.005233 (95% CI [-0.006533, -0.003966])
- V7-minus-V5-B displacement: -0.000287 (95% CI [-0.000320, -0.000254])

All intervals use the same molecule resample indices across metrics within a
comparison/subset. This preserves paired covariance and avoids record-level
pseudoreplication.

## V7 Angle solver stability

- Calls: `120000`
- Solved: `62172`
- Inactive: `57828`
- Failures: `0`
- Failure rate: `0.000000%`
- Mean condition number: `3.078657`
- Maximum condition number: `7879.276795`
- Mean effective rank: `2.280046`
- Truncated directions: `251`

## Admission checks

- `active_angle_gain_ci95_high_lt_zero`: `true`
- `bond_degradation_vs_d1_lt_0.005`: `true`
- `movement_ratio_vs_d1_lt_1.1`: `true`
- `acceptance_drop_vs_d1_lt_0.05`: `true`
- `ring_non_regressed`: `true`
- `chirality_non_regressed`: `true`
- `rmsd_noninferior_0.0001`: `true`
- `cov_p_non_regressed`: `true`
- `cov_r_non_regressed`: `true`
- `solver_failure_rate_zero`: `true`

Movement ratio V7/D1: `1.055457`.
Acceptance drop D1-V7: `0.001200`.
Bond degradation V7-D1: `0.001262`.

## Interpretation

The Active-Angle gain over D1 remains statistically significant at 10K scale,
and V7 also improves Active Angle over V5-B. Relative to D1, the Bond delta is
weaker by only `0.001262`, below the frozen
`0.005` margin. The gain is not explained by unrestricted movement: the V7/D1
movement ratio is `1.055457`, below `1.1`, while
the acceptance drop is only `0.001200`. Ring,
chirality, RMSD, and COV admission checks are non-regressed. The Angle solver
completed all `120000` calls with zero failures. These results
support the constraint-specific correction-manifold hypothesis on this frozen
10K unseen development cohort and satisfy the predeclared formal-large
admission gate. They do not constitute a formal-large or test result.

## Isolation record

```text
test_records_read=0
test_assets_opened=false
frozen_holdout_records_opened=0
formal_large_run=false
training_performed=false
target_rematerialization=false
```
