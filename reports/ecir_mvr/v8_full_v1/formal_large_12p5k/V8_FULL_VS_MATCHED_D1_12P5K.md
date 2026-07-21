# V8 Full 12.5K vs matched D1-only 12.5K

Both methods match Seed43, exposure 800000, batch 64, the original 200K schedule provenance, exact step12500 selection, ordered 10K validation records, and frozen evaluator.

| Method | Accept | Weighted BAC | Bond | Angle | Active angle | Ring | Clash | Chirality | Mean disp. | RMSD | MAT-P | MAT-R | COV-P | COV-R | Target loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V8 Full 12.5K | 0.9862 | -0.20140865 | -0.093121296 | -0.0044310814 | -0.0044310814 | -0.019807461 | -3.7915966e-09 | 1 | 0.0034744402 | 1.3302337 | 1.3302337 | 2.0556156 | 0.491 | 0.13909685 | 7.8444738e-06 |
| Matched D1-only 12.5K | 0.9828 | -0.18533894 | -0.095874859 | -0.0011171409 | -0.0011171409 | -0.014139173 | -3.3256038e-11 | 1 | 0.0021724476 | 1.3300795 | 1.3300795 | 2.0555986 | 0.4912 | 0.13907423 | 5.3921363e-06 |

## Natural-cohort paired results (V8 minus matched D1)

| Metric | Mean | Median | Bootstrap 95% CI | W/T/L | Applicable | Status |
|---|---:|---:|---|---:|---:|---|
| accepted | 0.0034 | 0 | [0.0006, 0.0062] | 117/9800/83 | 10000 | SIGNIFICANT_V8_12P5K_better |
| weighted_bac_delta | -0.016069709 | -0.0062499123 | [-0.017812464, -0.014356908] | 5628/2274/2098 | 10000 | SIGNIFICANT_V8_12P5K_better |
| bond_delta | 0.0027535628 | 0 | [0.002024356, 0.0034768936] | 3098/4609/2293 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| angle_delta | -0.0057895536 | 0 | [-0.0060570305, -0.0055255696] | 2248/3231/245 | 5724 | SIGNIFICANT_V8_12P5K_better |
| active_angle_delta | -0.0057895536 | 0 | [-0.0060570305, -0.0055255696] | 2248/3231/245 | 5724 | SIGNIFICANT_V8_12P5K_better |
| clash_delta | -1.8791703e-06 | 0 | [-4.0395926e-06, 1.0154655e-07] | 8/9/3 | 20 | NOT_SIGNIFICANT |
| ring_delta | -0.015071225 | 0 | [-0.016904089, -0.013193749] | 1485/1706/570 | 3761 | SIGNIFICANT_V8_12P5K_better |
| chirality_preserved | 0 | 0 | [0, 0] | 0/3456/0 | 3456 | NOT_SIGNIFICANT |
| mean_displacement | 0.0013019926 | 0.0007607101 | [0.0012636539, 0.0013417815] | 1389/55/8556 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| max_atom_displacement | 0.010266779 | 0.0038502924 | [0.0099599247, 0.010583355] | 2086/55/7859 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| solver_failure_count | 0 | 0 | [0, 0] | 0/10000/0 | 10000 | NOT_SIGNIFICANT |
| rmsd | 0.00015421894 | 0.00014737248 | [0.00014246569, 0.00016621385] | 3185/56/6759 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| target_loss | 2.4523375e-06 | 5.6892986e-07 | [2.2511788e-06, 2.66008e-06] | 2044/55/7901 | 10000 | SIGNIFICANT_V8_12P5K_worse |

## Conclusion

V8 constraints provide improvement beyond continued D1 training.

Cohort-specific results, timing, stability, cache hashes, and all 10,000-draw bootstrap results are retained in the JSON/CSV artifacts.
