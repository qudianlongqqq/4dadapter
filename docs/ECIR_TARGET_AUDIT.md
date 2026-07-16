# ECIR restrained-target audit

The audit covers all 600 train+validation targets (300 ETFlow, 300 Cartesian); test was not read.

## Status

| status | records | mean aligned displacement (Å) | mean torsion change (rad) | mean validity gain |
|---|---:|---:|---:|---:|
| converged | 4 | 0.02660 | 0.00000 | 0.05403 |
| accepted, not converged | 595 | 0.19279 | 0.08816 | 0.19891 |
| fallback to soft reference | 1 | 4.28921 | 1.81560 | 0.56214 |

There are no rejected cached targets after fallback, no MMFF-unsupported final targets, and no UFF final targets. `optimization_iterations=50` is the persisted maximum iteration budget, not an observed iteration count; RDKit status code 1 only says the limit was reached.

## Source comparison

| source | converged | aligned displacement (Å) | max atom displacement (Å) | torsion change (rad) | validity gain | energy drop |
|---|---:|---:|---:|---:|---:|---:|
| Cartesian 100k | 0/300 | 0.26922 | 0.46011 | 0.10317 | 0.34516 | 1347.76 |
| ETFlow formal | 4/300 | 0.12780 | 0.24805 | 0.07773 | 0.05195 | 10.61 |

## Required answers

1. **Are non-converged targets displaced farther? Yes**, `0.19279 Å` versus `0.02660 Å`, but only four converged records exist.
2. **Do they change torsions more? Yes in this cache**, `0.08816 rad` versus zero; the converged subset contains no effective rotatable-torsion change.
3. **Is validity gain proportional to displacement? Strongly associated**, accepted-target Spearman rho is `0.867`; this does not establish causality. More importantly, the current “gain” is the target-relative error removed by moving to the target, not a thresholded chemical-validity gain, so part of this association is mathematical coupling.
4. **Are there small-gain/large-movement targets?** None under the frozen definition (bottom-quartile gain and top-decile displacement), but the one soft-reference fallback is an extreme `4.29 Å` outlier and must not be treated as minimal repair.
5. **Are Cartesian targets heavy reconstruction? Yes.** They move about 2.1x farther than ETFlow targets and have 2.6x larger maximum atom displacement; none converged.
6. **Are current targets suitable minimal-repair labels? No.** The convergence profile, Cartesian displacement and soft-reference outlier conflict with the minimal-change objective. Stage C needs a threshold-stopping validity target anchored to input geometry.

Detailed rows and grouped summaries are in `diagnostics/ecir_mvr/target_audit/`. The requested `bond/angle/ring/clash_validity_gain` column names are retained for compatibility, but must be read with the metric-definition caveat above.
