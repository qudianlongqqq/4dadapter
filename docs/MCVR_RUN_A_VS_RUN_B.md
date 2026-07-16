# MCVR Run A versus Run B

## Incremental conclusion

Run B does not justify the active torsion branch. Relative to Run A, its small accuracy improvements coexist with a statistically clear validity regression, while the torsion-prior metric is exactly unchanged. The registered decision is **`RUN_B_HARMS`**, and Run A remains the medium candidate.

All deltas below are Run B minus Run A. For validity/error metrics, positive is worse; for RMSD/MAT, negative is better.

## All validation molecules

| Metric | Run A | Run B | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| Total thresholded validity | 0.565064 | 0.579737 | +0.014673 | [+0.007808, +0.022349] |
| Bond outlier rate | 0.197211 | 0.198769 | +0.001558 | [-0.001417, +0.004902] |
| Angle outlier rate | 0.027658 | 0.028274 | +0.000617 | [+0.000236, +0.001065] |
| Ring bond outlier rate | 0.108465 | 0.109213 | +0.000748 | [0.000000, +0.002051] |
| Torsion-prior outlier score | 3.147576 | 3.147576 | 0.000000 | [0, 0] |
| Severe clash | 0 | 0 | 0 | [0, 0] |
| Chirality error | 0 | 0 | 0 | [0, 0] |
| Aligned RMSD | 1.250293 | 1.250203 | -0.000090 Å | [-0.000156, -0.000033] |
| MAT-P | 1.250293 | 1.250203 | -0.000090 Å | [-0.000157, -0.000033] |
| MAT-R | 1.852374 | 1.852205 | -0.000169 Å | [-0.000256, -0.000087] |
| COV-P | 0.540000 | 0.540000 | 0 | [0, 0] |
| COV-R | 0.237912 | 0.237912 | 0 | [0, 0] |
| Diversity | 0.007525 | 0.007925 | +0.000400 | [+0.000265, +0.000539] |

Total validity worsened by 2.60% relative to Run A. Run B had a slightly smaller accepted RMS displacement (0.001451 vs 0.001702), but the registered chemical benefit was absent. Acceptance was 0.545 versus 0.555 for Run A, without collapse. For raw Run B proposals, validity worsened fraction was 0.50 and RMSD worsened fraction was 0.97; acceptance reduced those to 0 and 0.52 respectively.

## High-flex molecules

| Metric | Run A | Run B | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| Total thresholded validity | 0.736330 | 0.768725 | +0.032395 | [+0.014957, +0.049284] |
| Torsion-prior outlier score | 3.179807 | 3.179807 | 0 | [0, 0] |
| Aligned RMSD | 1.686694 | 1.686462 | -0.000232 Å | [-0.000361, -0.000103] |
| MAT-P | 1.686694 | 1.686462 | -0.000232 Å | [-0.000361, -0.000103] |
| MAT-R | 2.231577 | 2.231210 | -0.000367 Å | [-0.000563, -0.000186] |
| Mean torsion change | 0.001484 | 0.001116 | -0.000131 | [-0.000209, -0.000063] |
| P95 torsion change | 0.006032 | 0.004359 | - | - |

The high-flex accuracy and trust-limit gates passed, but high-flex validity was significantly worse and the torsion-prior score did not improve.

## Registered slices

| Slice | Run A validity | Run B validity | Run A RMSD | Run B RMSD | Interpretation |
|---|---:|---:|---:|---:|---|
| ETFlow normal | 0.287699 | 0.284414 | 1.124700 | 1.124757 | slight validity gain; accuracy neutral |
| Cartesian mild | 1.460378 | 1.525720 | 1.484207 | 1.483644 | validity worse; RMSD slightly better |
| Cartesian medium | 2.109823 | 2.269922 | 2.376772 | 2.375606 | validity worse; RMSD slightly better |
| Unseen scale 0.35 | 1.212250 | 1.268825 | 1.543344 | 1.542912 | validity worse; accuracy gate passes |

Cartesian severe had 0 records and is unavailable. The dataset's evaluated main records were all in the ring group, so non-ring had 0 records and is unavailable. Rotatable-bond groups (`<=2`, `3-5`, `>=6`) are present in the flexibility summary; the `>=6` result is the high-flex comparison above. Clean validation-reference controls are a separate 20-record identity set and remained 20/20 exact for both selected methods.

## Decision-rule audit

Run B passed all Run A upstream conditions and every accuracy/COV/trust/identity/acceptance gate. It failed the required torsion-or-high-flex CI benefit, minimum-effect, safety-not-worse, and unseen-validity conditions. In particular:

- torsion-prior delta was exactly zero;
- high-flex total validity significantly worsened;
- total validity significantly worsened;
- angle validity significantly worsened and ring bond outliers worsened on two molecules;
- unseen validity significantly worsened even though unseen accuracy improved slightly.

These failures meet the pre-registered `RUN_B_HARMS` rule. Active torsion repair is not suitable as the main method; the frozen Run A rigid-only method is retained.
