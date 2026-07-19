# MCVR V3-BAC Jacobian Report

## Decision

`JACOBIAN_NOT_SUPPORTED` as a replacement for the repaired Cartesian D1.

The fixed J0 solver demonstrates a real Angle mechanism: it improves the
Angle-active subset beyond D1 while moving coordinates less. It nevertheless
loses most of D1's Bond and weighted-validity gain and accepts only 26.56% of
records. This is useful mechanism evidence, but it fails the preregistered
overall replacement rule. No J1 tuning, learned Jacobian, 10k, or formal-large
run is recommended.

Formal test and frozen validation holdout remained unopened:
`test_records_read=0`, `test_assets_opened=false`, and
`frozen_holdout_records_opened=0`.

## Implementation

`etflow/ecir/bac_jacobian.py` is independent of MCVR models and checkpoints.
It builds local analytic rows for distance Bond residuals, cosine-Angle
residuals, and active clash penetration. It uses float64 weighted damped least
squares, active-count type normalization, mobility regularization, effective
rank and condition diagnostics, and a damped truncated-SVD fallback. It never
uses an explicit inverse.

The solver removes centroid translation and fitted infinitesimal rotation,
scales to 0.06 Angstrom graph RMS and 0.12 Angstrom atom trust limits, tries
only scales 1/0.5/0.25/0.125, and performs at most two relinearizations. Clash
edges are rebuilt after accepted steps. Every step must reduce the fixed
weighted residual objective and pass the unchanged hard safety evaluator.

Near-zero bond vectors and coincident clashes use deterministic finite fallback
directions. Degenerate angle arms are masked. Angles with sine below `1e-3`
are downweighted by 0.1. Rank-zero, nonfinite, factorization failure, and
nonpositive predicted reduction fail closed.

## Correctness and stability

Fifteen independent tests pass. Bond, cosine-Angle, and Clash analytic rows
match autograd. Tests cover arccos derivative amplification, SE(3)
equivariance, rigid zero-mode removal, near-linear angles, zero-length bonds,
coincident clashes, empty constraints, duplicate/rank-deficient rows,
truncated-SVD fallback, trust limits, objective decrease, hard-safety rollback,
and nonfinite fail-closed behavior.

On 1024 development records:

- solver failure: 0/1024
- accepted: 272/1024 (26.56%)
- final statuses: 753 backtracking rejected, 265 iteration accepted, three
  converged, three no active constraint
- condition number mean/p95/max: 9.45 / 10.47 / 3264.54
- near-linear angles: one
- degenerate Bond/Angle/Clash constraints: zero
- truncated singular directions reported: four total
- real-data truncated-SVD fallbacks: zero; all 1021 active solves used augmented
  least squares

The primary rejected-attempt reason is no BAC gain (2980 scale attempts), then
Bond regression (479), Angle regression (201), Ring regression (181), and atom
trust (43). Thus the acceptance gap is objective/safety alignment, not
factorization instability.

## Cartesian versus Jacobian

Both methods use the same 512 molecules/1024 records, source coordinates,
canonical detector, thresholds, and hard safety. D1 checkpoint SHA is
`9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`;
its rerun reproduces frozen Phase-1 metrics to `1e-12`.

| Metric | D1 Cartesian | J0 Jacobian | J0 - D1 |
|---|---:|---:|---:|
| Bond delta | -0.091570 | -0.015871 | +0.075698 |
| Angle delta | -0.002333 | -0.003394 | -0.001062 |
| Active-Angle delta | -0.004498 | -0.006042 | -0.001544 |
| Weighted BAC delta | -0.168421 | -0.045166 | +0.123254 |
| Acceptance | 97.27% | 26.56% | -70.70 pp |
| Ring delta | -0.008831 | -0.003607 | +0.005224 |
| RMSD delta | +0.000411 | +0.000010 | -0.000401 |
| Mean displacement | 0.006701 | 0.001073 | -0.005628 |

The paired J0-minus-D1 active-Angle difference is -0.001544 with 95% CI
`[-0.002536, -0.000585]`. Therefore J0's extra Angle benefit is real and is
not purchased with larger movement; J0 uses only 16.0% of D1's mean
displacement. However, on the same active-Angle subset J0 sacrifices 0.06748
Bond outlier-rate improvement and 0.10624 weighted BAC improvement relative to
D1, both with confidence intervals strictly above zero. Acceptance is also far
lower.

Clash remains inconclusive: only one development record is Clash-active. J0
has no aggregate Clash change; no Clash support or rejection is inferentially
valid.

## Runtime and memory

J0 solver time is 34.45 seconds total, 0.0336 seconds/graph, with per-graph p95
about 0.0464 seconds. The D1 reproduction including inference and metric
aggregation is 40.17 seconds, 0.0392 seconds/graph, with peak allocated GPU
memory about 185 MB. J0 process RSS peaks around 2.21 GB, but that process also
holds all development items and the previously loaded D1 comparator. These
measurement scopes are not identical, so runtime and memory are descriptive
and are not used to support either method. Sparse clash construction is not an
observed bottleneck on this low-Clash cohort.

## Numerical safeguards

Small singular values use a relative `1e-6` rank threshold and damping factor
`sigma/(sigma^2+1e-3)` in fallback. The true cohort never exceeds the frozen
`1e8` condition limit, so suppression is verified by boundary/rank-deficiency
tests rather than a real-data fallback event. Near-linear angles avoid arccos
and are downweighted. Coincident clashes use a deterministic direction and
rebuild their active graph after relinearization. No NaN, Inf, solver failure,
ring degradation relative source, or chirality change is accepted.

## Recommendation

Do not enter learned Jacobian. J0 proves analytic geometry can target Angle
efficiently, but fixed equal-weight Jacobian is not a better unified BAC
executor than D1. Learning weights or mobility now would add an unvalidated
degree of freedom on a development set with virtually no Clash support and
would violate the decision rule for proceeding.

Retain D1 as the best development Cartesian method, retain the Jacobian module
as a tested offline research tool, and do not start 10k/formal-large. A future
Jacobian study would require a newly preregistered development cohort with
adequate Clash support and explicit Bond/Angle trade-off criteria; it cannot
reuse the frozen holdout or formal test.
