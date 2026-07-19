# MCVR V3-BAC Jacobian Design

## Scope and comparator

Phase 2 starts from protected Phase-1 commit `6a86dc3` on branch
`feat/mcvr-v3-bac-jacobian`. It implements an independent, non-learning,
offline BAC solver. It has no neural network, checkpoint dependency, free
Cartesian residual branch, or trainable parameter. It uses the same canonical
constraint detector, development cohort, metric thresholds, and hard safety as
the repaired Cartesian D1 comparator.

D1 is fixed at checkpoint SHA
`9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`.
The development manifest identity is
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.
No frozen holdout or formal test is opened.

## Residuals and analytic rows

For an active bond `(i,j)`,

`r_b = ||x_i-x_j|| - d_boundary`.

The row contains unit direction `u_ij` at atom i and `-u_ij` at atom j.
Distances below `1e-8` use a deterministic finite fallback direction and are
counted as degenerate.

For an active angle `(i,j,k)`, the solver uses

`r_a = cos(theta) - cos(theta_boundary)`.

It differentiates normalized dot product directly; no `arccos` derivative is
used. Bond vectors shorter than `1e-8` are masked. Angles with sine below
`1e-3` are downweighted by 0.1 and counted as near-linear. The ordinary angle
residual exists only in correctness tests; J0 uses cosine residual.

For an active clash `(i,j)`,

`r_c = d_safe - ||x_i-x_j||`.

The row is `-u_ij` at i and `+u_ij` at j, so solving `J delta = -r`
increases separation. Active pairs are rebuilt after every accepted step.
Topology exclusions and deterministic overlap fallback are inherited from the
canonical sparse detector.

## Weighted damped solve

J0 solves

`min_delta ||W^(1/2)(J delta + r)||^2 + lambda ||M^(1/2) delta||^2`.

Weights are fixed and equal by type, normalized by active count. Mobility is a
fixed diagonal coordinate penalty; no network predicts it. The primary solve
uses an augmented `torch.linalg.lstsq`. Singular values of the mobility-scaled
weighted Jacobian determine effective rank and condition number. If the solve
is nonfinite, rank-deficient beyond the configured threshold, or exceeds the
condition limit, a damped truncated-SVD path uses factors
`sigma/(sigma^2+lambda)` only for `sigma/sigma_max >= rank_tol`.

No ordinary matrix inverse, `torch.inverse`, or `numpy.linalg.inv` is allowed.
Effective rank zero, nonfinite factors, nonpositive predicted reduction, or a
solver exception returns a zero update and explicit status.

## Rigid modes and trust

The raw update has centroid translation removed. An infinitesimal global
rotation is fit by least squares around centered coordinates and removed.
Diagnostics record norms before and after both projections.

The projected update is scaled to obey graph RMS 0.06 Angstrom and atom max
0.12 Angstrom. Each relinearization tries only 1, 0.5, 0.25, and 0.125.
A step must reduce the fixed weighted BAC objective and pass the existing hard
Bond/Angle/Clash/Ring, chirality, identity, finite, and trust checks. J0 uses at
most two relinearizations. Failure to find a safe scale stops and preserves the
last safe state; if no step was accepted, output is the exact source.

## Frozen J0 configuration

- bond/angle/clash weights: 1/1/1
- damping lambda: `1e-3`
- rank tolerance: `1e-6`
- maximum condition number: `1e8`
- near-linear sine threshold/weight: `1e-3` / `0.1`
- maximum relinearizations: 2
- backtracking scales: 1, 0.5, 0.25, 0.125
- graph RMS / atom max: 0.06 / 0.12 Angstrom
- clash cutoff/contact/topology exclusion/cap: 2.0 / 1.0 / 2 / 128

There is no J1 tuning. This configuration is frozen before implementation and
evaluation.

## Required tests

Analytic Bond, cosine-Angle, and Clash rows must match autograd/finite
difference on random and boundary coordinates. Solver tests cover rotation and
translation equivariance, centroid/rotation zero modes, very short bonds,
near-zero and near-pi angles, coincident clashes, duplicate constraints, empty
constraints, rank deficiency, small singular values, condition fallback,
single/two-atom inputs, concentrated clashes, trust limits, objective rollback,
and nonfinite/factorization failure. All paths must be NaN/Inf-free and fail
closed.

## Decision rule

J0 is supported only if active-Angle or active-Clash improves beyond D1 without
unacceptable Bond, Ring, chirality, displacement, failure, conditioning, or
runtime cost. Gains caused only by larger movement are not support. Because
Clash has one active development record, a Clash conclusion is expected to be
inconclusive. A negative or inconclusive result does not authorize J1 tuning,
learned Jacobian, 10k, or formal-large training.
