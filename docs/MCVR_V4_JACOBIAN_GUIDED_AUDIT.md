# MCVR V4 Jacobian-Guided Cartesian Audit

## Frozen inputs

V4 starts from commit `76208ce` on branch
`feat/mcvr-v4-jacobian-guided-cartesian`. The Cartesian baseline is frozen D1
checkpoint SHA `9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`.
The only permitted evaluation data is development manifest identity
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.

Formal test and frozen holdout are prohibited. There is no training, target
rematerialization, model-width change, new free Cartesian branch, or parameter
selection after evaluation.

## D1 coordinate path

`MCVRBACModel.forward` adds the D1-B base raw field and fused Angle/Clash
corrections. The combined field is atom/graph trust-clipped and multiplied by
the global safety gate to produce `v_final`. `infer_bac` integrates four teacher
steps with step size 0.25, evaluates trajectory states, and applies finite
hard-safe backtracking. V4 defines `delta_cart` as the final accepted D1
coordinate minus source, not as a single raw velocity.

## J0 coordinate path

J0 builds active Bond distance, cosine-Angle, and Clash penetration residuals
and analytic Jacobian rows. It solves weighted damped least squares in float64,
uses a relative singular-value threshold and truncated-SVD fallback, removes
rigid translation/rotation, applies trust limits, and performs finite line
search/relinearization. J0 is an independent coordinate optimizer; its weak
Bond and acceptance results make it unsuitable as a D1 replacement.

## Guided candidates

Candidate A linearizes at the D1 proposal and solves the minimum residual
correction. It evaluates fixed alpha values 0.25, 0.5, and 1.0. The correction
itself and combined source-relative movement obey trust limits. A result must
reduce the D1 proposal's Jacobian objective and pass unchanged hard safety;
otherwise it returns D1 exactly.

Candidate B computes the complete Jacobian row-space projection of
`delta_cart` for diagnostics. Its applied update removes only the row-space
component associated with rows whose first-order `residual * J delta_cart` is
positive. It therefore suppresses predicted BAC-worsening directions without
creating an independent free residual. The transformed update must not worsen
the D1 Jacobian objective or hard safety; otherwise it returns D1.

Candidate C does not generate a Jacobian update. It uses the Jacobian objective
as a trust-region judge for the D1 direction. It tries scales 1, 0.5, 0.25, and
0.125, accepting the first objective-decreasing, hard-safe coordinate. If none
passes, it returns source.

All candidate output is a single Cartesian coordinate tensor. The Jacobian is
an offline helper, not a model branch.

## Compatibility

No guided module is registered on `MCVRBACModel`, so the D1 state dict and
strict checkpoint load are unchanged. Inputs are source coordinates plus the
existing canonical static topology and train-derived ranges. Dataset schema,
cache schema, model-independent data layer, and materialized target assets do
not change.

## Decision rule

A guided candidate is supported only if its active-Angle gain over D1 has a
paired 95% CI strictly below zero, Bond and weighted-BAC degradation versus D1
are each at most 0.005, acceptance drops by at most five percentage points,
movement is at most 1.1 times D1, and Ring/chirality remain non-regressed. The
matrix is frozen before implementation and no follow-up alpha or projection
variant is allowed.
