# MCVR V8 Full v1 method

MCVR V8 is an end-to-end trainable neural-analytic refiner with deterministic deployment safety.
The frozen Seed43 D1 architecture is nested unchanged and supplies one equivariant Cartesian
prior per refinement step. New invariant error-state heads predict atom correction magnitude,
graph RMS/max correction, and bounded per-atom prior confidence.

For each graph and step, Bond and Angle residuals are rebuilt at the current coordinates. Bond
uses distance-to-valid-boundary; Angle uses the V7 cosine residual. A soft activity weight is
applied identically to each residual and Jacobian. Each type is independently divided by its
train-only frozen scale and by the square root of its soft active count.

The layer solves

`(W + lambda_move I + lambda_b Jb^T Jb + lambda_a Ja^T Ja) delta =`
`W delta_prior - lambda_b Jb^T rb - lambda_a Ja^T ra`.

There is no explicit inverse. The default backend uses `cholesky_ex` and `cholesky_solve` on a
per-graph float64 block with positive damping; `torch.linalg.solve` is available as an ablation.
The inactive, zero-movement case bypasses numerical damping and exactly returns `delta_prior`.
Nonfinite/factorization failure is reported as `SOLVER_FAILURE_FAIL_CLOSED` and returns the D1
prior. Condition numbers are detached diagnostics only and never select a training rank.

Full mode shares all D1/V8 parameters across two steps. Step 1 rebuilds constraints, Jacobians,
soft activity, confidence, and the matrix at `x1`. A zero-initialized invariant step embedding is
added to D1's existing deterministic feature input. The cumulative source-relative motion is
projected differentiably to the frozen `0.12 Å` atom and `0.06 Å` graph RMS trust region. This is
not rollback and does not block gradients.

Smooth Clash, Ring source-noninferiority, and signed-volume Chirality losses remain outside the
linear Bond/Angle solver. Every geometry type has its own active denominator and loss weight.
The sampler stores one record per train sample with overlapping cohort memberships, so rare-error
boosts do not materialize duplicate records. Cohort weights are derived from train prevalence and
capped; validation retains its natural distribution.

The optimizer has three groups. The D1 training baseline used `2e-4`; V8 therefore starts new
heads at `2e-4`, the initialized D1 correction head at `5e-5`, and the D1 backbones at `2e-5`.
These are conservative train-derived initialization choices, not formal-test tuning.

Parity mode is `constraint_layer.enabled=false`, `error_state.enabled=false`, and
`unroll_steps=1`. It bypasses V8 confidence, trust projection, scaling, clipping, and centering;
`delta_final` is bitwise equal to frozen D1 `v_final`.
