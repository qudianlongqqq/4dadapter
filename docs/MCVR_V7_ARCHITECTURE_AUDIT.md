# MCVR V7 Constraint-Specific Hybrid Architecture Audit

## Frozen scope

V7 is isolated on branch `feat/mcvr-v7-constraint-specific-hybrid`. Existing
D1, V2-BAC, V5-B, and V6 source files, checkpoints, configs, evaluators, and
results are immutable. The only permitted data are the existing 512-molecule,
1024-record Windows development cohort with identity
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.
Test, frozen holdout, formal-large evaluation, target rematerialization,
training, hidden-size changes, layer changes, learned gates, and result-based
candidate changes are prohibited.

The frozen neural checkpoint is D1 SHA256
`9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`.
V7 has no trainable parameters and produces no new checkpoint.

## Existing correction spaces

D1 emits one learned Cartesian field. Its raw field is trust-clipped by atom
and graph limits and multiplied by the learned global safety gate. It is
strong on Bond but has limited mechanism-specific Angle evidence.

V5-B strict-loads D1 as a neural Cartesian prior, builds one combined
Bond/Angle/Clash analytic system at the prior proposal, and adds its bounded
Jacobian correction before common trust and safety. On the frozen development
cohort it improves active Angle over D1 by `-0.001220`, paired 95% CI
`[-0.001742, -0.000703]`, while preserving Bond and acceptance. Its movement
is `1.1032x` D1 and narrowly exceeds the fixed `1.1x` envelope.

V6 learns Bond, Angle, and Clash fusion gates around the same stable solver.
Its solver has zero failures, but the pilot gates collapse toward Bond:
Bond `0.5627`, Angle `0.0525`, Clash `0.00036`. The V5-B Angle advantage is
lost. This rules out further gate tuning for the current study; it does not
rule out constraint-specific operators.

## V7 insertion point

V7 is a new inference-only wrapper around the frozen D1 `MCVRBACModel`:

```text
current coordinates + graph
             |
       frozen D1 forward
             |
     D1 raw Cartesian field ---------------- Bond component
             |
canonical Angle ranges -> cosine Angle Jacobian -> DLS/SVD component
             |
nonbond spatial neighbors -> pair penetration repulsion -> Clash component
             |
       per-component fixed trust normalization
             |
       common coordinate trust projection
             |
       existing D1 global safety gate
             |
       existing evaluator BAC/Ring/chirality rollback
```

There is no learned fusion, free residual, loss reweighting, or new encoder.

## Bond Cartesian component

The Bond component is the frozen D1 `v_raw` field multiplied by the inference
step size. It is capped at the original D1 coordinate-step limits:
`step_size * max_velocity_graph_rms` and
`step_size * max_velocity_atom_norm`. After final fusion it receives the same
D1 global safety gate exactly once.

Using `v_raw`, rather than adding to already gated `v_final`, preserves D1
semantics. If Angle and Clash are inactive, component and final trust
normalization are idempotent and V7 reduces to D1 up to floating-point error.

## Angle Jacobian component

The Angle operator reads only current coordinates and canonical allowed Angle
ranges. It selects out-of-range triplets, maps the nearest interval boundary
to cosine space, and constructs only analytic cosine-Angle rows. Bond and
Clash rows are absent from this solve.

The existing float64 damped least-squares/truncated-SVD solver is reused. It
retains relative rank thresholding, condition monitoring, small-singular
handling, near-linear downweighting, zero-length fail-closed behavior, rigid
translation/rotation removal, predicted-reduction checks, finite checks, and
fixed graph/atom caps. No explicit inverse or `acos` derivative is allowed.

## Clash spatial component

The Clash operator uses deterministic nonbond spatial neighbors from the
existing topology-aware sparse radius search. For each active pair with
penetration `p`, equal and opposite coordinate updates of magnitude `p/2` are
applied along the pair direction, then averaged by incident active-pair count.
This is translation invariant and rotation equivariant. Exactly coincident
pairs are masked and fail closed because a fixed fallback axis would violate
rotation equivariance.

Clash does not enter the Jacobian. It has the same fixed component trust cap
as Angle and records active edges, degenerate edges, raw RMS, trust scale, and
scaled RMS. The frozen cohort contains only one Clash-active record, so this
operator can be checked for numerical behavior but cannot support a broad
scientific claim from the current experiment.

## Fixed constraint-aware fusion

For graph `g`, each coordinate component is independently normalized:

```text
bond_g  = alpha_bond_g  * raw_bond_g
angle_g = alpha_angle_g * raw_angle_g
clash_g = alpha_clash_g * raw_clash_g
```

Each `alpha` is the deterministic minimum of 1, its graph-RMS trust ratio,
and its atom-maximum trust ratio. It is not a parameter. The normalized
components are added once, then the combined field receives the original D1
graph/atom trust projection. This final projection is a common safety budget,
not a learned competition between constraint types.

The frozen component caps are D1's original step limits for Bond and the
already audited V5-B limits `0.01 A` graph RMS and `0.02 A` atom maximum for
both Angle and Clash. These values are fixed before the experiment and will
not be changed after observing results.

## Safety and evaluator compatibility

The wrapper returns the normal MCVR keys (`v_raw`, `v_trust_clipped`,
`v_final`, `velocity`, and `global_safety_gate`) plus component diagnostics.
It is therefore compatible with `evaluate_bac_candidate` without a Cartesian
or Global4D entry point.

The evaluator performs four steps at step size 0.25, checks every trajectory
candidate against Bond, Angle, Clash, Ring, chirality, identity, finite, atom
trust, and molecule trust rules, uses deterministic backtracking, and rolls
back to the source conformer if no candidate is safe. A nonfinite V7 fusion
returns zero velocity so the evaluator fails closed to source.

## Checkpoint compatibility

V7 strict-loads the unchanged D1 state into an unchanged `MCVRBACModel`.
The wrapper owns no learned state and does not serialize a V7 checkpoint.
There are no missing-key allowances, prefix translations, or partial loads.
The checkpoint SHA and development identity must match before any asset is
opened for evaluation.

## Frozen experiment and success criteria

Run one 128-record smoke, then one full 1024-record development evaluation if
the smoke passes. Both use seed 43018, identical sample IDs, four inference
steps, batch size 64, and the existing safety configuration. No training or
sweep is permitted.

The fixed comparison is D1 versus V5-B versus V7. Report Bond, Angle,
active-Angle, Clash, Ring, chirality, weighted BAC, aligned RMSD, MAT-P/MAT-R,
COV-P/COV-R, acceptance, rollback, displacement, runtime, Angle solver rank,
condition number, singular values, truncated directions, solver failures,
and Clash activity.

V7 supports the constraint-specific hypothesis only if its active-Angle gain
versus D1 has paired CI upper bound below zero, Bond degradation is at most
0.005, movement is at most `1.1x` D1, acceptance drops by at most five
percentage points, Ring/chirality do not regress, and RMSD/COV remain
noninferior. A failure ends this frozen candidate without tuning.
