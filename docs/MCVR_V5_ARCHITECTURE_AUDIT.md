# MCVR V5 Constraint-Space Hybrid Architecture Audit

## Scope and frozen inputs

This audit precedes V5 implementation. The branch starts from V4 commit
`2f08104c9e91caaa22f054df6996fcbf5eec72f3`. The frozen D1 comparator is the
1000-step development checkpoint with SHA256
`9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426`.
The only permitted evaluation cohort is the 512-molecule, 1024-record
development manifest with identity
`3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51`.

Formal-large training, formal test, and frozen holdout are out of scope. Model
width, encoder depth, refinement depth, Dataset schema, runtime caches,
materialized targets, and all existing checkpoints are frozen.

## Current pipeline

```text
canonical source record + current coordinates
                    |
                    v
MCVRMixedDataset / graph_data
  static graph, x_input, training-only x_target,
  active mask, deterministic error features,
  canonical Bond/Angle ranges and topology
                    |
                    v
ECIRErrorEncoder + LightEGNNRefinerBackbone
                    |
       node embeddings + equivariant base field
                    |
       +------------+-----------------+
       |                              |
       v                              v
D1 Cartesian rigid/local heads   explicit Bond projection
       |                              |
       +----------- v_raw ------------+
                    |
       V2 Angle and Clash branches
                    |
       learned per-atom fusion gates
                    |
       atom/graph trust clipping
                    |
       global safety/uncertainty gate
                    |
                 v_final
                    |
       four fixed inference steps
                    |
       trajectory selection + hard safety
                    |
          accepted refined conformer
```

## Where delta_x is produced

`MCVRModel.forward` encodes graph and coordinates, then `_equivariant_head`
constructs `rigid_velocity` and `torsion_velocity` from scalar coefficients
times equivariant coordinate directions. The D1 field is the sum of gated
Cartesian contributions and the optional explicit Bond projection. It is
trust-clipped and safety-gated into `v_final`.

`MCVRBACModel.forward` reuses that D1 output, constructs sparse analytic
Angle and Clash directions, predicts strength/confidence/gates, and fuses the
two corrections with `base["v_raw"]`. It again clips and gates the single
Cartesian output. `infer_bac` integrates `v_final` for four steps at step size
0.25 and selects a hard-safe trajectory state. Thus the operational
`delta_x` is the accepted final coordinate minus source, not one raw network
velocity.

## Backbone, feature encoder, heads, and loss

- Backbone: `LightEGNNRefinerBackbone` produces invariant node embeddings and
  an equivariant base velocity without changing hidden width in V5.
- Feature encoder: `ECIRErrorEncoder` plus deterministic error embedding and
  graph pooling form a graph context. Canonical Bond/Angle ranges are static
  train-derived features; Clash contacts are rebuilt from current coordinates.
- D1 heads: rigid/local Cartesian scalar heads, fixed-zero torsion in the D1
  experiment, explicit Bond residual head/projection, safety gate, uncertainty,
  and error auxiliary head.
- V2 heads: Angle and Clash constraint encoders/heads plus two-way fusion gate.
- Base loss: `MCVRLoss` supervises flow/coordinate repair, validity, identity,
  anchors, error prediction, trust, and explicit Bond terms.
- BAC loss: `MCVRBACLoss` evaluates the complete first-step inference field in
  repaired D1 mode and adds per-record normalized Bond, Angle, Clash,
  preservation, no-new-violation, confidence, and gate objectives.

## Dataset and target flow

`MCVRMixedDataset` loads the existing Minimal Validity Target as `x_target`
for real errors. Synthetic errors use the clean reference and clean controls
use identity coordinates. `x_target` is consumed by training losses only. The
model forward path receives `x_init`/current coordinates, graph features,
active masks, deterministic source-derived diagnostics, static topology, and
canonical constraint ranges; it does not read `x_target`.

V5 therefore needs no new target fields or rematerialization. Both prototypes
can train against the existing unified target while differing only in how the
correction field is represented and fused.

## Evaluation metric boundary

Paper-facing conformer metrics are aligned RMSD, MAT-P, MAT-R, COV-P, COV-R,
and diversity. They measure agreement and set coverage against references and
must be reported with the same records and grouping as D1.

Mechanism and safety diagnostics are Bond/Angle/Clash/Ring validity deltas,
weighted BAC, chirality and stereocenter preservation, acceptance, rollback,
coordinate displacement, torsion change, gate activity, active-constraint
subsets, solver rank, condition number, singular values, and solver failure.
These explain behavior but must not silently replace the public metrics.

## Minimal-intrusion extension points

Prototype A should be a new `MCVRConstraintMultiHeadModel` subclass. It can
reuse the unchanged D1 encoder/backbone and explicit Bond result, add three
constraint-specific equivariant corrections, and use normalized learned gates
plus one global trust clip. It must expose each component for loss and
diagnostics while returning exactly one `v_final` tensor.

Prototype B should wrap a neural Cartesian prior with the existing analytic
Jacobian builder/solver outside the Dataset. The neural prior remains
label-free. The Jacobian operates on current coordinates and static canonical
ranges, uses damped/truncated float64 solves, removes rigid motion, and applies
bounded confidence/gating before the common trust and hard-safety path.

Both prototypes should have new model/loss/evaluation entry points. Existing
`MCVRModel`, `MCVRBACModel`, `MCVRBACLoss`, and evaluators remain unchanged.

## Checkpoint compatibility risks

Prototype A introduces new head and fusion parameters, so its own checkpoint
cannot strict-load as D1. Compatibility is one-way: initialize every shared
D1 parameter from the frozen checkpoint with an audited allow-list for only
new V5 keys, then strict round-trip the V5 checkpoint. Zero or conservative
initialization must preserve the initial D1 field before training.

Prototype B can preserve the neural prior state surface if the Jacobian module
has no learned parameters. If a scalar confidence/gate is learned, it must be
isolated in the V5 wrapper and explicitly allow-listed. Analytic solver state,
rank thresholds, and damping are configuration, never checkpoint-shaped data
or Dataset cache content.

Changing the existing classes, hidden dimensions, layer counts, canonical
batch schema, or target schema would invalidate comparison and is prohibited.

## Frozen prototype definitions

Prototype A uses the existing D1 field as a Cartesian prior and exposes Bond,
Angle, and Clash corrections as separate equivariant components. A softmax
gate allocates a bounded constraint budget among the three active components;
per-component RMS normalization prevents magnitude domination. The gated
constraint field is added to the D1 prior, followed by the unchanged atom and
graph trust clip and safety gate. It is not an unnormalized sum.

Prototype B uses the unchanged D1 neural field as `delta_prior`. At each
inference step it builds the analytic BAC system at the neural proposal,
solves one damped/truncated correction, removes rigid motion, trust-clips the
geometric update, and combines it through a fixed bounded lambda selected
before evaluation. The combined update must pass source-relative trust and
hard safety; otherwise it returns the D1 proposal. No direct inverse or
arccos derivative is permitted.

## Experiment design

Use seed 43018, hidden dimension 64, edge hidden dimension 64, four backbone
layers, three encoder layers, the existing training manifests and 1024-record
development cohort, 1000 optimizer steps, batch size 64, identical optimizer
schedule, four inference steps, step size 0.25, and identical hard safety.

Run one 200-step smoke for each trainable prototype before its 1000-step pilot.
The fixed comparison is D1 versus Prototype A versus Prototype B. Report Bond,
Angle, Clash, weighted BAC, Ring, aligned RMSD, MAT-P/MAT-R, COV-P/COV-R,
acceptance, rollback, displacement, and runtime. Prototype B additionally
reports singular-value extrema, effective rank, condition number, truncated
directions, solver failures, and fallback rate.

No result-dependent head addition, hidden-size increase, layer increase,
alpha/lambda sweep, threshold selection, or extra candidate is permitted. A
prototype supports a paper direction only if it improves active Angle or
constraint efficiency without materially degrading D1 Bond, acceptance,
public conformer metrics, Ring/chirality, or the fixed movement envelope.
