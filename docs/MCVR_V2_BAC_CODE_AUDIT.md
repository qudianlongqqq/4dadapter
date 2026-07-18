# MCVR V2-BAC Code Audit

## Scope and isolation

This audit covers the current D1-B train/validation implementation and the
proposed unified Bond-Angle-Clash (BAC) method. Formal test assets were not
opened, enumerated, or used. The development contract is:

- `validation_only=true`
- `test_records_read=0`
- `test_assets_opened=false`
- train and validation are the only permitted data splits
- the validation cohort is split deterministically by `molecule_id` into an
  80% tuning cohort and a 20% holdout cohort
- the holdout is evaluated once for at most two frozen candidates; no method
  changes are permitted after holdout evaluation

The audited formal source identity is
`3d86eec9ebd82ae96860330ded0fad35938be74111929ed29b9487f8b7e39a0a`.
It contains 150,000 train records from 50,000 molecules and 10,000 validation
records from 5,000 molecules. Train and validation have zero molecule overlap.
The frozen validity-statistics identity is
`66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3`.

## Current D1-B data flow

1. The formal source manifests point to upstream-neutral generated-conformer
   records. `formal_rdkit_adapter.py` validates and adapts their atom order,
   explicit hydrogens, bond graph, atom maps, and disconnected components.
2. `build_ecir_mvr_formal_large_targets.py` uses
   `MinimalValidityTargetBuilder` offline. It never builds a target inside the
   training process.
3. The existing target builder performs one joint Cartesian optimization of
   anchor, bond, angle, clash, ring, chirality, and torsion-preservation
   penalties. It is not a sum of independently generated coordinate targets.
4. `MCVRMixedDataset` joins source and target rows by `sample_id`. One
   conformer record remains one sample. Real-error examples read the offline
   `x_target`; synthetic examples use a clean coordinate target; clean-control
   examples use an identity target.
5. The dataset emits a canonical PyG batch with atom features, directed bond
   graph, static topology indices, input/target coordinates, active-mode mask,
   affected-atom mask, and deterministic graph diagnostics. Runtime caches are
   limited to molecule/sample-static data and are versioned independently of
   model width, layer count, weights, and hidden representations.
6. `MCVRLoss` samples flow time, interpolates between `x_input` and `x_target`,
   and supervises one Cartesian velocity. It adds internal-mode, identity,
   anchor, sparsity, error, uncertainty, trust, and D1-B explicit-bond losses.
7. `MCVRModel` runs an error encoder and a shared `LightEGNNRefinerBackbone`.
   Rigid and torsion branches produce equivariant Cartesian fields. D1-B adds
   a symmetric bond head and a damped joint bond-Jacobian projection.
8. The raw Cartesian field is clipped by per-atom and per-graph trust radii,
   multiplied by a learned global safety gate, and returned as one `[N, 3]`
   `v_final`/`velocity` tensor.
9. Inference performs four teacher steps. `run_a_evaluation.infer_mvr` records
   the trajectory and calls `select_trajectory_candidate`.
10. Acceptance compares each proposal with the source using label-free
    train-derived validity statistics. It enforces positive validity gain,
    identity/chirality/ring/clash non-regression, trust limits, uncertainty,
    and finite coordinates. If no candidate qualifies, it returns the exact
    source coordinates.
11. Checkpoints are selected only when RMSD/MAT/COV noninferiority and safety
    requirements pass. Within the qualified set, the current D1-B trainer
    orders candidates by validation validity delta, displacement, identity,
    and registered high-flex/unseen diagnostics.

## Current model inputs and outputs

The current model consumes only source-side graph features, current
coordinates, flow time, deterministic source diagnostics, and optional
upstream metadata. Training targets and reference coordinates are consumed by
the loss/evaluator, not by the model forward path. The public output used for
coordinate integration is one equivariant Cartesian tensor, `v_final`.

D1-B's explicit bond head consumes symmetric endpoint hidden states, bond
attributes, current bond length, and time embedding. It predicts a bounded
signed residual, confidence, and uncertainty. All unique-bond residuals for a
molecule are projected jointly through a damped Jacobian solve.

## Current target and loss limitations

- The target is already joint, deterministic, trust-bounded, and fail-closed,
  which is reusable. However, its clash penalty calls dense `torch.cdist`,
  excludes only directly bonded atoms, and allocates an O(N^2) pair matrix.
- Ring planarity is currently an active target penalty. V2-BAC requires ring to
  be context and a safety constraint, not an independently optimized head.
- Current angle supervision is an internal-mode derivative loss. There is no
  explicit angle triplet encoder or angle-specific equivariant proposal.
- Current clash supervision has no dynamic nonbonded spatial graph or explicit
  clash encoder.
- Several current loss reductions pool all constraints globally. A large
  molecule or constraint-rich type can therefore carry more influence than a
  sparse record. V2 must normalize each type per record before averaging
  records.
- The current general validity clash helper is dense. V2 training, target
  construction, and inference require a sparse deterministic implementation;
  the legacy helper must remain unchanged for D1-B compatibility.

## Bond-only compatibility boundary

`V2_A_BOND_ONLY` must construct exactly the existing `MCVRModel` parameter
surface and call its forward and loss paths without an additional numerical
operation. The frozen seed43 D1-B checkpoint has SHA256
`c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca`,
step 25,000, and strict-loads into the audited 384,678-parameter model with no
missing or unexpected keys. Fixed-input forward, loss, correction, acceptance,
and strict-load regression tests are required before any pilot.

## Reusable code

- formal source adapter and model-independent canonical topology cache
- `MCVRMixedDataset` sample planning and offline target join
- train-derived `ChemicalValidity` environment distributions
- `LightEGNNRefinerBackbone` and D1-B bond head/projection
- trust clipping and finite-value guards
- trajectory construction, displacement/torsion diagnostics, and rollback
- RMSD/MAT/COV validation implementation and paired bootstrap
- formal identity checks and atomic checkpoint writers

## Required implementation changes

1. Add a model-independent canonical constraint schema with schema, feature,
   statistics, and source identities. It stores only static topology,
   train-derived allowed ranges, protected ring/chirality indices, and sample
   identity. Model-specific encodings remain in a separate feature builder.
2. Add deterministic sparse angle and clash construction. Angle identities
   canonicalize `(i,j,k)` and `(k,j,i)`. Clash edges use spatial cells/radius
   search, exclude registered topology distances, and sort before top-k.
3. Add a unified BAC target builder that optimizes one `delta_x`, uses
   per-record/per-type normalized residuals, and treats ring/chirality/identity
   as preservation and hard acceptance constraints.
4. Add angle and clash encoders plus gated fusion around the shared D1-B
   backbone. They generate invariant scalar weights on equivariant local
   directions. Fusion occurs once before a single Cartesian correction and
   trust clip; there are no sequential coordinate states.
5. Add BAC losses with raw and weighted per-type values, per-record
   normalization, zero-error no-op, no-new-violation penalties, confidence,
   gates, and per-module gradient diagnostics.
6. Extend acceptance with explicit no-new-bond, no-new-angle, no-new-clash,
   aromatic-planarity, identity, chirality, finite-value, and trust rejection
   reasons. Failed proposals backtrack and finally return the exact source.
7. Add a validation-only overnight runner with preregistered hypotheses,
   deterministic tune/holdout manifests, at most six sequential GPU runs, and
   hard wall-clock/run-count guards.

## Leakage and path audit

Production V2 code must reject split `test`, must not import or inspect any
formal-test manifest/result, and must persist all three isolation fields in
every run artifact. References may be used by validation metrics only, never
as model inputs or acceptance features. Target coordinates may be used only by
the offline target builder and training loss.

Formal manifests contain historical Linux absolute paths. Existing dataset
relocation supports explicit source and target cache roots without rewriting
manifests. V2 configs must use the verified Windows roots through config fields,
not hardcoded drive paths in production modules.

## Numerical risks

- Angle derivatives become ill-conditioned near 0 and pi. V2 uses clamped
  cosine values and norm floors; degenerate triplets become inactive.
- Coincident nonbonded atoms have undefined directions. V2 uses a deterministic
  finite fallback direction and records the degeneracy.
- Dense pair construction is forbidden. Cell-list candidates are bounded by a
  per-graph maximum and deterministically sorted on CPU-compatible integer
  keys before tensor assembly.
- Constraint counts may be zero. Every reduction is empty-safe and returns an
  exact scalar zero.
- Joint projections may be rank-deficient. Damped solves are finite-checked and
  fail closed to the exact source/zero correction.
- New heads are zero- or small-output initialized so initialization is a D1-B
  numerical no-op.

## Capacity and compute estimate

The baseline is 384,678 trainable parameters (`hidden_dim=64`, four backbone
layers). Two small constraint encoders, type embedding, gates, and fusion are
expected to add roughly 30k-80k parameters while retaining the same backbone.
Sparse constraint work is O(B + A + C), where B is unique bonds, A is canonical
angles, and C is the capped spatial-edge count. No model-width-dependent value
is stored in the dataset/runtime cache. Expansion beyond 64x4 is prohibited
unless train/validation evidence shows shared-backbone underfitting after
target, representation, gradient, and fusion diagnostics pass.

## Frozen implementation and experiment plan

1. Implement and test the canonical sparse constraints, unified target,
   compatibility wrapper, unified fusion, BAC loss, and safety policy.
2. Run unit tests, Ruff, py_compile, a two-batch real-data CPU/GPU smoke, and at
   most 200 optimizer steps. Stop on any compatibility, identity, finite-value,
   or target-solver failure.
3. Before each run, append the hypothesis and immutable identities to
   `diagnostics/ecir_mvr/v2_bac_overnight/decision_log.jsonl`.
4. Run four matched 2k pilots sequentially: bond-only, bond+angle, bond+clash,
   and bond+angle+clash. Keep data, exposures, batch, optimizer, scheduler,
   hidden width, layers, initialization, and selection rules fixed.
5. Use validation-tune only for selection and at most two evidence-driven
   follow-ups. Parameter changes are capped at 2x per decision. Capacity is the
   last permitted intervention.
6. Freeze at most two candidates, then evaluate each exactly once on the
   validation holdout. Do not tune afterward.
7. Report negative or conflicting results without further unregistered search.
