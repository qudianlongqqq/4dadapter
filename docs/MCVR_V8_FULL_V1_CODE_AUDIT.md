# MCVR V8 Full v1 code audit

## Audit boundary and frozen baseline

- Audit branch: `feat/mcvr-v8-full-v1`.
- Starting HEAD: `9178ec49876c7bf30ec71baa930f5bc3123fde80`, tagged
  `mcvr-v7-seed43-formal-v1`.
- The worktree was clean before V8 work began. Frozen D1/V5/V7 source was read but not edited.
- No formal-test manifest, prediction, record, checkpoint-selection result, or holdout asset was
  opened. Diagnostics directories were not recursively audited.
- The frozen Seed43 D1 checkpoint is
  `artifacts/ecir_mvr/formal_large/d1_b_seed43/best_noninferior_validity.ckpt`, SHA256
  `c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca`, schema
  `ecir-mvr-medium-rescue-formal-large-d1b-checkpoint-v1`, step 25000. A real strict load of all
  176 state entries passes with no missing or unexpected keys.

## Actual D1 data flow

`MCVRMixedDataset` reads an offline source manifest and an offline Minimal Validity Target
manifest, joins them by `sample_id`, preserves atom order, and loads `x_input` and `x_target`.
The old mixed sampler can synthesize corruptions, but V8 binds it in real-error-only mode; V8's
primary source is therefore the actual upstream conformer.

`MCVRLoss` samples one time per graph, constructs
`x_t = (1-t) x_input + t x_target`, and supervises `v_final` against the constant flow velocity
`x_target - x_input` with SmoothL1. Time is expanded through `atom_batch`. Target generation uses
proper-rotation Kabsch alignment: `kabsch_align` explicitly rejects reflections.

`MCVRModel` contains an error-encoder EGNN and the D1 Cartesian correction EGNN. The correction
backbone returns invariant node states and equivariant vector messages. Rigid/torsion Cartesian
heads, a learned safety gate, and the explicit Bond branch form `v_raw`; D1 then applies atom and
graph trust clipping and the safety gate to produce `v_final`. V8 reuses this complete forward
unchanged and consumes its `node_embedding`; it does not rewrite or rescale the checkpoint output.

## Actual V5-B semantics

`MCVRNeuralJacobianHybrid` is an inference-only wrapper around a frozen D1 prior. It first forms
`prior_coordinate = pos + 0.25 * v_final`, then explicitly calls `detach().to(float64)` before
building Bond/Angle constraints. Its analytic correction is solved separately, rigid components
are projected out, the correction receives independent graph/atom trust caps, and the resulting
velocity is added to D1 `v_raw` before D1-style clipping/safety. The method is mathematically
useful as a baseline but cannot return solver gradients to D1.

## Actual V7 semantics

`MCVRConstraintSpecificHybrid` freezes D1 parameters and always evaluates D1 under `no_grad`.
For each graph it treats `0.25 * D1 v_raw` as the Bond/Cartesian component, solves a separate
cosine-space Angle system on detached float64 coordinates, and constructs a sparse deterministic
Clash repulsion. Bond, Angle, and Clash receive separate trust caps, are added sequentially, and
are capped again as a fused update. D1's graph safety gate is applied last. V7 therefore has one
returned Cartesian velocity, but its components are independently constructed and it is not the
unified differentiable objective required by V8.

V7's Angle residual is `cos(theta_current) - cos(theta_boundary)`. Its analytic Jacobian avoids
`arccos`, handles short arms with an epsilon, downweights near-linear rows, and uses inference-time
SVD/rank diagnostics plus damped least squares/truncated SVD fallback. These definitions are
reusable; V7's detach, `no_grad`, discrete rank path, per-component solve, backtracking, and
rollback are deployment/evaluator mechanisms and cannot enter the V8 training graph.

## Constraint and evaluator semantics

- Bond/Angle ranges come from `ChemicalValidity._prepare`. Each row is
  `[lower, upper, robust_scale]`; Bond units are Angstrom and Angle units are radians.
- Bond constraints use unique undirected edges. The signed interval residual is negative below
  the lower boundary and positive above the upper boundary; valid rows have zero residual.
- Angle triplets are deterministic `i-j-k` rows with `j` central and mirrored neighbor pairs
  removed. V8 uses the validated cosine residual/Jacobian.
- Sparse Clash candidates exclude self pairs, duplicate direction, bonds, and topology neighbors
  through the configured graph distance. Candidate construction is discrete; selected distances
  and the smooth barrier remain differentiable.
- Ring bonds and RDKit-mapped chirality quads are already materialized by canonical constraint
  fields. Chirality quads are ordered `(center, first, second, third)`.
- `ChemicalValidity`, V7 acceptance, hard Ring/Chirality checks, backtracking, and rollback remain
  frozen validation/deployment semantics. They are not substituted by V8's smooth training losses.

## Reusable and non-reusable code

Reusable without changing old defaults:

- `MCVRModel` and its checkpoint/config contract;
- canonical PyG topology/constraint fields from `mvr_dataset` and `bac_constraints`;
- analytic Bond and cosine-Angle residual/Jacobian definitions;
- sparse Clash topology exclusion;
- proper-rotation Kabsch utilities;
- frozen ChemicalValidity and V7 deployment safety/evaluator.

Training-incompatible paths:

- V5-B/V7 `detach`, `no_grad`, inference-only decorators, Python-float trust decisions, hard SVD
  rank masks, component-wise additive correction, and rollback;
- CPU cell-list candidate identity is not differentiable with respect to pair membership, though
  the selected pair distances are differentiable;
- evaluator functions returning Python/NumPy metrics are reporting-only.

## Risks found and mitigations

- **Batch/index risk:** PyG increments fields ending in `_index`; V8 slices every graph using
  `ptr` and verifies that both/all atoms lie in the graph. Per-graph and batched tests match.
- **Checkpoint risk:** the old checkpoint is `MCVRModel`, not a V8 state. V8 nests it as `prior`
  and strict-loads the submodule before creating new heads.
- **Solver risk:** V7's SVD/lstsq path is stable but discrete and detached. V8 uses a positive
  definite normal equation and differentiable Cholesky/solve in float64, never an inverse.
- **Small-vector/near-linear risk:** epsilon-safe analytic rows, a continuous sine weight, positive
  damping, condition diagnostics, and fail-closed fallback are implemented.
- **Dense-memory risk:** V8 never constructs a batch-global matrix. It constructs one dense
  `3N x 3N` block per graph. Formal-large maximum atom-count profiling remains required before a
  large run; matrix-free CG is the planned fallback if that profile exceeds memory policy.
- **Bond dominance risk:** raw concatenation would let numerous Bond rows dominate. V8 separately
  scales and divides each type by the square root of its soft active count, aggregates losses by
  type, and uses a train-only rare-error sampler.
- **Movement risk:** an early real-train gate exposed a single large but finite linearized update.
  V8 now applies a differentiable cumulative trust projection (not detach/rollback) at the frozen
  D1/V7 atom and graph limits; parity mode bypasses it exactly.
- **Data risk:** the machine has the frozen checkpoint and medium train/validation development
  assets, but the configured formal-large train/validation manifests and caches are absent.
  Formal-large pilot commands therefore fail closed instead of falling back silently.

## Environment

The default `E:\python` interpreter is CPU-only and lacks PyArrow. The validated runtime is
`E:\miniconda\envs\etflow-5080-v2\python.exe`: Python 3.11.15, PyTorch 2.11.0+cu128,
CUDA 12.8, RTX 5080 16 GB, PyG 2.8.0, RDKit 2026.03.4, pandas 3.0.3, and PyArrow 25.0.0.

## Isolation counters

`formal_test_records_read = 0`; `formal_test_assets_opened = false`;
`minimal_validity_target_test_used = false`; `frozen_holdout_records_read = 0`;
`parameter_selection_from_formal_test = false`.
