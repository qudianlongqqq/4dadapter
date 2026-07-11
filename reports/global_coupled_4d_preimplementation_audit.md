# Global Coupled 4D pre-implementation audit

## Scope and evidence

The audit covered `etflow/commons`, `etflow/models`, `scripts`, `configs`, and `tests`, with particular attention to FlexBond, bond-local, kinematic, Jacobian, projection, residual, gate, local-time, checkpoint, rollout, RMSD, COV, and MAT paths.

## Findings

1. **Old four variables.** Legacy FlexBond-4D uses one stretch coefficient and three angular coefficients in a per-bond local frame. `build_atom_jacobian` constructs `[e0, -skew(lever) @ frame]`; the three predicted angular scalars are converted to a global angular vector with `frame @ q[:,1:]`.
2. **Local frame.** Yes. `build_bond_frames` uses the bond axis plus the affected atom with the largest perpendicular lever arm.
3. **Frame sign/continuity.** The frame is rotation-covariant away from degeneracies, but the discrete farthest-atom selection can switch at ties or near-ties. Therefore local perpendicular coefficient labels can change discontinuously even when the physical angular vector is continuous.
4. **Target solve.** It is per-bond. `solve_q_targets` accumulates a separate `[4,4]` normal matrix and four-vector RHS for each bond, then calls a batched independent solve.
5. **Complete Jacobian.** No complete `[3N,4M]` legacy matrix is constructed.
6. **Overlapping contributions.** They are summed with `index_add_` and then divided by an atom contribution count in `apply_jacobian_4d_correction`; thus overlapping joint effects are averaged.
7. **Cross terms.** No `J_b^T J_c` terms are computed for `b != c`; the old normal system is block diagonal by construction.
8. **Coefficient target.** It is not a global pseudoinverse target. Each bond independently fits the residual on its affected atoms with ridge regularization and filtering.
9. **Cartesian residual orthogonality.** The old hybrid adds `v_cart + correction_scale * v_4d`; the Cartesian branch is not projected onto the orthogonal complement of the entire internal subspace.
10. **Training/sampling geometry.** The legacy model uses the same bond selector, anchors, moving atoms, and affected-side mapping in forward inference. Training-only pseudo-labels use the same mapping. However, this mapping selects a capped smaller side per bond, not a rooted complete downstream articulated tree.
11. **Reference leakage.** No inference leakage was found. Reference coordinates are used only to create detached training pseudo-labels; the inference dataset structurally omits labels.
12. **Checkpoint policy.** Old training checkpoints are primarily selected by validation flow/final loss, with `last.ckpt` retained.
13. **Rollout selection.** Historical diagnostic/sweep scripts exist, but the old model's trainer does not make rollout RMSD the checkpoint-selection criterion.
14. **Why Gated 1D is slow.** It performs Python topology/fragment work during forward, calls `_topologies` again in `_shared_step`, pools fragments in Python loops, and projects with repeated matrix-free operations.
15. **Projection backend.** Gated 1D defaults to a fixed 24-iteration conjugate-gradient projection.
16. **Topology repetition.** Yes. The coordinate-independent topology is rebuilt across forward calls and again for target construction; it is also rebuilt at every rollout step.
17. **Reusable parts.** Deterministic fragment-tree orientation and fail-closed topology status from `molecular_kinematics.py`, the EGNN trunk, inference-only dataset/manifest validation, refinement clipping, fair evaluator, provenance, and crash-safe run-state helpers are reusable.

## Decision

The research direction is mathematically well-defined and can reuse validated infrastructure, but the legacy 4D implementation does not satisfy global coupling or strict residual orthogonality.

Decision: **GO_WITH_REQUIRED_FIXES**

Required fixes are the complete molecular Jacobian, complete Gram cross terms, global pseudoinverse targets, strict residual projection, stable global angular-vector supervision, cached topology, exact solver fallbacks, and rollout-based checkpoint selection. These are implemented in the new namespace; legacy paths remain untouched.

