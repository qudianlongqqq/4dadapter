# LSGO historical implementation and overlap audit

Audit date: 2026-07-26 (Asia/Shanghai). This report was written before LSGO training or external evaluation. The independent worktree was created from common public base `1a9b1633fb88a0ef26cba46c9f9e1534d7f22b7e` on branch `research/learned-structured-geometry-objective`.

## Frozen MVT implementation

The formal-large coordinate teacher is implemented by `etflow/ecir/minimal_validity_target.py` and configured by `configs/ecir_mvr_formal_large_minimal_targets.yaml`.

For a real upstream Source coordinate tensor, the builder evaluates train-Reference-derived validity envelopes and minimizes the following fixed, handcrafted objective:

| term | formal-large coefficient | implementation |
|---|---:|---|
| Kabsch-aligned Cartesian anchor | 2.0 | mean squared displacement from Source |
| Bond threshold excess | 1.0 | mean squared, divided by frozen robust type scale |
| Angle threshold excess | 0.5 | mean squared, divided by frozen robust type scale |
| Ring | 1.0 | ring-bond excess plus ring planarity excess |
| Clash | 2.0 | squared distance-cutoff penetration score |
| Chirality | 2.0 | signed-volume barrier relative to Source |
| Torsion anchor | 0.05 | periodic squared change; multiplied by 2 for high-flex records |

The coordinate optimizer is Adam, `max_steps=40`, `learning_rate=0.001`, patience 5 and minimum gain `1e-5`. Every step is projected to graph RMS at most 0.15 A and atom displacement at most 0.35 A. It terminates on all active violations resolved, a new safety risk, trust-radius contact, five-step improvement plateau, numerical anomaly, or step 40. Candidate selection is also handcrafted: validity gain minus `0.25 * aligned_RMS_displacement`, `0.05 * max_torsion_change`, and `2.0 * new_risk`. If no safe improving trajectory point exists, Source is returned exactly.

The Bond and Angle targets are not single universal distances. `ChemicalValidity` selects typed, hierarchical train-Reference statistics and supplies lower/upper envelopes and robust scales. Ring and Clash terms are likewise programmatically defined. The resulting selected coordinate is the MVT label. No Reference coordinate or force-field fallback is used by this implementation.

There is no repository evidence that the frozen formal-large MVT used `Bond x10 / Angle x20 / Ring x40`. Those numbers must not be attributed to MVT. Separate quantities exist: MVT coefficients above; V8 auxiliary loss coefficients; and `chemical_validity.py` reporting-score weights (`bond rate=1`, bond magnitude=0.25, angle rate=1, angle magnitude=0.25, severe clash=2, clash penetration=1, ring bond=1, ring planarity=1, stereocenter degeneracy=2).

## Frozen V8 target construction

`scripts/train_ecir_mvr_v8.py` binds the formal-large target cache and the batch exposes `x_input` and `x_target`. `MCVRV8Loss.forward` computes its primary coordinate loss directly between `output['x_final']` and cached `x_target`; the error-state labels are atom displacement magnitude, graph RMS and graph maximum of `x_target - x_input`. The frozen configuration traces the cache to `build_ecir_mvr_formal_large_targets.py`, hence to `MinimalValidityTargetBuilder` above.

V8 is therefore substantively a D1-initialized LightEGNN/MCVR network learning Source-to-MVT coordinate correction, augmented by a learned bounded confidence/error-state head and a two-step differentiable Bond/Angle constraint layer. Its target is not a clean experimental Reference coordinate. This statement does not imply that the analytic constraint layer merely copies V7: V8 jointly trains the D1 prior, error state and shared differentiable constraint path, but its supervised coordinate endpoint remains MVT.

## DRCSR audit

The frozen DRCSR implementation is at commit `5b019afa635c35c962d54b3199c02a64436d53ec` in `etflow/ecir/direct_strain.py`.

- Bond contexts are discrete hierarchies over element pair, bond order, aromaticity, ring membership and formal-charge pair.
- Angle contexts are discrete hierarchies over center element/hybridization, adjacent bond orders, aromatic/ring state and ring size.
- Ring contexts use aromaticity, ring size, fused state and element composition.
- Every hierarchy backs off deterministically to coarser keys and finally `all`; the minimum fitted context count is 64.
- Locations are train-Reference medians. Scales are `max(1.4826*MAD, floor)`, with floors 0.01 A for Bond, 0.02 in angle-cosine space and 0.005 A for Ring.
- Both Gaussian and Student-t (`df=4`) local NLLs exist. Standardized group scores subtract a train median and divide by a train IQR (floor 0.10); group aggregation is either equal standardized mean or smooth-max with `tau=0.5`.
- Tail clipping at standardized residual 12, context definitions, scale floors, group normalization and smooth-max are handcrafted. The fitted medians, MAD scales and group distributions are learned statistically from TRAIN Reference.
- DRCSR contains no neural conditional mean or uncertainty predictor. Its later neural Stage N predicts coordinate updates from Graph+Source against the frozen strain objective, not `mu/sigma` from graph context.

DRCSR therefore removed the MVT coordinate teacher and replaced fixed validity envelopes with continuous train-Reference distributions, but retained handcrafted discrete buckets, backoff, robust estimator family/floors, group normalization and aggregation.

## Existing differentiable geometry and solver assets

Stable differentiable primitives already exist in `etflow/ecir/geometry.py` and `etflow/commons/geometry_diagnostics.py`: unique Bond extraction, angle triplets, Bond lengths and guarded Bond angles/cosines. `etflow/ecir/bac_jacobian.py` contains analytic Bond and cosine-Angle residual Jacobians, degeneracy/near-linear guards, rank tolerance, damping, condition auditing, backtracking and trust caps. `etflow/ecir/v8_solver.py`, `v8_constraint_layer.py`, `mvr_v7_constraint_specific.py`, and `mvr_v5_constraint_hybrid.py` provide additional batched normal-equation and trust-region precedents. `etflow/commons/molecular_kinematics.py`, `kinematic_projection.py`, and the global/flexbond Jacobian modules provide torsional and SVD fallbacks. Chirality quads and strict RDKit/cache atom mapping exist in `etflow/ecir/rdkit_utils.py` and `target_building.py`.

The existing Jacobian machinery is reusable, but LSGO must introduce learned per-primitive means and precisions and audit the actual `J^T W J` spectrum. Existing code does not already implement the complete requested chain `GNN -> local conditional mu/sigma -> structured differentiable likelihood -> precision-weighted analytic micro-projection`.

## Required overlap answers

1. **Which MVT quantities are handcrafted?** Primitive choice, typed validity envelopes, robust scaling, all coefficients in the first table, optimizer/learning rate/40-step limit, trust caps, termination, safety rules and candidate ranking are handcrafted. The numerical envelopes themselves are fitted from TRAIN Reference.
2. **Which DRCSR quantities are handcrafted buckets?** The exact Bond/Angle/Ring context tuples, six/five-level backoff trees, minimum count 64, distribution family, scale floors, tail clip, group IQR normalization and smooth-max.
3. **Does a neural mu/sigma predictor already exist?** No. Repository-wide and DRCSR-branch audit found learned coordinate/error/confidence/score heads, but no graph-conditioned local Bond/Angle mean-and-uncertainty model trained only on Reference geometry likelihood.
4. **Is a stable differentiable Bond/Angle Jacobian available?** Yes. Both autograd primitives and analytic guarded Jacobians exist; LSGO will test them again with finite differences and real-molecule conditioning.
5. **How is LSGO different from V8?** V8 learns an MVT Cartesian endpoint and uses fixed solver statistics plus auxiliary handcrafted losses. LSGO learns only `p(local Bond/Angle geometry | graph)` from TRAIN Reference and applies its gradient/projection directly to Source without any Cartesian label or MVT trajectory.
6. **How is LSGO different from DRCSR?** DRCSR fits discrete-context medians/scales and aggregates standardized handcrafted groups. LSGO B replaces the discrete conditional mean with a continuous neural graph context; LSGO C additionally learns heteroscedastic uncertainty, making primitive precision data-dependent; LSGO C-P uses that precision in an explicit Jacobian projection.

## Access ledger at audit completion

- MVT coordinate cache opened for LSGO training: no
- Source-to-Reference Cartesian delta used: no
- xTB/MMFF/PoseBusters used for training or selection: no
- formal test reads: 0
- frozen holdout reads: 0
- FULL10K used for tuning: false

