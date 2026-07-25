# BAT scientific reasonableness audit

## Judgment

The staged hypothesis is mathematically reasonable as a **Reference-calibrated local internal-geometry prior**, not as an energy model. Bond/Angle centers already have prospective support. A periodic mixture is appropriate for non-ring rotatable torsions, and a one-sided steric barrier is more appropriate for clashes than a Gaussian target. Source preservation must remain an explicit trust constraint.

## Main risks and binding corrections

1. **BA active-set mismatch.** Frozen LSGO-B used all BA primitives, not a calibrated active-set. BAT keeps BA bit-for-bit and applies activity only to the new torsion and steric groups.
2. **Torsion identifiability.** A graph context can support multiple modes but cannot encode the current 3D mode. The model is allowed to describe the Reference ensemble, not select a unique conformer. Torsion correction is activated only in a calibrated low-likelihood tail.
3. **Canonical orientation.** Central-bond and terminal selection must be deterministic under the cache↔RDKit mapping. Symmetric substituents can make a particular terminal physically arbitrary; deterministic canonical ranks and map-ID tie breaks make labels reproducible, while the audit must disclose symmetry-derived apparent multimodality.
4. **Concentration cheating.** Learned kappa can uniformize exactly as learned sigma inflated. The primary model freezes kappa; learned kappa is diagnostic-only and cannot enter formal coordinates.
5. **Mixture non-identifiability.** Component permutations are harmless, but duplicated modes or unused components are not evidence of multimodality. Occupancy, circular separation and entropy gates are mandatory.
6. **Graph-only transfer.** Memorizing molecule identity can improve train likelihood. Molecule-disjoint DEV_A/DEV_B and new external identities are required.
7. **Reference is not energy.** `Reference likelihood != GFN2-xTB energy`. External energy transfer, if present, is independent evidence only.
8. **Steric/BAT conflict.** A repulsive direction can oppose BA/T directions. Groups are activated independently, normalized by their own Cartesian gradient RMS, combined without tuned 10/20/40 weights, then backtracked through hard guards.
9. **Softplus leakage in safe regions.** Softplus is nonzero everywhere. The entire steric group is exact-zero unless at least one chemically defined pair penetrates its safe distance.
10. **Tiny budget.** 0.003 Å may reduce penetration without crossing a discrete PB threshold. Internal penetration improvement is not equivalent to external PB improvement; both will be reported.
11. **Torsion Jacobian degeneracy.** Direct autograd gradient is primary. No learned-precision or closed-form torsion projection is reintroduced.
12. **Local scope.** Bond/Angle/non-ring rotor/steric terms do not constitute a full 3N-6 conformational distribution or physical potential.

## Pre-external falsification logic

Steric must improve penetration consistently on both dev cohorts without BA, Reference, topology, chirality or trust regression. Only then is the torsion pilot run. Fixed-kappa K=3 must beat uniform, not lose to single VM, avoid component pathology, separate Source from Reference and preserve modes on both dev cohorts. External PB/xTB are inaccessible until all eligible conditions and their coordinate hashes are frozen.

