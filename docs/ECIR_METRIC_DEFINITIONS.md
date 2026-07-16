# ECIR metric definitions and units

This document freezes the meanings used by the current ECIR atlas and evaluator. Several legacy names contain “violation” or “invalidity”, but the implementation does **not** apply chemical validity thresholds; they are mean deviations from a comparison coordinate set.

| Metric | Exact current definition | Unit | Aggregation/reference |
|---|---|---|---|
| atlas `bond_error` | mean absolute bonded-length difference | Å | unique undirected bonds; input versus nearest aligned reference conformer |
| evaluation `bond_violation` | same mathematical operator as bond error | Å | candidate versus persisted restrained/soft target, then conformer → molecule mean |
| atlas `angle_error` | mean absolute bond-angle difference | radians | all unordered neighbor pairs around each center; nearest reference |
| evaluation `angle_violation` | same operator, different target | radians | candidate versus persisted target; no threshold |
| `torsion_circular_error` | mean `abs(atan2(sin Δ, cos Δ))` | radians in `[0, π]` | one deterministically selected outer-atom quadruple per rotatable bond; not degrees, normalized score, or abnormality rate |
| atlas `ring_geometry_score` / evaluation `ring_invalidity` | mean absolute ring-bond length difference | Å | ring edges only; reference differs between atlas/evaluation; no ring-validity threshold |
| `clash_score` | mean `max(1.0 Å - d_ij, 0)` over all nonbonded unordered atom pairs | Å penetration averaged over all eligible pairs | absolute coordinate penalty, not comparison-target error |
| `severe_clash` | any nonbonded atom pair below `0.6 Å` | boolean/rate after aggregation | hard threshold |
| `chirality_error` | fraction of detected stereocenters whose signed local tetrahedral volume disagrees with the target | fraction `[0,1]` | centers with nondegenerate volumes; target-dependent |
| `MMFF_energy_drop` | constrained force-field energy before minus energy after | kcal/mol as reported by RDKit | raw molecule total; MMFF94s and UFF must never be pooled |
| `relaxation_RMSD` | Kabsch-aligned per-atom RMSD between relaxed and input coordinates | Å | `sqrt(mean_i ||R x_i+t-y_i||²)` over all atoms, including H |
| `aligned_RMSD` | minimum Kabsch RMSD from generated conformer to available references | Å | molecule-set metrics then use pairwise generated/reference RMSD matrices |
| `COV-P/R` | fraction of generated/reference conformers with nearest distance below 1.25 Å | fraction | mean per molecule |
| `MAT-P/R` | mean nearest-reference/generated RMSD | Å | mean per molecule |
| `diversity` | mean pairwise Kabsch RMSD among generated conformers | Å | zero when only one conformer exists |

Consequences:

- Atlas bond/angle/ring errors and evaluation “violations” share operators but use different targets, so their values cannot be compared as if they were the same metric.
- The legacy “violation/invalidity” labels must not be interpreted as threshold-exceedance rates.
- Summing bond, angle, torsion, ring and clash values mixes Å and radians. Such a sum is allowed only as an explicitly labeled diagnostic, never as a chemically calibrated score or formal gate.
