# Scientific weakness analysis

| issue | classification | evidence and interpretation | paper consequence |
|---|---|---|---|
| Bulk corrections are only about 0.005–0.006 Å | PLAUSIBLE_REVIEWER_CONCERN | Seed307 movement is small yet V3D and xTB single-point energy improve. Small magnitude does not make the effect trivial, but the causal explanation must be explicit. | report paired transitions and effect sizes, not only movement mean |
| Reference RMSD is nearly unchanged | VERIFIED_WEAKNESS | Restricted/Unrestricted Reference RMSD is about 1.31964 Å and differs from Source only at roughly 1e-4 Å scale. | do not claim global conformer recovery; claim local validity/geometry refinement |
| Reliability independent role | PLAUSIBLE_REVIEWER_CONCERN | Six-arm R0/R1 comparisons support R1, but the exact final full-joint architecture lacks a clean no-Reliability ablation. | staged evidence is defensible with qualification; final ablation would strengthen it |
| Adaptive BA necessity | VERIFIED_WEAKNESS | Earlier matched Adaptive-BA-v2 was mixed/neutral; the final full-joint increment is positive but causally confounded by joint retraining. | avoid claiming independently proven necessity until isolated |
| sigma calibration | UNKNOWN | NLL/pathology diagnostics exist; calibration curves, coverage, or calibration error for current sigma were not found. | call sigma a learned predictive scale, not calibrated uncertainty |
| Bond/Angle primitives omit explicit torsion | PLAUSIBLE_REVIEWER_CONCERN | Current action uses Bond and Angle derivatives only. Global RMSD barely changes, consistent with local rather than torsional refinement. | scope method as local geometry refinement; do not imply full conformational search |
| Learned tau may act mostly as global scale | PLAUSIBLE_REVIEWER_CONCERN | tau is noncollapsed and variable, but direct evidence that its conditional variation—not merely mean scale—drives gains is limited. | show distribution/mechanism diagnostics; avoid strong personalization claim |
| Graph-RMS normalization limits amplitude expression | NOT_SUPPORTED | It intentionally separates direction from scalar tau. No failure attributable to the normalization is established. | explain design; deletion experiment is optional unless claimed as essential |
| Fixed graph topology | PLAUSIBLE_REVIEWER_CONCERN | Bond topology is frozen; the method cannot repair wrong connectivity/bond order. Current cohort/evaluators assume topology identity. | state non-applicability to topology errors |
| Reference objective vs validity/xTB mismatch | PLAUSIBLE_REVIEWER_CONCERN | Reference RMSD barely changes while local validity and xTB improve; post objective is geometry-based, not energy or V3D/PB loss. | frame downstream improvements as empirical transfer, not objective equivalence |
| ETFlow-specific source defects | VERIFIED_WEAKNESS | Current complete evidence is same-upstream ETFlow only. | general-refiner claims require cross-upstream evidence |
| Rare unrestricted energy catastrophe | VERIFIED_WEAKNESS | One unrestricted seed307 record has DeltaE > +100 kcal/mol and max about +570 kcal/mol; Restricted has none >+100. | report robust location and tails; do not hide mean sensitivity |
| Seed robustness | UNKNOWN | Only seed307 is complete; partial later seeds are excluded from this audit. | multiseed is a submission blocker |
| Size/flexibility/source-quality dependence | PLAUSIBLE_REVIEWER_CONCERN | Seed307 quintiles show gains but high-size/flexibility/poor-Source evidence is descriptive and single-seed. | limitations/subgroup reporting required; no defect established |
| Explicit first-order optimizer interpretation | NOT_SUPPORTED | The code builds an analytic primitive action but does not establish convergence or formal optimization guarantees. | avoid optimizer/Gauss–Newton overclaim |

## Core scientific position

The verified weakness is not an unstable architecture. It is a mismatch between the strongest empirical effect (local geometry/validity and single-point energy) and the tempting broader story (global conformer recovery, calibrated uncertainty, or universal refinement). Claim discipline resolves much of this gap without changing the model.

