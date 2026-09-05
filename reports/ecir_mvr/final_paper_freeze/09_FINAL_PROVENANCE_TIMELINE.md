# Final provenance timeline

1. Restricted and Unrestricted were prospectively frozen as operating points A and B before primary-final outcome access (`reports/ecir_mvr/final_development_freeze/11_FINAL_DEVELOPMENT_FREEZE.md`).
2. The 2,500-molecule prospective cohort membership was frozen with zero documented historical overlap (`reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json`).
3. The prospective ETFlow NFE10 source and final evaluation protocol were frozen before scientific outcome access (`reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json`).
4. The primary-final evaluation then opened the prospective outcome (`reports/ecir_mvr/sixs_primary_final_evaluation/FINAL_STATUS.json`).
5. Later evidence synthesis labeled Unrestricted the quality-oriented primary and Restricted the constrained control; a sole primary was not predeclared (`reports/ecir_mvr/final_evidence_closure/16_PROVENANCE_TIMELINE.md`).
6. The matched final ablation completed on the frozen design (`reports/ecir_mvr/sixs_final_matched_ablation/10_FINAL_ABLATION_CONCLUSION.md`).
7. AvgFlow and DiTMC cross-upstream runs completed without retraining (`reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/*/RESULT.json`).
8. MMFF94s was repaired by restoring `num_atoms`; 4,977/5,000 optimizations succeeded and 23 fixed-denominator rows used documented Source fallback (`reports/ecir_mvr/final_evidence_closure/03_MMFF_FINAL_RESULTS.csv`).
9. Reference contextual xTB was recovered with scientific-identity deduplication; Reference fidelity and AvgFlow Reference audits completed (`reports/ecir_mvr/final_evidence_closure/09_REFERENCE_CONTEXTUAL_METRICS.csv`, `AVG_REFERENCE_METRICS.csv`).
10. Final aggregation completed after a zero-mismatch `molecule_id` suffix repair (`reports/ecir_mvr/final_evidence_closure/18_FINAL_MAIN_TABLE.csv`).
11. This release freezes those completed artifacts without scientific recomputation.

```text
PROSPECTIVELY_FROZEN_OPERATING_POINTS = Restricted; Unrestricted
QUALITY_ORIENTED_PRIMARY = Unrestricted
CONSTRAINED_CONTROL = Restricted
SOLE_PRIMARY_PREDECLARED = NO
SCIENTIFIC_SEMANTICS_CHANGED = NO
```
