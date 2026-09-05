# Current-final AvgFlow Reference fidelity audit

```text
AVG_REFERENCE_MAPPING_REUSED = YES
AVG_REFERENCE_MAPPING_STATUS = PASS_REUSED_EXACTLY
AVG_REFERENCE_EVAL_N = 7908
AVG_REFERENCE_ENSEMBLE_N = 7782
AVG_COV_AMR_MOLECULES = 3431
AVG_COV_AMR_THRESHOLD_ANGSTROM = 0.75
AVG_STRUCTURAL_VALIDITY_EFFECT = V3D_IMPROVES__PB_ESSENTIALLY_STABLE
AVG_REFERENCE_FIDELITY_EFFECT = MIXED
AVG_ENERGY_EFFECT = SMALL_POSITIVE_DELTA_E_SHIFT
```

Reference RMSD uses all 7,908 legal mappings. COV-P/R and AMR-P/R use the pre-outcome frozen complete-ensemble subset (7,782 records in 3,431 molecule clusters). Both are heavy-atom, symmetry-aware AvgFlow protocols and are not numerically interchangeable with the primary prospective cohort's fixed-order all-atom Kabsch metric.

Bond/Angle errors are not reported because 5,255 legal symmetry-ambiguous records do not have a unique atom permutation. Structural validity, Reference fidelity, and energy are interpreted separately; closeness of aggregate V3D/PB/xTB values is not treated as Reference fidelity.
