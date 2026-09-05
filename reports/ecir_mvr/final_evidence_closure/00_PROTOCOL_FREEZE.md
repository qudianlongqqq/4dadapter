# SIXS final evidence closure protocol freeze

This closure reuses all frozen Source, Restricted, Unrestricted, AvgFlow, DiTMC, and matched-ablation outputs. The only rerun scientific baseline is current-final MMFF94s. Reference work is evaluation/aggregation of already-frozen coordinates and references. No model, checkpoint, cohort, evaluator semantics, metric, seed, or movement policy may change.

```text
COHORT = 2500 molecules x 2 records = 5000
PRIMARY_PROTOCOL_SHA256 = d6c57c192d17b4b1d10662f08af1221110ee20fb09a151049c0dbe7474b78238
PRIMARY_MANIFEST_SHA256 = 2a1d07af8c9e3150d1f2f3719d0bd43bd33819ca7674c364d0770c010cb86ee1
MMFF_METHOD = MMFF94s
MMFF_NUM_ATOMS_REPAIR = authoritative frozen RDKit topology molecule atom count
MMFF_FAILURE_POLICY = explicit failure; Source fallback only for fixed-denominator structural evaluation; never counted as MMFF success
PAIRED_CLUSTER = molecule
BOOTSTRAP_RESAMPLES = 10000
REFERENCE_RMSD = all-atom fixed-order float64 proper-rotation Kabsch; nearest frozen ensemble; no symmetry permutation
BOND_ANGLE_REFERENCE = first frozen Reference conformer
REFERENCE_CONTEXT = first frozen Reference conformer, contextual-only, duplicated solely for matched record identity
COV_AMR_APPLICABLE = NO (no frozen primary threshold/protocol and no independent Reference self split)
REFERENCE_SELF_COV_AMR = NOT_REPORTED
GFN2_XTB_GEOMETRY_OPT = DO_NOT_RUN
NO_MODEL_TRAINING = YES
NO_SIXS_REINFERENCE = YES
ONE_HEAVY_STAGE_AT_A_TIME = YES
```
