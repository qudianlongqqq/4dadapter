# STEP 2D provenance and fail-closed decision

The official archive is bound by byte size and SHA256. Canonical identity uses
RDKit parsing, atom-map removal, ordinary explicit-H removal, sanitization, and
canonical isomeric SMILES with stereochemistry preserved. Canonical duplicates
collapse to one identity; failures are ineligible and explicitly reported.

The base historical union contains `55806` canonical identities and
covers current TRAIN, full historical VAL/DEV, Formal100, the reused factorial/
ablation/multiseed DEV cohort, and the inspected DiTMC large holdout. The frozen
domain is the entire eligible unused GEOM-DRUGS universe; native split is
provenance metadata and is not an eligibility gate. No missing historical
identity is interpreted as nonuse, so membership remains fail-closed while the
legacy union is incomplete.

```text
SELECTION_RULE = ASCENDING_SHA256_DOMAIN_NUL_CANONICAL_IDENTITY__TIE_BREAK_CANONICAL_IDENTITY__TAKE_FIRST_2500
SELECTOR_SHA256 = 41df4a438e4acca2a545555ea9e5f0b1adf14d330cb9a60a413859d9a88e8907
PRIMARY_SELECTION_DOMAIN = ENTIRE_ELIGIBLE_UNUSED_GEOM_DRUGS_UNIVERSE
EXACT_NATIVE_TEST_GATE_STATUS = IMPLEMENTATION_OVERCONSTRAINT_REMOVED
HISTORICAL_REQUIRED_REASON_CLASSES_MISSING = NONE
HISTORICAL_EXCLUSION_UNION_COMPLETE = NO__LEGACY_IDENTITIES_OR_PROVENANCE_INCOMPLETE
ELIGIBLE_UNUSED_POOL_N_MOLECULES = 233312
PRIMARY_FINAL_MEMBERSHIP_FROZEN = NO
PROTECTED_OUTCOME_READ = NO
```
