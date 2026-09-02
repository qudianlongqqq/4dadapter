# STEP 2D provenance and fail-closed decision

The official archive is bound by byte size and SHA256. Canonical identity uses
RDKit parsing, atom-map removal, ordinary explicit-H removal, sanitization, and
canonical isomeric SMILES with stereochemistry preserved. Canonical duplicates
collapse to one identity; failures are ineligible and explicitly reported.

The base historical union contains `55806` canonical identities and
covers current TRAIN, full historical VAL/DEV, Formal100, the reused factorial/
ablation/multiseed DEV cohort, and the inspected DiTMC large holdout. All source
identities whose native-test membership is not exactly recoverable are added to
the conservative exclusion union. No absence is interpreted as nonuse.

```text
SELECTION_RULE = ASCENDING_SHA256_DOMAIN_NUL_CANONICAL_IDENTITY__TIE_BREAK_CANONICAL_IDENTITY__TAKE_FIRST_2500
SELECTOR_SHA256 = 41df4a438e4acca2a545555ea9e5f0b1adf14d330cb9a60a413859d9a88e8907
HISTORICAL_REQUIRED_REASON_CLASSES_MISSING = NONE
HISTORICAL_EXCLUSION_UNION_COMPLETE = NO__LEGACY_IDENTITIES_OR_PROVENANCE_INCOMPLETE
ELIGIBLE_UNUSED_POOL_N_MOLECULES = 0
PRIMARY_FINAL_MEMBERSHIP_FROZEN = NO
PROTECTED_OUTCOME_READ = NO
```
