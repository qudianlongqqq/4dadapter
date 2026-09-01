# Data split and integrity

## Verified identities

- TRAIN: 50,000 unique molecules and 150,000 ETFlow records.
- VAL: 5,000 molecules and 10,000 ETFlow records.
- DEV: deterministic 2,500-molecule subset of VAL, 5,000 records.
- Current hashes match both configs: prepared payload `b19bd8...`, source payload `c6b2ea...`, TRAIN manifest `fbfeff...`, VAL manifest `e7d29f...`, DEV manifest `15db86...`.
- Restricted and Unrestricted seed307 coordinate freezes have the same 5,000-record identity hash `2690c66e...`.

## Reused completed integrity evidence

The current hashes were rechecked against disk. The expensive overlap analysis was not repeated; its verified prior outputs were reused:

- TRAIN–DEV molecule overlap: 0.
- TRAIN–DEV source coordinate hash overlap: 0.
- TRAIN–DEV reference coordinate hash overlap: 0.
- Reference visible at inference: NO.
- Eight non-stereochemical graph overlaps correspond to different stereochemical identities, not identical molecules.

Evidence: `sixs_final_project_integrity_generalization_audit/05_DATA_LINEAGE.md`, `07_CONFORMER_REFERENCE_LEAKAGE.md`, and `FINAL_STATUS.json`; reconfirmation in `sixs_final_candidate_selection_audit/06_DATA_INTEGRITY_RECONFIRMATION.md`.

## Interpretation

```text
TRAIN_DEV_INTEGRITY = PASS
DATA_LEAKAGE_FOUND = NO_EVIDENCE
REFERENCE_LEAKAGE = NO
MANIFEST_FROZEN_AND_HASH_MATCHED = YES
SEED307_RESTRICTED_UNRESTRICTED_RECORD_MATCH = PASS
MULTISEED_MATCHED_MANIFEST_INTENT = VERIFIED_IN_CODE
MULTISEED_MATCHED_EXECUTION = INCOMPLETE
DEV_ROLE = DEVELOPMENT_SET
FINAL_UNBIASED_TEST_SET = NO
```

“No evidence” is used for the global leakage statement because this audit reused the prior full scan rather than independently rescanning every coordinate.


