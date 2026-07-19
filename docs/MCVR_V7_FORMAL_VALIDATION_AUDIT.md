# MCVR V7 Formal-Large Validation Audit

## Decision

`FORMAL_VALIDATION_RUNNER_WAS_MISSING_AND_IMPLEMENTED`

The pre-existing `scripts/run_v7_formal_large_seed.sh` ends after prior
train/resume, strict checkpoint binding, provenance, and checksums. It does not
load or evaluate the 5,000-molecule, 10,000-record formal validation split.

The missing execution path is implemented by:

- `scripts/run_ecir_mvr_v7_formal_validation.py`;
- `scripts/report_ecir_mvr_v7_formal_validation.py`;
- `tests/test_ecir_mvr_v7_formal_validation.py`.

No V7 model, Jacobian, fusion, safety, loss, hidden-size, or layer logic was
changed.

## Frozen contract

- Methods: exactly `D1`, `V5-B`, and `V7`.
- V7 config SHA256:
  `5737ce5aa3bad729a6748a3fb9f0eea515bd96765df15e99bba6bd70297b8b4b`.
- Validation source SHA256:
  `e7d29f971124f51bd385ec987372ab85181b152250ec0789407a867ff81e3c1a`.
- Validation target SHA256:
  `4b4ef42c9905c3bbe2dbe911c57827ce594583c66a52f94d7c4d9b5ca70de4c7`.
- Formal source identity:
  `3d86eec9ebd82ae96860330ded0fad35938be74111929ed29b9487f8b7e39a0a`.
- Formal target identity:
  `4d2d45950c92894066e347a966c6d5b877afcb5fe0abe6cdb7c06e70a3148e62`.
- Cohort: 5,000 molecules, 10,000 records, exactly two records per molecule.
- Split: validation only.
- Training and target rematerialization: disabled.

The runner rejects checkpoint/config/data SHA changes, wrong seeds, test or
holdout paths, `test_record=true`, duplicate or missing samples, source/target
pairing changes, incomplete method sets, binding checksum changes, and output
directory conflicts.

## Execution chain

Each chunk loads one paired source/target cohort and evaluates all three frozen
methods through the existing `etflow.ecir.bac_evaluation.evaluate_bac_candidate`
path. D1 supplies the learned Cartesian prior; V5-B supplies the frozen global
Jacobian comparator; V7 applies its fixed Angle Jacobian, Clash operator,
constraint-specific fusion, and existing safety/backtracking.

The runner verifies identical source metrics across methods, writes atomic
chunk outputs with SHA256 and row counts, and skips a completed chunk on resume
only after integrity and sample-identity checks. Final reporting uses paired
per-molecule bootstrap comparisons for V7-D1 and V7-V5-B and aggregates V7
solver and component statistics.

## Isolation

The validation runner does not import or open the formal-test manifest, formal
test cache, or frozen holdout. All launch, progress, chunk, metadata, and report
records carry:

```text
test_records_read=0
test_assets_opened=false
frozen_holdout_records_opened=0
formal_test_run=false
training_performed=false
```

Generated validation outputs, logs, checkpoints, parquet files, and formal-test
results are not source artifacts and must not be committed.
