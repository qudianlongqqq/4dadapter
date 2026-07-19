# MCVR V7 10K Development Validation Preflight

## Decision

**PREFLIGHT PASSED WITH A RECORDED DIRTY WORKTREE.** The frozen V7 method,
evaluator, checkpoint, thresholds, and seed are internally consistent. The
existing validation split cannot supply 10,000 development molecules without
opening the frozen holdout: it contains 5,000 molecules total, partitioned
into 4,000 validation-tune and 1,000 frozen-holdout molecules.

The 10K cohort is therefore frozen as a deterministic, train-derived unseen
development subset. It selects 10,000 molecules from the 50,000-molecule
formal train source pool after excluding every molecule in the D1 training
manifest. This is a 30,000-record development evaluation, not training, not a
formal-large full scan, and not test or frozen-holdout evaluation.

## Repository state

- Branch: `feat/mcvr-v7-constraint-specific-hybrid`
- HEAD: `056c6cd9ac09d6752369eac8f43da82a5cbe290c`
- Worktree clean: false
- Policy: no reset, stash, clean, deletion, or overwrite

Dirty paths recorded before 10K manifest construction:

```text
 M reports/ecir_mvr/SHA256SUMS.txt
 M reports/ecir_mvr/progressive_state.json
 M scripts/build_ecir_mvr_stage_e0_calibration_data.py
 M scripts/train_ecir_mvr_medium_rescue_v2.py
 M scripts/train_ecir_mvr_run_a.py
 M tests/test_ecir_mvr_formal_large_targets.py
 M tests/test_ecir_mvr_medium_rescue_v2.py
?? diagnostics/ecir_mvr/formal_large/
?? diagnostics/ecir_mvr/formal_test/
?? diagnostics/ecir_mvr/stage_d/d2/variant_cache/
?? diagnostics/ecir_mvr/stage_d/d2_interrupted_worktree.patch
?? diagnostics/ecir_mvr/stage_e0/calibration_dataset.parquet
?? diagnostics/ecir_mvr/stage_e0/smoke/
?? diagnostics/ecir_mvr/stage_e1/
?? diagnostics/ecir_mvr/stage_f/
?? diagnostics/ecir_mvr/stage_g/
?? diagnostics/ecir_mvr/stage_h0/
?? diagnostics/ecir_mvr/v2_bac_overnight/
?? diagnostics/ecir_mvr/v2_bac_recovery/manifests/development_sources.parquet
?? diagnostics/ecir_mvr/v2_bac_recovery/manifests/development_targets.parquet
?? diagnostics/ecir_mvr/v2_bac_recovery/manifests/diagnostic_sources.parquet
?? diagnostics/ecir_mvr/v2_bac_recovery/manifests/diagnostic_targets.parquet
?? diagnostics/ecir_mvr/v2_bac_recovery/runs/
?? diagnostics/ecir_mvr/v5_constraint_hybrid/runs/
?? diagnostics/ecir_mvr/v6_adaptive_jacobian/
?? diagnostics/ecir_mvr/v7_constraint_specific/
?? docs/MCVR_STAGE_E1_FAILURE_ATTRIBUTION.md
?? docs/MCVR_STAGE_E1_NEXT_METHOD_DECISION.md
?? docs/MCVR_STAGE_E1_UNSEEN_ANALYSIS.md
?? docs/MCVR_STAGE_F_FEATURE_CONFIDENCE.md
?? docs/MCVR_V6_ARCHITECTURE_AUDIT.md
?? docs/MCVR_V6_EXPERIMENT_REPORT.md
?? docs/MCVR_V7_ARCHITECTURE_AUDIT.md
?? docs/MCVR_V7_EXPERIMENT_REPORT.md
?? etflow/ecir/mvr_v6_adaptive_jacobian.py
?? etflow/ecir/mvr_v6_adaptive_loss.py
?? etflow/ecir/mvr_v7_constraint_specific.py
?? etflow/ecir/stage_e1_attribution.py
?? reports/ecir_mvr/D1B_FORMAL_WINDOWS_SEED43.yaml
?? reports/ecir_mvr/D1B_FORMAL_WINDOWS_SEED43_DATA_VALIDATION.json
?? reports/ecir_mvr/D1B_FORMAL_WINDOWS_SEED43_SMOKE.json
?? reports/global4d_profile_bundle_verification.json
?? scripts/audit_ecir_mvr_stage_e1.py
?? scripts/report_ecir_mvr_v6_adaptive_jacobian.py
?? scripts/report_ecir_mvr_v7_constraint_specific.py
?? scripts/run_ecir_mvr_v6_adaptive_jacobian.py
?? scripts/run_ecir_mvr_v7_constraint_specific.py
?? scripts/smoke_ecir_mvr_formal_windows.py
?? tests/test_ecir_mvr_stage_e1.py
?? tests/test_ecir_mvr_v6_adaptive_jacobian.py
?? tests/test_ecir_mvr_v7_constraint_specific.py
```

## Frozen implementation identities

| Asset | SHA256 |
|---|---|
| `etflow/ecir/mvr_v7_constraint_specific.py` | `74bde53ee2ab1ac22137f90c66bfa3d25a5c8fe97141731b99c4f37656fc711f` |
| `scripts/run_ecir_mvr_v7_constraint_specific.py` | `3d515b337c67e1089c9b58ecd2dadc698ed1fb1ef31e6663dffa90eb0f3d6887` |
| `scripts/report_ecir_mvr_v7_constraint_specific.py` | `401c688883e4c603386e066b4b478e2c3df68490812256014bd29412a118af51` |
| `tests/test_ecir_mvr_v7_constraint_specific.py` | `99577cf67efb7ddf0fd26cb134912f149bbc0588ccfdec8f130c34570e80cbbd` |
| V7 resolved config | `cea71c6d9e5c12565707a127c7b7390f4caed7d22611fadaebb1fab321cfa645` |
| `etflow/ecir/bac_evaluation.py` | `19c58781dbde09df7c29a9d4856436b192540be58381e545a13366a67dff2f63` |
| `etflow/ecir/run_a_evaluation.py` | `b890459edc7244047a0d2c7547681523315f1ccc95778f753625aba05670576d` |
| `etflow/ecir/bac_jacobian.py` | `a405ebbf0ab99128abe93fc2ecb1d5ec432409f5beedbd78c0f528bcd0603a00` |
| `etflow/ecir/bac_constraints.py` | `cf084afd4c23d00faac66b0194d2bb4ec125e7cfaa0d51c44100da480caa83cf` |
| `etflow/ecir/bac_safety.py` | `51a5972f9e3d2032baaee31e574b1683ece0170460bec5e521104aecfc6b3c37` |
| `etflow/ecir/chemical_validity.py` | `03a6c64ba1e275198d955fc83a683a704d7c63c8c659e5deb7ca9c5eac9f63d6` |
| `data/ecir_mvr/validity_reference_stats.json` | `ae5afaa8d3fce1b5418295309bf2c3197997180298e1781b4efc5c265258852e` |
| D1 model source | `b0441c9852b38a7bd603a480fab7d5c56851f08224d3eed4e440fabdbd3c1439` |
| V5-B model source | `46b9c0810f8e490039ea3c86ea5d09439a0287753dba28b89863405723b954d3` |
| V5-B resolved config | `d1e70583f77d98e95194fe7ee06eac797da4cc268ef875d54a267295eef92a41` |
| D1 checkpoint | `9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426` |

The 10K orchestration must verify these hashes before opening a cohort PT.
It may add manifest and orchestration/reporting scripts, but it must not alter
the frozen method or evaluator files.

## Data source audit

Formal source identity:
`3d86eec9ebd82ae96860330ded0fad35938be74111929ed29b9487f8b7e39a0a`.

| Asset | Records | Molecules | SHA256 |
|---|---:|---:|---|
| Formal train source manifest | 150,000 | 50,000 | `fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d` |
| Existing train target manifest | 150,000 | 50,000 | `7e97c5d92529608cfcace8cd279cbd25f20e08b28e1739a191483ba3b574c242` |
| D1 training source manifest | 4,096 | 3,978 | `767eb0db025d85df7421c7418dd4460463c5e332cf21673870833bed34a85c14` |
| D1 training target manifest | 4,096 | 3,978 | `9145bdfb21cca209c42d92e46d6b773c711ca6b7269749134085506ffedb8ee` |

Selection algorithm:

1. Read molecule IDs only from the existing formal train source manifest.
2. Remove all 3,978 molecule IDs present in the D1 training source manifest.
3. Rank the remaining 46,022 IDs by SHA256 of
   `43018|formal_source_identity|d1_checkpoint_sha256|molecule_id`.
4. Select the first 10,000 IDs and include all three source records per ID.
5. Pair by `sample_id` with existing target-manifest rows; do not build or
   modify any target.

Frozen prospective cohort facts:

- Molecules: 10,000
- Records: 30,000
- Records per molecule: exactly 3
- D1 training molecule overlap: 0
- Validation-tune molecule overlap: 0
- Frozen-holdout molecule overlap: 0
- Missing source PT files: 0
- Missing target PT files: 0
- Missing target rows: 0
- Source `test_record=true`: 0
- Target `test_records_read` maximum: 0
- Ordered molecule IDs SHA256:
  `17f19269598d7985b16bd0beb82f8e00f0401b2a44ba91c42b631bdc8489bf78`
- Ordered sample IDs SHA256:
  `880c68ced3e8f3e74b9aa44a207ea1abbc0715776e542298d06514840695c0a3`
- Ordered source coordinate hashes SHA256:
  `40561efd8201bc663b24dda58769cf5f09cf1b28fffa4ffd772c92a5684ba340`
- Ordered target hashes SHA256:
  `f16b9483df647e20860e324fc55db6f64e89cc2548038d2f9242e4f36eef3e04`

Frozen manifest output:

- Canonical manifest identity SHA256:
  `764d7a19fe40d6795553b37291efebd0e62ad604c54ba4c89a9dcdd0bb5705fc`
- `manifest.json` file SHA256:
  `a01571daf0cf337f105a19e32632bdae11d6b1840b6a6f4d0d2e05baa001c435`
- Derived source manifest SHA256:
  `6b3fd9e5a353a79123d1338d10da6c853f5c2dc7d2afa682052475e6d3b2992a`
- Derived target manifest SHA256:
  `a0b739a38fb605b262391547251099c02cce2c5f369c8bf3f3f0c2bc592aafa1`

Reading validation cohort ID lists for overlap checks does not open a
frozen-holdout record. No test manifest, test parquet, or test PT is opened.

## Frozen experiment

- Seed: 43018
- Methods: D1, V5-B, V7
- Teacher steps: 4
- Step size: 0.25
- Batch size: 64
- Evaluator and validity thresholds: unchanged from the 512-molecule V7 run
- Bootstrap: 10,000 molecule-level paired resamples
- V7-D1 and V7-V5-B comparisons: same sample IDs
- Training: none
- New target materialization: none

Formal-large admission gates remain unchanged:

- Active-Angle V7-D1 CI upper bound below zero
- Bond degradation versus D1 below 0.005
- Mean displacement below `1.1x` D1
- Acceptance drop versus D1 below five percentage points
- Ring/chirality non-regressed
- RMSD and COV noninferior
- Solver failure rate zero or otherwise explicitly fail-closed and explained

## Isolation contract

Every manifest, launch record, method result, and final report must record:

```text
test_records_read=0
test_assets_opened=false
frozen_holdout_records_opened=0
formal_large_run=false
training_performed=false
target_rematerialization=false
```

Preflight status: `V7_10K_PREFLIGHT_PASSED`.
