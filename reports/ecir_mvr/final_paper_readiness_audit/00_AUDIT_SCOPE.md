# Final paper evidence gap audit — scope

Audit date: 2026-09-05 (Asia/Shanghai)

This is a read-only scientific-evidence audit. It did not train or modify a model, run inference, execute MMFF/xTB/MD, alter an evaluator, or recompute a scientific endpoint. Small audit reports are the only new artifacts.

## Frozen method and evidence boundary

- Requested paper-primary identity: `J1_R1_FULL_JOINT_ADAPTIVE_BA_UNRESTRICTED_MOVEMENT`, seeds 307/331/353, step 17,500.
- Prospective primary cohort: 2,500 unique molecules, two ETFlow source conformers per molecule, 5,000 records. Membership SHA256: `2a1d07af8c9e3150d1f2f3719d0bd43bd33819ca7674c364d0770c010cb86ee1`.
- Primary-final protocol SHA256: `d6c57c192d17b4b1d10662f08af1221110ee20fb09a151049c0dbe7474b78238`.
- Primary-final source record manifest SHA256: `e3e69398b3fd888c3e1a4a9dcc79e48d602a22d6098169704c327773c15ab3dc`.
- Evidence inspected: prospective primary evaluation, Restricted/Unrestricted multiseed, AvgFlow and DiTMC zero-shot results, final matched ablations, development/release freezes, current-code audit, and existing runtime/baseline audits.
- Development, historical 10K, and prospective-final numbers are kept separate.

## Important provenance tension

The 2026-09-01 development freeze retained two predeclared operating points and explicitly said a unique winner was not justified. Later cross-upstream artifacts label Unrestricted the final primary method. The requested audit accepts Unrestricted as the intended paper-primary identity, but this single-method designation is not yet closed by an immutable, outcome-independent selection record. Submission closure must either retain both predeclared operating points or document a defensible selection rationale that does not use the already-opened prospective outcome.

## Authoritative inputs

- `reports/ecir_mvr/sixs_step2d_primary_final_2500/FINAL_STATUS.json`
- `reports/ecir_mvr/sixs_step3a_etflow_source_generation/07_SOURCE_ASSET_FREEZE.json`
- `reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json`
- `reports/ecir_mvr/sixs_primary_final_evaluation/03_FINAL_METHOD_SUMMARY.csv`
- `reports/ecir_mvr/sixs_primary_final_evaluation/04_FINAL_COMPONENT_SUMMARY.csv`
- `reports/ecir_mvr/sixs_primary_final_evaluation/05_MOLECULE_CLUSTER_BOOTSTRAP.csv`
- `reports/ecir_mvr/sixs_primary_final_evaluation/06_SEED_LEVEL_SUMMARY.csv`
- `reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed/FINAL_STATUS.json`
- `reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/{avgflow,ditmc}/RESULT.json`
- `reports/ecir_mvr/sixs_final_matched_ablation/STATUS.json`
- `reports/ecir_mvr/sixs_final_matched_ablation/05_PAIRED_BOOTSTRAP.csv`
- `reports/ecir_mvr/final_development_freeze/*`

```text
NO_MODEL_MODIFICATION = YES
NO_NEW_TRAINING = YES
NO_NEW_XTB = YES
NO_NEW_MMFF_RUN = YES
NO_MD_RUN = YES
```
