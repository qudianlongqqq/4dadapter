# SIXS final paper freeze scope

This release is a read/aggregate/audit/hash operation over completed authoritative artifacts. It performed no model change, training, inference, MMFF, xTB, V3D, PoseBusters, Reference, runtime, or other scientific evaluation.

## Evidence partitions

- CURRENT FINAL: prospective ETFlow cohort, 2,500 molecules and 5,000 records.
- CROSS-UPSTREAM: AvgFlow and DiTMC zero-shot evidence; never inserted into the current-final table.
- ABLATION: matched three-seed final ablation; never inserted as a current-final method.
- HISTORICAL: provenance only; historical DEV/Formal/10K values are excluded from final-number tables.

## Pre-freeze worktree inventory

- ` M .gitignore`
- `?? configs/sixs_final_cross_upstream_unrestricted.json`
- `?? configs/sixs_j1r1_joint_magnitude_interaction.json`
- `?? configs/sixs_musigma_reliability_factorial.json`
- `?? reports/ecir_mvr/final_cohort_identity_freeze/`
- `?? reports/ecir_mvr/final_evidence_closure/`
- `?? reports/ecir_mvr/final_paper_freeze/`
- `?? reports/ecir_mvr/final_paper_readiness_audit/`
- `?? reports/ecir_mvr/prospective_final_cohort_freeze/`
- `?? reports/ecir_mvr/sixs_final_ablation/`
- `?? reports/ecir_mvr/sixs_final_candidate_selection_audit/`
- `?? reports/ecir_mvr/sixs_final_cross_upstream_unrestricted/`
- `?? reports/ecir_mvr/sixs_final_matched_ablation/`
- `?? reports/ecir_mvr/sixs_final_project_integrity_generalization_audit/`
- `?? reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed/GPU_PREFLIGHT.json`
- `?? reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed/RUN_STATUS.json`
- `?? reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed/SUPERVISOR_TRACEBACK.txt`
- `?? reports/ecir_mvr/sixs_full_joint_final_capacity_audit/`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/BA_DISTRIBUTION.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/DEV_SUMMARY.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/FAILURE_DIAGNOSTICS.md`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/FINAL_CHECKPOINT_SHA256.txt`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/FINAL_DECISION.md`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/FULL_JOINT_MECHANISM_DIAGNOSTICS.md`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/GRADIENT_PATH_AUDIT.json`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/PAIRED_BOOTSTRAP.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/PARAMETER_GROUP_AUDIT.json`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/PB_COMPONENTS.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/PB_TRANSITIONS.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/PIPELINE_TRACEBACK.txt`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/RECOVERY_PREFLIGHT.json`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/REFERENCE_LEAKAGE_AUDIT.json`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/RELIABILITY_DIAGNOSTICS.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/SIGMA_DIAGNOSTICS.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/TAU_DISTRIBUTION.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/TRAIN_LOG.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/TRAIN_SUMMARY.json`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/V3D_COMPONENTS.csv`
- `?? reports/ecir_mvr/sixs_j1r1_full_joint_adaptive_ba_movement_seed307/V3D_TRANSITIONS.csv`
- `?? reports/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial/`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/00_PROTOCOL_FREEZE.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/01_IMPLEMENTATION_AUDIT.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/01_IMPLEMENTATION_AUDIT.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/02_J0_R0.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/03_J0_R1.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/04_J1_R0.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/05_J1_R1.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/06_J2_R0.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/07_J2_R1.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/08_PREDICTIVE_SIGMA_COMPARISON.csv`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/09_ACTION_COMPARISON.csv`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/10_PAIRED_BOOTSTRAP.csv`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/11_FACTORIAL_ANALYSIS.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/12_CROSS_UPSTREAM.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/13_FINAL_DECISION.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/ASYNC_PIPELINE_FEASIBILITY.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/ASYNC_PIPELINE_IMPLEMENTATION_AUDIT.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/ASYNC_SUPERVISOR_STATUS.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/CPU_EVALUATION_QUEUE_STATUS.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/CUDA_EVALUATOR_NATIVE_CRASH_AUDIT.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/FINAL_DECISION.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/GPU_TRAIN_QUEUE_STATUS.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/J0_R1_CPU_EVAL_SUPERVISOR_STATUS.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/LIVE_RESULTS.csv`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/LIVE_SUMMARY.md`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/RUN_STATUS.json`
- `?? reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda/resource_audit_runtime/`
- `?? reports/ecir_mvr/sixs_objective_module_gradient_forensic_audit/`
- `?? reports/ecir_mvr/sixs_objective_read_only_audit/`
- `?? reports/ecir_mvr/sixs_primary_final_evaluation/`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/01_REFERENCE_RMSD_INTEGRITY_AUDIT.md`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/02_MATCHED_SOURCE_REFERENCE_RMSD.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/03_MMFF_XTB_MATCHED_COMPARISON.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/04_XTB_ENERGY_COMPARISON.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/05_CURRENT_FINAL_EVIDENCE_SUMMARY.md`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/PHASE_I_COMPLETE.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/01_CURRENT_FINAL_EVIDENCE/PHASE_I_STATUS.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/CURRENT_VS_UNRESTRICTED_BOOTSTRAP.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/CURRENT_VS_UNRESTRICTED_PAIRED.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/FINAL_CAPABILITY_DECISION.md`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/FINAL_CHECKPOINT_SHA256.txt`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/GPU_PREFLIGHT.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/GPU_TRAINING_VERIFICATION.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/GRADIENT_PATH_AUDIT.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/PARAMETER_GROUP_AUDIT.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/PIPELINE_TRACEBACK.txt`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/TRAIN_SUMMARY.json`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_LOCALITY_AUDIT.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_MOVEMENT_DISTRIBUTION.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_PB.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_REFERENCE_RMSD.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_TRAIN_LOG.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_V3D.csv`
- `?? reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/UNRESTRICTED_XTB.csv`
- `?? reports/ecir_mvr/sixs_step3a_etflow_source_generation/06_SOURCE_COMPLETION_INTEGRITY_AUDIT.json`
- `?? reports/ecir_mvr/sixs_step3a_etflow_source_generation/07_SOURCE_ASSET_FREEZE.json`
- `?? scripts/audit_factorial_external_sdf.py`
- `?? scripts/audit_factorial_post_sdf_core.py`
- `?? scripts/audit_factorial_training_dry.py`
- `?? scripts/audit_sixs_final_candidate_selection.py`
- `?? scripts/audit_sixs_final_project_integrity_generalization.py`
- `?? scripts/audit_sixs_objective_module_gradient_forensic.py`
- `?? scripts/audit_sixs_objective_weighting_read_only.py`
- `?? scripts/audit_sixs_step3a_source_completion.py`
- `?? scripts/build_sixs_step2d_historical_exclusion.py`
- `?? scripts/download_google_drive_ranged.py`
- `?? scripts/evaluate_sixs_final_cross_upstream_external.py`
- `?? scripts/evaluate_sixs_final_cross_upstream_reference_rmsd.py`
- `?? scripts/evaluate_sixs_primary_final_external.py`
- `?? scripts/extract_sixs_step2d_torsional_assets.py`
- `?? scripts/finalize_sixs_existing_ablation_suite.py`
- `?? scripts/finalize_sixs_final_evidence_closure.py`
- `?? scripts/finalize_sixs_final_matched_ablation.py`
- `?? scripts/finalize_sixs_final_paper_freeze.py`
- `?? scripts/finalize_sixs_final_project_integrity_generalization.py`
- `?? scripts/finalize_sixs_full_joint_capacity_audit.py`
- `?? scripts/finalize_sixs_primary_final_evaluation.py`
- `?? scripts/finalize_sixs_step2d_freeze.py`
- `?? scripts/finalize_sixs_unrestricted_movement_capability.py`
- `?? scripts/materialize_sixs_final_reference_context.py`
- `?? scripts/recover_sixs_restricted_multiseed_posttrain.py`
- `?? scripts/resume_factorial_after_j0_r1_eval.ps1`
- `?? scripts/resume_sixs_final_multiseed_after_restricted331_recovery.py`
- `?? scripts/run_sixs_factorial_async_supervisor.py`
- `?? scripts/run_sixs_final_avgflow_reference_audit.py`
- `?? scripts/run_sixs_final_closure_xtb.py`
- `?? scripts/run_sixs_final_cross_upstream_unrestricted.py`
- `?? scripts/run_sixs_final_cross_upstream_xtb.py`
- `?? scripts/run_sixs_final_matched_ablation_evaluation.py`
- `?? scripts/run_sixs_final_matched_ablation_training.py`
- `?? scripts/run_sixs_final_matched_ablation_xtb.py`
- `?? scripts/run_sixs_final_mmff94s_repair.py`
- `?? scripts/run_sixs_full_joint_capacity_extension.py`
- `?? scripts/run_sixs_full_joint_final_capacity_audit.py`
- `?? scripts/run_sixs_j1r1_joint_magnitude_interaction.py`
- `?? scripts/run_sixs_j1r1_joint_magnitude_supervisor.py`
- `?? scripts/run_sixs_post_cross_upstream_ablation_pipeline.ps1`
- `?? scripts/run_sixs_post_cross_upstream_ablation_pipeline.py`
- `?? scripts/run_sixs_primary_final_coordinates.py`
- `?? scripts/run_sixs_primary_final_xtb.py`
- `?? scripts/start_sixs_final_cross_upstream_unrestricted.ps1`
- `?? scripts/start_sixs_final_evidence_closure.ps1`
- `?? scripts/start_sixs_primary_final_evaluation.ps1`
- `?? scripts/supervise_sixs_final_cross_upstream_unrestricted.py`
- `?? scripts/supervise_sixs_final_evidence_closure.py`
- `?? scripts/supervise_sixs_primary_final_evaluation.py`
- `?? scripts/wait_and_finalize_sixs_final_multiseed.py`

Large checkpoints, coordinates, per-record datasets, tool caches, and work directories remain external/ignored and are SHA256-bound by `11_FINAL_RELEASE_MANIFEST.json`.

## Engineering/report diff audit

| Change | Evidence | Class | Scientific semantics changed | Reason |
|---|---|---|---|---|
| STATUS.json BOM tolerance | `scripts/run_sixs_post_cross_upstream_ablation_pipeline.py:42-61` | ENGINEERING_FIX | NO | UTF-8/UTF-8-BOM orchestration serialization only |
| tabulate-independent Markdown | `scripts/run_sixs_final_cross_upstream_unrestricted.py:493-502; scripts/run_sixs_musigma_reliability_factorial.py:142-147` | REPORT_ONLY | NO | Formatting fallback only |
| pipeline recovery | `scripts/run_sixs_post_cross_upstream_ablation_pipeline.py:161-176` | ENGINEERING_FIX | NO | Same frozen stage identity/state; no outcome-dependent branch |
| MMFF num_atoms restoration | `scripts/run_sixs_primary_final_coordinates.py:107-110; scripts/run_sixs_final_mmff94s_repair.py:57-64` | ENGINEERING_FIX | NO | Restores authoritative topology count required by validator |
| Reference xTB cache/workdir dedup | `scripts/run_sixs_final_closure_xtb.py:90-120` | ENGINEERING_FIX | NO | Executes identical scientific identity once and fans immutable result to frozen duplicate rows |
| finalizer molecule_id suffix repair | `scripts/finalize_sixs_final_evidence_closure.py:332-343` | ENGINEERING_FIX | NO | One-to-one merge plus explicit zero-mismatch identity assertion |

`SCIENTIFIC_SEMANTICS_CHANGED = NO`

Planned immutable tag: `sixs-final-evidence-freeze-2026`. The tag target is the release commit and resolves the deliberate non-self-referential commit binding in the manifest.
