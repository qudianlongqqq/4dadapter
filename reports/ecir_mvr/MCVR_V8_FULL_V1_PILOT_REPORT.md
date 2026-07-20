# MCVR V8 Full v1 pilot status

Status: `MCVR_V8_FULL_V1_BLOCKED`.

This report contains development validation only. It is not a formal-test report.

Completed implementation and gates: `['code_audit', 'method_and_experiment_plan', 'unified_differentiable_constraint_layer', 'end_to_end_d1_training_path', 'two_step_relinearized_unrolling', 'train_only_residual_scales', 'train_only_stratified_sampler', 'losses_diagnostics_and_safety', 'checkpoint_resume', 'gate_1_forward_backward', 'gate_2_hard_tiny_overfit', 'gate_3_development_1k_smoke', 'development_validation_safety_gate', 'v8_and_focused_regression_tests']`.

Formal pilot running: `None`.

Missing required assets: `['E:\\3dconformergenerationcode\\4dadapter-v8\\data\\ecir_mvr\\formal_large\\real_sources\\train.parquet', 'E:\\3dconformergenerationcode\\4dadapter-v8\\data\\ecir_mvr\\formal_large\\minimal_targets\\train.parquet', 'E:\\3dconformergenerationcode\\4dadapter-v8\\data\\ecir_mvr\\formal_large\\real_sources\\val.parquet', 'E:\\3dconformergenerationcode\\4dadapter-v8\\data\\ecir_mvr\\formal_large\\minimal_targets\\val.parquet', 'E:\\3dconformergenerationcode\\4dadapter-v8\\data\\flexbond_cache_formal_large', 'E:\\3dconformergenerationcode\\4dadapter-v8\\data\\ecir_mvr\\formal_large\\minimal_targets']`.

## Development validation

- accepted: `1.0`
- angle_delta: `-0.027933551874011756`
- bond_delta: `-0.12733716875314713`
- chirality_preserved: `1.0`
- clash_delta: `7.41425174055621e-07`
- confidence_mean: `2.5997556138038633`
- max_atom_displacement: `0.038099544942379`
- mean_displacement: `0.012961152750067413`
- ring_delta: `-0.04845001481473446`
- solver_angle_contribution: `7.441364067980808`
- solver_bond_contribution: `14.372633826923556`
- solver_failure_count: `0.0`
- target_loss: `3.9382253480653165e-05`
- weighted_bac_delta: `-0.8259754329524094`

Isolation: formal_test_records_read=0; formal_test_assets_opened=false; frozen_holdout_records_read=0.
