# MCVR Stage G/H0 Dependency Audit

Stage G entry points directly import `bounded_residual_confidence`, Stage F evaluation helpers, Stage E0 data loaders, D1-B model/safety modules, pandas, PyTorch, PyG, NumPy, scikit-learn, YAML, and RDKit-backed validity code. Stage H0 directly imports `conflict_aware_fusion` and the same HEAD-resident D1-B/Stage F evaluation stack.

The shared runtime dependencies (`confidence_calibration.py`, `feature_conditioned_confidence.py`, `evaluate_ecir_mvr_stage_f.py`, `evaluate_ecir_mvr_stage_e0.py`, model, acceptance, validity, audit and evaluation helpers) are present in HEAD `40858ae`. No Stage E1 dirty file or tracked state/report modification is required.

The minimum dirty source closure is therefore the Stage G eight-file implementation and tests, followed by the Stage H0 seven-file implementation and tests. Formal diagnostics, calibration parquet, checkpoints, caches and smoke directories are runtime assets and must not be committed. Pulling only entry scripts without the two new core modules/configurations would fail on Linux.
