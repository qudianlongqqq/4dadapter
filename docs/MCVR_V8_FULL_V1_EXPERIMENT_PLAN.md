# MCVR V8 Full v1 experiment plan

All parameter and checkpoint decisions use train and validation only. Formal test, Minimal
Validity Target test, frozen holdout, and one-time subsets remain inaccessible.

1. Build the train-only cohort manifest and full robust residual scales; freeze their identities.
2. Gate 1: real-train forward/backward with Full two-step mode, finite losses/deltas/gradients,
   bounded confidence/movement, and zero solver failures.
3. Gate 2: 100-step tiny overfit on a fixed train subset; require declining target/total loss,
   nonzero D1/new-head gradients, nonzero deltas, unsaturated confidence, and bounded movement.
4. Gate 3: 1K Full smoke with train/validation, checkpoint save/load/resume, frozen validation
   evaluator, solver contribution, gradient attribution, and anti-Bond diagnostics.
5. Gate 4: 5K Full Seed43, then matched D1-only and no-Angle with identical train/validation,
   sample exposure, optimizer steps, effective batch, checkpoint cadence, and evaluator.
6. Gate 5 priority: no-normalization, one-step, fixed-confidence; remaining prepared variants are
   no-Error-State, no-Clash, no-Ring/Chirality, and frozen-D1.

The available machine currently lacks the configured formal-large train/validation assets. Medium
train/validation development data is authorized for implementation gates and is reported as
development, never as formal-large or formal test. The formal-large 5K launch is blocked until the
four manifests and their source/target caches can be identity-bound.

Every run state records:

- `formal_test_records_read=0`
- `formal_test_assets_opened=false`
- `minimal_validity_target_test_used=false`
- `frozen_holdout_records_read=0`
- `parameter_selection_from_formal_test=false`
