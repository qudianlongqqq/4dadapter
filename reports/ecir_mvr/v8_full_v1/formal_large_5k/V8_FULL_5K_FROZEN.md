# MCVR V8 Full v1 Formal-Large 5K Freeze

The Seed43 formal-large Full V8 pilot completed all 5,000 optimizer steps and its 10,000-record validation. This freeze binds the completed run to method HEAD `6b3b58c2a1d0b920e1970b7a7b2d0e55ca21f3b4` and training launch commit `e53f1813ba978c2147001fcd2b4fbb2c4445239e`.

The immutable run remains under `diagnostics/ecir_mvr/v8_full_v1/formal_large_5k/full_seed43`. No subsequent performance work may overwrite its checkpoint, validation output, logs, or status. Checkpoints and diagnostics remain outside Git; this directory contains only small identity and freeze records.

Validation optimization parity is measured against this frozen run. Its 10K validation result must not be used to change V8 architecture, losses, solver weights, confidence bounds, sampler weights, damping, displacement limits, or any other scientific parameter. Later changes are limited to evaluator scheduling, caching, I/O, batching, parallelism, and status reporting that preserve the frozen semantics.

## Frozen result

- Acceptance: 98.49%
- Weighted BAC delta: -0.1837117030954342
- Bond delta: -0.07969993709446863
- Angle delta: -0.003652969984151423
- Ring delta: -0.01767894923426211
- Chirality preserved: 100%
- Mean/max atom displacement: 0.0038525349 / 0.0197616151 Å
- Solver failures: 0

The exact hashes and identities are recorded in `V8_FULL_5K_FROZEN.json` and `V8_FULL_5K_SHA256SUMS.txt`.

## Isolation

- `formal_test_records_read=0`
- `formal_test_assets_opened=false`
- `minimal_validity_target_test_used=false`
- `frozen_holdout_records_read=0`
- `parameter_selection_from_formal_test=false`
