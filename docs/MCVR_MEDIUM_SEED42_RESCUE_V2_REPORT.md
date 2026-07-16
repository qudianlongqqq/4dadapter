# MCVR Medium Seed42 Rescue V2 Final Report

Decision: **MEDIUM_SEED42_FAIL**

Rescue V2 preserved batch size 8, effective batch size 8, learning rate 0.0002, 20,000 optimizer steps, model, loss, and data mixture. It changed only the invalid standalone velocity-growth stop semantics and allowed operational throughput controls.

## Timing and completion

| Item | Value |
|---|---:|
| Pipeline wall seconds | 500.453 |
| Training wall seconds | 344.625 |
| Active optimizer seconds | 159.791 |
| Validation seconds | 77.937 |
| Checkpoint I/O seconds | 0.218 |
| Final evaluation seconds | 71.172 |
| Bootstrap seconds | 2.109 |
| Report seconds | 0.062 |
| Completed optimizer steps | 2450 / 20000 |
| Stop reason | velocity_graph_rms_hard_limit |
| Mean wall seconds per completed 1000-step interval | 145.945 |
| Mean active seconds per completed 1000-step interval | 65.257 |
| Mean optimizer steps/s | 15.3325 |
| Mean examples/s | 122.6602 |
| Estimated 100k active hours (estimate only) | 1.812 |
| Automatic recovery occurred | no |
| Selected checkpoint step | 2000 |

## GPU and memory

| Item | Value |
|---|---:|
| Peak PyTorch allocated MiB | 51.5 |
| Peak PyTorch reserved MiB | 74.0 |
| Peak whole-card used MiB | 3783.0 |
| Minimum whole-card free MiB | 12195.0 |
| Shared-memory usage | unavailable from the GPU driver query |
| GPU utilization mean | 11.90% |
| GPU utilization p95 | 14.00% |

Low memory occupancy is not treated as evidence of an invalid run.

## Checkpoint validation

| Step | Validity delta | RMSD delta | MAT-P delta | MAT-R delta | High-flex validity delta | Unseen validity delta | Accuracy noninferior |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | -0.015900 | 0.000167 | 0.000167 | 0.000127 | -0.016532 | -0.026006 | True |
| 2000 | -0.090117 | 0.000700 | 0.000700 | 0.000995 | -0.090919 | -0.211591 | True |

## Final selected-checkpoint metrics

| Metric | Upstream | Rescue V2 accepted | Delta |
|---|---:|---:|---:|
| Total thresholded validity | 0.793989 | 0.703860 | -0.090128 |
| Aligned RMSD | 1.321787 | 1.322487 | 0.000700 |
| MAT-P | 1.321787 | 1.322487 | 0.000700 |
| MAT-R | 2.375986 | 2.376981 | 0.000995 |
| COV-P | 0.482000 | 0.482000 | 0.000000 |
| COV-R | 0.068843 | 0.068776 | -0.000067 |

High-flex accepted validity: `0.690949`.
Unseen-scale accepted validity: `1.339266`.
Clean identity fraction: `1.000000` (clean summary unchanged fraction `1.000000`).

## Gate 2

Passed conditions: **26/27**.

Failed conditions: `02_one_core_metric_relative_improvement_ge_10pct`.

Seed43/44 permitted for a future separately authorized task: **no**.
Seed43 and seed44 were not run and no launch command was generated.
100k remains prohibited and was not run. Test records read: 0.

## Per-1000-step timing

| Step end | Interval s | Active optimizer s | Steps/s | Examples/s | GPU mean/p95 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 154.171 | 65.420 | 15.2858 | 122.2256 | 11.2 / 13.0 |
| 2000 | 137.719 | 65.094 | 15.3624 | 122.8377 | 12.3 / 14.1 |
