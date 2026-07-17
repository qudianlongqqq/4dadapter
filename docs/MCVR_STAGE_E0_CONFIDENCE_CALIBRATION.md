# MCVR Stage E0 Confidence Calibration

Stage E0 fits only a monotonic two-parameter confidence map on training molecules. The D1-B checkpoint, residual predictor, Cartesian branch, solver, safety, trust clipping, acceptance, and torsion-disabled architecture remain frozen.

## Calibrator

| Field | Value |
|---|---:|
| raw_a | -0.765382106693 |
| a = softplus(raw_a) + epsilon | 0.381961839096 |
| b | -0.074037880430 |
| Internal-check original MAE | 0.243249964591 |
| Internal-check calibrated MAE | 0.322611995207 |

`confidence_all_one` remains `DIAGNOSTIC_ORACLE_ONLY` and is not deployable.

Calibration fitting read training data only. Validation was evaluated once after fitting; test records read remained zero.
