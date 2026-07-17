# MCVR Stage D Bond-Explicit Method

Stage D is a new preregistered method stage, not Medium Rescue V5. The formal V4 decision remains `MEDIUM_SEED42_SCHEDULE_V4_FAIL`.

## D0 authorization

The validation-only D0 oracle passed all 11 registered conditions. Its accepted bond outlier relative improvement was `0.624611165102`, target recovery was `1.006277413498`, RMSD delta was `0.001250363614` Angstrom, and numerical failure fraction was `0`.

## Fixed D1 methods

Both candidates retain the Run A encoder, Cartesian head, safety gate, trust clipping, deterministic acceptance, four teacher steps, and zero torsion gate.

| Method | Bond auxiliary losses | Bond correction at inference | Alpha |
|---|---|---|---:|
| D1-A auxiliary-only | enabled | disabled | 0.0 |
| D1-B explicit-bond | enabled | enabled | 1.0 |

The head consumes symmetric endpoint embeddings, the frozen edge feature, current bond length, and time embedding. It predicts signed residual, confidence logit, and uncertainty. It never receives reference coordinates, Minimal Target coordinates, target residuals, nearest-reference information, validation statistics, or test statistics during inference.

Predicted residuals use `0.05 * tanh(raw) * sigmoid(confidence)` Angstrom. The `0.05` Angstrom cap was fixed from the train-only Minimal Target distribution: 348,650 unique-bond residuals across 7,500 training records had p90 `0.028477919102`, p95 `0.039713841677`, p99 `0.063252890706`, and `97.448157%` were at or below `0.05` Angstrom. No validation or test record was read for this choice.

Each molecule maps all unique-bond residuals jointly through `J^T (J J^T + 1e-4 I)^-1 r`, removes translation, and returns zero correction on numerical failure. Per-bond independent Cartesian moves are not used.

## Fixed supervision

The original Run A loss weights remain unchanged. Both D1 methods add the same fixed weights: residual `0.5`, direction `0.1`, sparsity `0.1`, confidence `0.05`, uncertainty `0.05`, and Cartesian-bond consistency `0.1`. Direction uses signed softplus with a fixed `0.01` Angstrom temperature to avoid a zero-vector cosine singularity.

Both runs start from step 0 with seed 42, batch size 8, 5,000 optimizer steps, 500-step warmup to `2e-4`, cosine decay to `2e-5`, and checkpoints at 500, 1000, 1500, 2000, 3000, and 5000. No resume, alpha sweep, 20k, 100k, seed43/44, or test evaluation is permitted.
