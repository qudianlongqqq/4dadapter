# MCVR Medium Failure Attribution Report

Primary cause: **MODEL_PROPOSAL_LIMITED**

Recommendation: **REDESIGN_LOCAL_BOND_PREDICTION**

The formal decision remains **MEDIUM_SEED42_SCHEDULE_V4_FAIL**. This audit is validation-only and did not train, alter checkpoints, read test data, or create a training command.

## Exact formal metric

| Quantity | Value |
|---|---:|
| Upstream bond outlier rate | 0.262031877853 |
| Accepted bond outlier rate | 0.235871849317 |
| Relative improvement | 0.099835290081 |
| Percent | 9.9835290081% |

## Stagewise attribution

| Factor | Signed loss | Target-gap share | Positive-loss share |
|---|---:|---:|---:|
| model_proposal | 0.147009521005 | 1.077095 | 0.991527 |
| atom_clipping | 0.000000000000 | 0.000000 | 0.000000 |
| graph_clipping | 0.000000000000 | 0.000000 | 0.000000 |
| safety_gate | 0.001256273858 | 0.009204 | 0.008473 |
| acceptance | -0.011778789565 | -0.086300 | 0.000000 |

The signed terms telescope to target gap `0.136487005298` with numerical error `0.000000000000`.

Minimal Target available relative improvement: `0.620714682376`.
Model-to-target recovery ratio: `0.160839259833`.

## Bond transitions

Accepted output repaired `1268` original outlier bond observations, left `9417` bad, and created `192` new outlier observations.
Minimal Target repaired `6295` and created `280`.

## Classification

| Category | Molecules | Records | Target-gap contribution |
|---|---:|---:|---:|
| MODEL_PROPOSAL_LIMITED | 420 | 575 | 59.210130169522 |
| CANCELLATION_OR_NEW_OUTLIER | 54 | 83 | 9.288416142575 |
| ALREADY_VALID_OR_NO_HEADROOM | 13 | 17 | -0.259338557720 |
| TARGET_LIMITED | 7 | 14 | -0.058866105974 |
| SAFETY_GATE_LIMITED | 4 | 8 | 0.072420284152 |
| MIXED | 2 | 3 | -0.009259283543 |

Counterfactual outputs are labeled `DIAGNOSTIC_ORACLE_ONLY` and were not used for checkpoint selection or Gate decisions.
No Rescue V5, seed43/44, 100k, or test evaluation was run.

## Verification

Targeted tests: `37 passed`.

Full repository tests: `353 passed`, `0 failed`.

Experimental test records read: `0`.
