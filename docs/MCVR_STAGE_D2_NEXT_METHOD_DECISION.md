# MCVR Stage D2 Next Method Decision

Primary cause: **CONFIDENCE_CALIBRATION_WEAK**

Secondary causes: `EDGE_RESIDUAL_PREDICTION_WEAK, LOCAL_MULTI_BOND_INCONSISTENCY`.

Recommendation: **RECALIBRATE_BOND_CONFIDENCE**

D0 recovery `1.006277413498` minus D1-B recovery `0.197713981803` leaves `0.808563431695`. The approximate components sum to `0.808563431695` with nonadditive remainder `-0.000000000000`.

| Gap component | Recovery units |
|---|---:|
| residual_magnitude_error | 0.099349087655 |
| residual_sign_error | 0.015802511040 |
| missed_active_bond | 0.242256754434 |
| false_positive | 0.103159242071 |
| confidence_attenuation | 0.347995836495 |
| cartesian_bond_cancellation | 0.000000000000 |
| safety_attenuation | 0.000000000000 |
| acceptance | 0.000000000000 |
| ring_local_coupling_damage | 0.000000000000 |

Setting learned residual confidence to one raises recovery to `0.545709818298`, the largest single A-J gain, while the oracle active mask reaches only `0.153275775611`.

Confidence recalibration is therefore the clearest single next design change, but it does not fully close the D0 gap: residual correlation/recall and local multi-bond consistency remain secondary limitations.

The formal Stage D result remains **STAGE_D_NO_ADDED_VALUE**. This recommendation authorizes no implementation, training, 20k, 100k, seed43/44, or test evaluation.
