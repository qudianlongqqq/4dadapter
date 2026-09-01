# SIXS J1-R1 full-joint model identity

This is one new seed307 optimization run. The standard six-arm base-geometry initialization loads only the frozen Softplus-v2 backbone and mu scope; it does not load J1-R1, learned-magnitude, Adaptive-BA-v2, or V3 Joint BA-Movement candidate weights. J1 sigma, R1 Reliability, Adaptive BA, and magnitude use their deterministic standard initializations and every scientific module is trainable.

The total objective is `L_J1_beta_NLL + L_post + lambda_move * L_move`. Reliability supervision and post loss are the same mathematical object and are not counted twice. Adaptive BA has no auxiliary target. The Cartesian primitive derivatives remain detached and first order.

```text
FULL_JOINT_TRAINING = YES
WARM_START_FROM_SCIENTIFIC_CHECKPOINT = NO
BASE_GEOMETRY_PROTOCOL_INITIALIZATION = SOFTPLUS_V2_BACKBONE_AND_MU_ONLY
NAIVE_MU_SIGMA_JOINT_REINTRODUCED = NO
SECOND_ORDER_REQUIRED = NO
REFERENCE_USED_AT_INFERENCE = NO
```
