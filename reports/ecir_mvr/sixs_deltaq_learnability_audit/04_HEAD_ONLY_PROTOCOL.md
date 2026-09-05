# Frozen head-only learnability diagnostic

Frozen before outcome:

```text
ROLE = DIAGNOSTIC_ONLY__NOT_REPLACEMENT_SANITY_RESULT
DATA = SAME_FROZEN_32_TRAIN_RECORDS
SEED = 307
MODEL = SAME_BACKBONE_PLUS_SOURCE_CONDITIONED_DELTAQ_HEADS
TRAINED_GROUPS = BACKBONE; DELTAQ_HEADS
EXCLUDED_FROM_LOSS_AND_OPTIMIZER = SIGMA; RELIABILITY; ADAPTIVE_BA; TAU; CARTESIAN_ACTION
LOSS = 0.5 * (BOND_MSE + ANGLE_COSINE_MSE)
OPTIMIZER = ADAMW
BACKBONE_LR = 1.5e-4
DELTAQ_HEAD_LR = 3.0e-4
WEIGHT_DECAY = 1.0e-6
BATCH = ALL_32_FIXED_RECORDS
STEPS = 5000
SCHEDULER = COSINE_ANNEALING_TMAX_5000
GRADIENT_CLIP = 1.0
```

PASS is predeclared as: finite outputs; final Bond and Angle MAE each no more than 10% of its own initial MAE; and signed prediction/target correlation at least 0.90 for both families. This threshold is a diagnostic definition of near-memorization and cannot change after outcome.
