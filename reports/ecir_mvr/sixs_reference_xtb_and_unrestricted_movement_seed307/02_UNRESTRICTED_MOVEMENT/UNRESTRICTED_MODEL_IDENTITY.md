# Unrestricted Full Joint model identity

This is an independent seed307 capability branch. The frozen current final model is unchanged. The branch starts from the same Softplus-v2 backbone and mu initialization and trains all J1/R1/Adaptive-BA/magnitude modules for exactly 17,500 steps.

```text
MOVEMENT_REGULARIZER = NONE
TAU = softplus(raw)
INITIAL_TAU_TARGET = 0.003
INITIAL_LOGIT_OR_RAW_PARAMETER = -5.807642615314055
TAU_FINITE_UPPER_BOUND = NONE
PER_ATOM_CAP = NONE
ROLLBACK_USED = NO
TRAINING_DEVICE = cuda
CPU_TRAINING_FALLBACK = NO
```
