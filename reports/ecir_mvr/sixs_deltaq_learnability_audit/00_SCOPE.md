# DeltaQ learnability root-cause audit scope

This audit is diagnostic only. The frozen 1,250-step small-overfit result remains `FAIL`; its threshold is not changed. No DEV or Formal outcome is read, and full training is not authorized.

The audit reuses the exact 32 frozen TRAIN records and seed 307. It examines the persisted learning curve, target distribution, initial/final J1-sigma state, and full-TRAIN representation ambiguity. The only new optimization is the separately labelled head-only diagnostic defined in `04_HEAD_ONLY_PROTOCOL.md`.
