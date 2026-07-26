# Test audit

- New mechanism suite: **15 passed**.
- Focused inherited LSGO + BAT + new mechanism suites: **108 passed**.
- Broader `test_mcvr*` + external-refinement collection: **194 passed, 2 failed**. Both failures are unrelated historical integration tests whose large ignored cache files are deliberately absent from this isolated worktree (`raw/smoke100/evaluation.json` and `validation_cache/.../prediction_manifest.json`); no mechanism assertion failed.
- Full repository collection is not available in this lean environment because optional legacy dependencies `torch_cluster` and `pydantic` are not installed. This is an environment limitation, not a new test regression.

The new suite covers Reference ensemble identity, energy-coordinate identity, B/A masks, exact BA equivalence, TRAIN-calibrated threshold freeze, no-op mask, B/A/T finite differences, BA/BAT joint SVD ranks, small-singular-value handling, rigid-mode removal, force projection, xTB force protocol/order, protected splits and artifact SHA identities.
