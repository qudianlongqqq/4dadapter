# ECIR Stage 2 root-cause analysis

Stage A establishes three linked causes for the old `NO_GO`:

1. **Source imbalance:** the 5k ECIR model performs substantial repair only on the extreme Cartesian 100k source. That source also contributes essentially all RMSD/MAT degradation. ETFlow inputs move only `0.0065 Å` on average and remain close to accuracy-neutral.
2. **Cartesian rollout protocol:** the formal ten-step rollout is identity-correct and exactly reproducible, but it evaluates time up to `1.0` although the checkpoint was trained only through `0.25`. Internal errors and displacement grow monotonically from one to ten steps. This is classification B (multi-step divergence), not a molecule/atom/unit/cache mismatch.
3. **Target is not minimal:** 595/600 relaxed targets hit the iteration limit, all 300 Cartesian targets are non-converged, and Cartesian labels move `0.269 Å` aligned RMS on average. The target therefore teaches broad reconstruction rather than minimal validity repair.

The old ECIR is consequently rewarded for undoing an artificial, rollout-amplified source with a large non-minimal target. Its gate learns a strong source-severity response (`0.102` Cartesian versus `0.036` ETFlow), producing internal improvements but paying accuracy and diversity costs on the extreme source.

No evidence indicates duplicated update scale, repeated cache correction, atom-order mismatch, hydrogen mismatch, coordinate-unit mismatch, or Kabsch write-back. The missing selected reference ID in the Stage 2 cache is a provenance warning but does not explain the observed drift because the persisted reference tensor is stable and SHA-audited.

Stage A decision is `HOLD_FOR_STAGE_B`: conservative inference scanning is allowed next, but no new training is allowed until Stages B and C are complete.
