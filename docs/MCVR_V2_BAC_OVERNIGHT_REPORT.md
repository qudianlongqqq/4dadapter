# MCVR V2-BAC Overnight Report

## Decision

`METHOD_EFFECT_NOT_VALIDATED`: the unified method is implemented and numerically safe, but the current 2k pilot does not justify 10k or formal-large training.

All development and selection used train plus validation only. `test_records_read=0`, `test_assets_opened=false`, and `validation_only=true` for every artifact and run.

## Method answers

1. The target is one deterministic projected BAC optimization, not a sum of three coordinate targets.
2. The model emits one fused Cartesian `delta_x`; no sequential Bond/Angle/Clash coordinate states exist.
3. `V2_A_BOND_ONLY` has the exact D1-B parameter/state surface and passed fixed-forward, loss, correction, and strict frozen-checkpoint regression tests.
4. Angle triplets are explicit, canonical, permutation-consistent, and SE(3)-safe. They receive gradients, but incremental holdout Angle improvement over A is zero.
5. Clash edges are dynamic, sparse, topology-excluded, deterministic after sorting, and avoid dense pair allocation. Holdout Clash delta is numerical noise.
6. Ring, identity, chirality, finite coordinates, and trust radii are hard acceptance constraints; failures roll back to the exact source.
7. RMSD, MAT-P/R, COV-P/R, identity, and chirality are reported as public protocol metrics. GenBench3D and PoseBusters are unavailable and were not approximated as official results.
8. Bond/Angle/Ring outlier diagnostics, total validity, acceptance, and displacement remain explicitly custom diagnostics.
9. Four matched 2k runs were executed: A Bond-only, B Bond+Angle, C Bond+Clash, and D unified BAC. Two two-batch diagnostics and one 200-step D smoke preceded them.
10. One implementation correction preceded the 200-step smoke: robust standardized residual clipping and isolation of new-branch supervision fixed measured Angle gradient domination. No post-tune network adjustment was made.
11. A and D entered holdout: A was the strongest Bond baseline; D was the only unified candidate with the best tune Angle delta under hard constraints.
12. The main conflict is efficacy versus rollback: small validity gains coincide with 93.6-97.5% tune rollback.
13. Both candidates pass hard holdout safety/noninferiority, but D fails scientific incremental efficacy because Angle equals A and Clash is unchanged.
14. The completed 2k stage is sufficient to reject 10k/formal-large escalation for this implementation.
15. Keep hidden=64 and layers=4. Do not expand capacity; investigate target-to-proposal scale and acceptance bottlenecks in a new train/tune-only preregistered stage.
16. All test access counters are zero.

Process deviation: counting both two-batch diagnostics as GPU training runs gives seven optimizer invocations (2 diagnostics + 1 smoke + 4 pilots), exceeding the strict maximum of six by one. No additional run was launched after this audit.

## Target assets

- Train-only pilot targets: 4096 records
- Success: 4010
- Already valid: 17
- Safe fallback: 69 (1.6846%)
- Formal train/validation assets were not modified.

## Tune comparison

| Mode | Params | Bond delta | Angle delta | Clash delta | RMSD delta | Acceptance | Rollback |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2_A_BOND_ONLY | 384678 | -0.00149322802 | -8.76013748e-06 | 0 | 6.51818816e-07 | 6.4000% | 93.6000% |
| V2_B_BOND_ANGLE | 410603 | -0.00054807244 | -1.10749523e-05 | -7.73934516e-12 | 5.06547513e-08 | 2.5000% | 97.5000% |
| V2_C_BOND_CLASH | 406507 | -0.000806816729 | -4.85482626e-06 | 0 | 2.44262512e-07 | 3.5625% | 96.4375% |
| V2_D_BOND_ANGLE_CLASH | 427694 | -0.00101867348 | -1.82798005e-05 | -1.80216375e-11 | 2.01184768e-07 | 4.6500% | 95.3500% |

## Frozen holdout

| Mode | Bond delta | Angle delta | Clash delta | RMSD delta | Acceptance | Rollback |
|---|---:|---:|---:|---:|---:|---:|
| V2_A_BOND_ONLY | -0.00131709809 | -1.749992e-05 | 0 | 5.50404191e-07 | 5.9000% | 94.1000% |
| V2_D_BOND_ANGLE_CLASH | -0.00100799759 | -1.749992e-05 | -7.17727744e-12 | 1.91926956e-07 | 4.6000% | 95.4000% |

Holdout evaluation count is exactly one per candidate. Further tuning on this holdout is prohibited.

## Recommendation

Do not launch 10k or formal-large V2-BAC. Retain the 64x4 backbone. A future stage must preregister a train/tune-only attribution of target residual scale, new-branch output scale, fusion attenuation, trust clipping, and rejection reasons. The frozen holdout cannot be reused for that work.
