# Minimum remaining ablation assessment

The minimum is claim-dependent. For the scoped claim—an end-to-end learned ETFlow local refiner with two movement operating points—no new ablation is required before the final test. Claims that an individual module is uniquely necessary require the corresponding exact-formulation control.

| Item | Existing evidence | Final-formulation matched | Multiseed required | Paper claim depends on it | Must run | Why |
|---|---|---:|---:|---:|---:|---|
| Adaptive BA | Historical V3 BA/movement joint experiment and mechanism audits | No | No | Only for an independent Adaptive-BA necessity claim | No | Drop/qualify the isolated necessity claim; the final end-to-end method remains evaluable |
| Predictive sigma action weighting | J0/J1/J2 factorial; predictive uncertainty alone was insufficient | Partly | No | Only for a calibrated/independently necessary sigma claim | No | Describe sigma as a learned action-weighting input; fixed-sigma control is Tier 2 unless elevated to a headline claim |
| Bond-only / Angle-only | Bond and angle components and transitions are measured, but no exact final single-family training controls | No | No | Only for claims that both families are individually indispensable | No | Use a joint-local-geometry claim; do not infer causal necessity for each family |
| Reliability | Six-arm R0/R1 factorial across J0/J1/J2; positive/neutral/positive effects | Staged, not exact full-joint | No | Yes, as a supported component | No | Existing factorial directly isolates Reliability and is sufficient for the scoped component claim |
| Movement magnitude | Matched J1-R1 fixed versus joint-magnitude experiment | Yes for the then-current core; staged before Adaptive BA | No | Yes | No | Existing matched interaction supports learned magnitude; multiseed further establishes complete-policy stability |
| Restricted constraints | Three-seed Restricted/Unrestricted matched comparison | Yes | Completed | Yes, as a trade-off rather than necessity | No | Bundle-level trade-off is resolved; decomposing tau ceiling, atom cap and regularizer is not needed |

```text
MUST_RUN_ABLATIONS = NONE
NO_NEW_ABLATION_REQUIRED = YES
CONDITIONAL_TIER2_ABLATIONS = ADAPTIVE_BA_FINAL_CONTROL; FIXED_SIGMA_STAT_ACTION_WEIGHT; BOND_ONLY; ANGLE_ONLY
UNNECESSARY_EXPERIMENTS = RESTRICTED_CONSTRAINT_COMPONENT_SWEEP; BETA_SWEEP; LOSS_RATIO_SWEEP; MORE_TRAINING_STEPS
```
