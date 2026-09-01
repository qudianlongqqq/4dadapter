# Ablation completeness

## Decision

```text
CORE_ABLATION_COVERAGE = PARTIAL
NO_NEW_ABLATION_REQUIRED_FOR = J1_DEVELOPMENT_PROVENANCE; R1_DEVELOPMENT_PROVENANCE; LEARNED_MAGNITUDE_SEED307_PROVENANCE
MISSING_CORE_ABLATIONS = FINAL_FORMULATION_ADAPTIVE_BA; BOND_VS_ANGLE_PRIMITIVES; PREDICTIVE_SIGMA_AS_ACTION_WEIGHT
```

| question | existing evidence | quality | missing evidence | paper importance |
|---|---|---|---|---|
| Does J1 improve over other mu/sigma training routes? | frozen J0/J1/J2 x R0/R1 six-arm factorial | matched seed307 DEV with paired molecule bootstrap | replication is absent; exact beta value was not isolated | HIGH, but no broad sweep is needed |
| Does Reliability add value? | R0/R1 effects within every J; V3D +0.0326/+0.0560/+0.0388 | strong matched development evidence | no-R1 arm in exact final full-joint training | HIGH; existing staged evidence is usable |
| Is Adaptive BA independently necessary? | early BA-v1/v2 mixed-neutral; joint/full-joint later positive | conflicting/stage-dependent and not causal in final increment | exact final training with Equal BA and all else matched | HIGH |
| Is learned magnitude useful? | fixed vs learned magnitude under J1-R1; positive V3D interaction | matched seed307 evidence | multiseed mechanism replication | HIGH; current evidence adequate as development ablation |
| Are Restricted constraints useful? | Restricted vs Unrestricted removes regularizer, bound, cap, rollback as a bundle | matched seed307, but bundle ablation | individual regularizer/bound/cap effects | MEDIUM; individual sweep is not required for the main formulation comparison |
| Are both Bond and Angle primitives needed? | component improvements and legacy family studies | indirect/current-family effects are visible | final Bond-only and Angle-only action ablations | HIGH if both families are claimed as essential |
| Does predictive sigma improve action beyond fixed sigma_stat? | J-factorial evaluates training variants; prior fixed/predictive routes exist historically | not a clean exact-final action-weight ablation | final fixed-sigma-stat/no-predictive-sigma action control | HIGH if uncertainty weighting is a headline contribution |
| Are rigid projection and graph-RMS normalization necessary? | invariance/stability rationale and finite diagnostics | theoretical/engineering, not causal | deletion controls | LOW; do not run unless making a performance-necessity claim |

## Minimal evidence plan after formulation freeze

This audit does not authorize execution. If the paper claims all named modules as independent innovations, the smallest informative control set is:

1. final architecture with Equal BA instead of Adaptive BA;
2. final action with fixed `sigma_stat` weighting instead of predictive sigma;
3. Bond-only and Angle-only action inference controls.

Do not add a parameter sweep, new controller, second-order method, tau grid, or individual safety-constant sweep. The existing Restricted/Unrestricted bundle comparison is already the correct formulation-level safety/capability experiment.

