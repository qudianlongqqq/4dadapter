# Minimum baseline set

| Baseline | Classification | Decision |
|---|---|---|
| Source | MUST_HAVE | Already intrinsic to every paired refinement evaluation; retain as the primary no-refinement control |
| MMFF94s geometry optimization | MUST_HAVE_FOR_SUBMISSION | Cheap, familiar local optimization comparator on the exact final cohort; run only after protocol freeze, not during this audit |
| GFN2-xTB geometry optimization | SHOULD_HAVE | Useful physics-based quality/cost comparator, especially if the paper contrasts learned refinement with quantum optimization; single-point xTB is not this baseline |
| Reference | MUST_HAVE_AS_TARGET | Context/evaluation target, not a generative baseline; preserve the frozen correspondence/Kabsch protocol |
| Other learned refiner | OPTIONAL | Required only if an exact same-input, fixed-topology, comparable-output method exists and can be run fairly |
| Other upstream | NOT_REQUIRED_FOR_CURRENT_CLAIM | Needed only for a cross-upstream or general-refiner claim; DiTMC large holdout may be secondary evidence |

For the deliberately narrow ETFlow-local-refinement claim, MMFF94s is the only missing baseline judged submission-critical. GFN2-xTB optimization is strongly informative but not required to open the primary final test; it becomes effectively mandatory if the paper claims superiority to physics-based optimization rather than a speed/quality operating point.

```text
MUST_RUN_BASELINES = MATCHED_MMFF94S_GEOMETRY_OPTIMIZATION
SHOULD_RUN_BASELINES = MATCHED_GFN2_XTB_GEOMETRY_OPTIMIZATION
OTHER_LEARNED_REFINER_REQUIRED = NO
CROSS_UPSTREAM_BASELINE_REQUIRED = NO
```
