# Final submission readiness

## Decision

The method, prospective cohort, three-seed primary result, cross-upstream executions, and core ablations are substantial. The paper is not ready to freeze today because the classical baseline is invalid, key inferential summaries are missing, DiTMC exposes a serious movement-tail failure that the current conclusion masks, and the final single-primary/release provenance is not coherent. These are evidence, reporting, and release gaps—not a reason to reopen model design.

```text
CURRENT_SUBMISSION_READINESS = PARTIAL

MAIN_BENCHMARK_COMPLETENESS = FAIL
MMFF_BLOCKER_STATUS = ENGINEERING_ROOT_CAUSE_IDENTIFIED__SCIENTIFIC_BASELINE_INVALID
ROOT_CAUSE_KNOWN = YES
REPAIR_SCOPE = ADD_MISSING_NUM_ATOMS_TO_RECONSTRUCTED_FROZEN_RECORD__IDENTITY_SMOKE__RERUN_FIXED_5000
SCIENTIFIC_RERUN_REQUIRED = YES
ESTIMATED_JOBS = 5000_MMFF94S_OPTIMIZATIONS__PLUS_5000_MATCHED_XTB_SINGLE_POINTS_AND_FROZEN_AGGREGATION

AVGFLOW_FINAL_READINESS = PARTIAL__SCIENTIFIC_RESULTS_COMPLETE__CLUSTERED_PAIRED_STATS_MISSING__ENERGY_REGRESSION_MUST_BE_DISCLOSED
DITMC_FINAL_READINESS = PARTIAL__SCIENTIFIC_RESULTS_COMPLETE__SEED307_CATASTROPHIC_MOVEMENT_TAIL__XTB_FAILURE_DENOMINATORS_AND_PAIRED_STATS_REQUIRED
CROSS_UPSTREAM_FINAL_READINESS = PARTIAL

STATISTICS_MAIN = PARTIAL
STATISTICS_CROSS_UPSTREAM = PARTIAL
STATISTICS_ABLATION = PASS_WITH_MINOR_REPORTING_GAP

RUNTIME_EVIDENCE = MISSING_FOR_CURRENT_FINAL_FAIR_COMPARISON
ENERGY_CLAIM_SAFETY = PASS_IF_QUALIFIED
MOVEMENT_TAIL_REPORTING = PARTIAL
REFERENCE_PROTOCOL = PASS
TO_OUR_KNOWLEDGE_FIRST_CLAIM_SAFE = NEEDS_LITERATURE_AUDIT
PROVENANCE_READINESS = PARTIAL
MD_MMPBSA_FOR_MAIN_CLAIM = OPTIONAL

P0_BLOCKERS = VALID_MMFF94S_BASELINE; DIRECT_CLUSTERED_PAIRED_FINAL_AND_CROSS_STATS; DITMC_SAFETY_CLAIM_CORRECTION; FINAL_METHOD_ROLE_AND_RELEASE_PROVENANCE_FREEZE
P1_STRONGLY_RECOMMENDED = FAIR_RUNTIME_BENCHMARK; MATCHED_GFN2_XTB_OPTIMIZATION_OR_NARROWER_CLAIM; REFERENCE_CONTEXT_ROW; MOVEMENT_EFFECT_DISTRIBUTION_REPORTING; LITERATURE_CLAIM_AUDIT
P2_OPTIONAL = MD_MMPBSA; MORE_UPSTREAMS; SIGMA_CALIBRATION_STUDY; FULL_COHORT_XTB_OPTIMIZATION; NEW_MODEL_DESIGN

NEW_MODEL_TRAINING_REQUIRED = NO
NEW_ABLATION_TRAINING_REQUIRED = NO
MMFF_REPAIR_REQUIRED = YES
RUNTIME_BENCHMARK_REQUIRED = YES
MORE_CROSS_UPSTREAM_REQUIRED = NO
MD_MMPBSA_REQUIRED = NO
```

## Minimum path to submission

1. Patch only the known MMFF record-construction bug, validate on an engineering smoke set, and rerun the fixed 5,000-member MMFF baseline with the already frozen evaluators.
2. Reaggregate existing primary/cross per-record outputs into direct molecule-cluster paired CIs, win/tie/loss, distributions, finite rates, and movement tails; do not rerun models.
3. Rewrite cross-upstream conclusions to report AvgFlow structure/energy disagreement and DiTMC seed307 instability as observed limits.
4. Run one fair runtime benchmark and either include a predeclared xTB-optimization subset or remove any physics-optimizer superiority comparison.
5. Resolve single-primary versus two-operating-point provenance, then create a clean tagged release with code/config/checkpoint/dataset/evaluator/result hashes and safe claim language.

After those steps, no new model or ablation training is needed for a defensible main-track submission package.
