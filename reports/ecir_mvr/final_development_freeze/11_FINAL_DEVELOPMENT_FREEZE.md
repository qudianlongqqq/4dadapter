# Final development freeze

SIXS model development is closed with two predeclared Pareto operating points: Restricted A and Unrestricted B, each at step 17,500 for seeds 307/331/353. No further architecture, objective, tau, Reliability, Adaptive BA, sigma, training-step or seed search is authorized by this freeze.

The scientific definitions, corrected RMSD vocabulary, V3D/PoseBusters/xTB protocols, molecule-cluster paired analysis, metric hierarchy, paper-claim boundary, sigma wording, active code/evaluator set, environments and six checkpoints are now immutable under Git tag `final-development-freeze-step1` plus SHA256-bound external artifacts.

Historical seed307 executed-runner provenance remains PARTIAL. Forward reproducibility is PASS because future work must use the tagged snapshot and frozen hashes.

```text
STEP_1_STATUS = PASS
OPERATING_POINTS_FROZEN = YES
SCIENTIFIC_DEFINITION_FROZEN = YES
METRIC_VOCABULARY_FROZEN = YES
XTB_PROTOCOL_FROZEN = YES
STATISTICAL_PLAN_FROZEN = YES
PRIMARY_SECONDARY_METRICS_FROZEN = YES
CLAIM_VOCABULARY_FROZEN = YES
ACTIVE_CODE_IDENTIFIED = YES
FORWARD_PROVENANCE_FROZEN = YES
CHECKPOINTS_FROZEN = YES
ENVIRONMENT_FROZEN = YES
PROTECTED_OUTCOME_READ = NO
NO_NEW_SCIENTIFIC_EXECUTION = YES
READY_FOR_STEP_2_FINAL_COHORT_IDENTITY_AUDIT = YES
NEXT_STEP = STEP 2 — PRIMARY FINAL COHORT IDENTITY AND ZERO-OVERLAP AUDIT
```

STEP 2 was not started.
