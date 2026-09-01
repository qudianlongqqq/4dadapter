# COV/AMR and diversity applicability audit

## Task identity

SIXS consumes an upstream Source conformer and produces exactly one locally refined counterpart with fixed topology and record identity. The DEV cohort happens to contain two Source records per molecule, but SIXS does not sample a new diverse conformer ensemble.

Standard conformer-generation COV-P/COV-R and AMR-P/AMR-R ask set-generation questions. Applying them to the two inherited Source records would primarily measure ETFlow's sampling and Reference-set match, not SIXS's local-refinement capability. A matched before/after calculation could be defined, but it would be an adapted secondary diagnostic and should not be presented as a standard generation benchmark.

If ever computed, the only fair protocol is: identical Source/refined record sets; a frozen Reference ensemble; fixed all-atom correspondence/Kabsch convention; predeclared RMSD threshold; both precision and recall directions; molecule-level aggregation; and Source-versus-refined paired differences. It must be labeled adapted coverage preservation, not new conformer generation.

## Better-matched diversity question

The scientifically relevant check is whether one-to-one refinement collapses the diversity already supplied upstream. With two records per molecule, a low-cost diagnostic can compare within-molecule pairwise Kabsch RMSD before and after refinement and verify nearest-Source identity preservation. It has limited resolution with only one pair per molecule, but directly targets collapse.

```text
COV_STATUS = NOT_REQUIRED
AMR_STATUS = NOT_REQUIRED
COV_AMR_APPLICABLE_TO_CURRENT_TASK = NOT_APPLICABLE_AS_STANDARD_PRIMARY__VALID_ONLY_WITH_ADAPTED_SECONDARY_DEFINITION
COV_AMR_NOT_REQUIRED_FOR_CURRENT_TASK = YES
DIVERSITY_PRESERVATION_EVIDENCE_REQUIRED = SHOULD_HAVE_LOW_COST_DIAGNOSTIC__NOT_A_PAPER_BLOCKER
```
