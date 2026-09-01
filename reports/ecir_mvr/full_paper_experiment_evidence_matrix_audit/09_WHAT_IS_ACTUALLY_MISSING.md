# What is actually missing

## Paper blockers

1. A clean, outcome-unseen, molecule-level zero-TRAIN/DEV-overlap same-ETFlow primary final cohort and one-shot result.
2. Exact-record MMFF94s geometry optimization evaluated with the identical validity, RMSD, xTB and denominator protocol.
3. Final-model inference/runtime/resource benchmark with matched MMFF94s timing.
4. Frozen RMSD vocabulary and final molecule-cluster paired analysis, including xTB median CIs and win/tie/loss distributions.
5. Immutable release provenance for both operating points and the evaluator/environment dependency closure.

## Important additions

- Full V3D and PB component reporting on final data.
- Source-quality/initial-failure subgroup and concise size/flexibility diagnostics.
- Runtime for xTB single point.
- Low-cost within-molecule diversity-preservation diagnostic.
- Conservative sigma wording or a calibration figure if calibrated uncertainty is retained as a claim.

## Optional

- GFN2-xTB geometry optimization quality/runtime on a deterministic matched subset.
- DiTMC secondary cross-upstream holdout.
- Task-compatible learned refiner if a truly fair implementation exists.
- Qualitative molecule examples.

## Do not do

- Standard COV/AMR as mandatory endpoints.
- A new downstream property/docking study.
- Cross-dataset expansion for the current ETFlow-scoped claim.
- More architecture design, training steps, seeds or hyperparameter sweeps.
- A broad subgroup fishing exercise.

```text
MISSING_PRIMARY_METRICS = NONE_IN_DEFINITION__OUTCOME_UNSEEN_VALUES_NOT_YET_OBTAINED
MISSING_SECONDARY_METRICS = KABSCH_SOURCE_RMSD_FINAL_VALUE; MMFF_SUCCESS_AND_RUNTIME; LEARNED_RUNTIME_THROUGHPUT_MEMORY; MOLECULE_LEVEL_PAIRED_DISTRIBUTIONS
```
