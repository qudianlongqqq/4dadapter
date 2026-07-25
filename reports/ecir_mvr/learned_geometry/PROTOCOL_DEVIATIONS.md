# LSGO protocol deviations and execution incidents

1. The first `internal-finalize` execution completed all likelihood, Reference/Source and DEV_A three-budget diagnostics. During DEV_B it did not restrict the helper call to the already frozen DEV_A-selected budget and therefore began redundant calculations for all three budgets. The process ended before any DEV_B file or selection report was written. No PB/xTB evaluation was unlocked or run, and no result from the incomplete call was used.
2. The runner was amended to SHA/schema/count-check and reuse the complete DEV_A file and to pass exactly the frozen selected budget into the one-shot DEV_B helper. This is an execution/resume correction only: models, checkpoints, data, seeds, objectives, solver, thresholds, budget candidates, DEV_A selection rule and the selected budget are unchanged. There was no retraining or result-driven parameter adjustment.

