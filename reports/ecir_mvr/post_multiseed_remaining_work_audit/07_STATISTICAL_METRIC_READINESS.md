# Statistical and metric readiness

## Statistical readiness

The completed evidence now has the right two-level structure:

- seed-level mean and sample SD across 307/331/353;
- matched-seed formulation differences;
- molecule-cluster bootstrap for record-dependent endpoints;
- robust xTB central tendency (median and 5% trimmed mean), lower-energy fraction and explicit upper-tail counts;
- separate seed variability and within-cohort uncertainty.

For the final test, predeclare the molecule as the resampling/aggregation unit, paired estimands, confidence intervals, binary discordance counts, missing/failure handling, xTB subset rules and the separation of seed-level from molecule-level uncertainty. A generic p-value is not required; paired effect sizes and confidence intervals are more faithful to the design.

```text
STATISTICS_READY_FOR_FINAL_TEST = YES
SIGNIFICANCE_TEST_REQUIRED = NO
FINAL_ANALYSIS_PLAN_FREEZE_REQUIRED = YES
```

## Metric readiness

V3D overall/components, PB overall/components, Bond MAE, Angle MAE, Kabsch Reference RMSD, xTB median/lower fraction/trimmed mean/tails and movement quantiles have frozen implementations or completed equivalence audits.

One nomenclature defect remains: the multiseed column called `source_rmsd` is the raw, unaligned RMS displacement from Source and equals proposal movement. It must be renamed **Proposal displacement RMS (unaligned)**. **Kabsch Source RMSD** must be reserved for rigidly aligned proposal-to-source RMSD. This is a definition/label freeze, not authorization to recompute outcomes.

The final protocol must also state units, per-record versus per-molecule aggregation, validity failure denominators, Reference ensemble/correspondence policy, xTB failure policy and tau/per-atom displacement summaries.

```text
METRIC_DEFINITION_FREEZE_READY = NO
BLOCKING_METRIC_ISSUE = SOURCE_RMSD_LABEL_CONFLATES_UNALIGNED_DISPLACEMENT_WITH_KABSCH_RMSD
```
