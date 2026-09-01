# Frozen statistical analysis plan

> This plan is outcome-blind with respect to the future primary final cohort. It separates seed variability, molecule sampling uncertainty and record-level description.

## Statistical levels

### Seed level

- Development robustness uses seeds 307, 331 and 353.
- For each operating point report every seed estimate, arithmetic mean across the three seed estimates and sample SD (`ddof=1`).
- Report matched-seed Restricted-minus-Unrestricted differences for every endpoint.
- Do not pool all records across seeds and present them as independent observations.

### Molecule level — primary paired unit

One molecule is one paired resampling cluster. All methods must use the same frozen molecule and record identities.

- Binary record endpoints (V3D overall/components and PB overall/components): per-molecule success fraction = mean of the molecule's record-level 0/1 values.
- Continuous record-mean endpoints (Bond MAE, Angle-cosine MAE, Raw Source Displacement RMS, Kabsch-aligned Source RMSD, Reference RMSD and implementation diagnostics): per-molecule arithmetic mean.
- xTB endpoint location remains the median across all successful matched records, with the molecule as the bootstrap cluster and all selected records retained inside each resampled molecule. For molecule-level win/tie/loss only, aggregate each method's records within molecule by the median before comparing.
- Movement/tau distributions are descriptive diagnostics unless explicitly identified as a source-preservation endpoint.

### Record level

Record-level rates and distributions may be shown descriptively. Records from the same molecule are not treated as independent scientific observations and are never resampled independently for the primary confidence interval.

## Predeclared final contrasts

The future final cohort must report, on the same matched identities:

1. Source vs Restricted;
2. Source vs Unrestricted;
3. Source vs MMFF94s;
4. Restricted vs Unrestricted as a predeclared trade-off contrast, not an outcome-dependent winner-selection rule.

For each suitable endpoint report the paired effect, central tendency, molecule-level win/tie/loss fractions and a molecule-cluster bootstrap 95% interval.

Direction conventions:

- V3D/PB: higher is better; candidate minus baseline positive is a win.
- Bond/Angle MAE and Reference RMSD: lower is better; candidate minus baseline negative is a win.
- xTB delta-E versus Source: negative means lower candidate energy. For candidate-to-candidate contrasts, lower delta-E is better.
- Raw Source Displacement RMS is intervention magnitude/source preservation, not a generic quality win against unchanged Source.
- Exact equality is a tie. No post-outcome numerical tolerance may be introduced.

## Bootstrap

```text
RESAMPLING_UNIT = MOLECULE
RESAMPLES = 10000
RNG_SEED = 20260901
INTERVAL = PERCENTILE_95
QUANTILES = 0.025;0.975
RECORD_WISE_BOOTSTRAP = PROHIBITED
```

For each bootstrap draw, sample molecules with replacement, retain all their records, and recompute the complete endpoint. This applies to rates, means, medians and paired differences; do not replace the xTB median interval with a mean interval.

## Missingness and failures

- The eligible denominator and record/molecule identities are frozen before outcome access.
- V3D/PB missing or evaluator-invalid records count as failures and remain in denominators.
- Continuous/RMSD/xTB engineering failures are never silently imputed or removed. Report successful matched pairs and all failure categories against the full frozen denominator. Primary continuous summaries use finite matched pairs; any missingness triggers an explicit sensitivity/accounting table.
- No outcome-dependent exclusion, formulation-specific cohort, checkpoint selection, early stopping or metric redefinition is permitted.

```text
STATISTICAL_PLAN_FROZEN = YES
MOLECULE_LEVEL_UNIT_FROZEN = YES
FINAL_PAIRED_ANALYSIS_FROZEN = YES
```
