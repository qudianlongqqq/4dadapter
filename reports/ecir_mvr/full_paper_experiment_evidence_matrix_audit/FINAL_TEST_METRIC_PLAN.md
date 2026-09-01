# One-shot final-test metric plan

> This is a protocol definition only. No protected outcome was accessed and no evaluation was started.

## Primary endpoints

1. **Validity3D overall**: paired rate difference versus Source for Restricted and Unrestricted, with the exact MMFF94s contrast on the same records. Report all four V3D components alongside it so the conjunction is interpretable.
2. **GFN2-xTB single-point median delta-energy** relative to matched Source. Report paired formulation/MMFF contrasts using the same xTB version and failure policy.

These are two domain-specific primary endpoints—structural validity and energetic central tendency—not a large undifferentiated metric family. Predeclare the decision rule; do not select one formulation after opening outcomes.

## Secondary endpoints

- PoseBusters overall and every selected component.
- Bond raw MAE in Å and cosine-Angle raw MAE (dimensionless).
- Raw Source displacement RMS in Å.
- Reference RMSD: fixed-order, explicit-all-atom, proper-rotation Kabsch, minimum over frozen Reference ensemble, Å.
- xTB fraction deltaE below zero and 5% trimmed mean.
- xTB P90/P95/P99 and counts above +25/+50/+100 kcal/mol.
- MMFF94s optimization success/failure rate and matched quality endpoints.

## Diagnostic endpoints

- Kabsch-aligned Source RMSD in Å, explicitly distinct from raw displacement.
- Tau median/P90/P95/P99/P99.5/P99.9/max and threshold fractions.
- Per-atom displacement quantiles/max, finite-coordinate rate and Restricted cap activity.
- V3D invalid primitive counts and PB transition/discordance tables.
- Internal post objective and direction improvement, labeled implementation diagnostics rather than external performance.
- Optional within-molecule pairwise-RMSD preservation diagnostic.

## Statistical unit and contrasts

- Primary unit: molecule; two records per molecule remain clustered.
- Contrasts: Restricted–Source, Unrestricted–Source, MMFF94s–Source, and Restricted–Unrestricted.
- Report record-level rates descriptively, molecule-cluster paired bootstrap CIs, and binary molecule-level win/tie/loss or discordance summaries.
- Keep seed variation from development separate from final-cohort sampling uncertainty.
- For xTB, bootstrap the median and paired median differences; do not substitute an arithmetic-mean CI.
- Freeze missingness, invalid SDF, MMFF failure and xTB failure denominators before outcome access.

## Fixed method identities

- Source: unchanged input coordinates.
- Restricted and Unrestricted: both predeclared step-17,500 operating points; no protected-outcome winner selection.
- MMFF94s: same Source coordinates, same record IDs, fixed implementation/termination/thread policy.
- GFN2-xTB: version 6.7.1, GFN2, single point, no geometry optimization, kcal/mol delta relative to the matched Source.

```text
PRIMARY_METRICS = VALIDITY3D_OVERALL; GFN2_XTB_MEDIAN_DELTAE
SECONDARY_METRICS = V3D_COMPONENTS; PB_OVERALL_AND_COMPONENTS; BOND_MAE; ANGLE_MAE; RAW_SOURCE_DISPLACEMENT; REFERENCE_RMSD; XTB_LOWER_FRACTION_TRIMMED_MEAN_AND_TAILS; MMFF_SUCCESS
DIAGNOSTIC_METRICS = KABSCH_SOURCE_RMSD; MOVEMENT_DISTRIBUTIONS; INTERNAL_OBJECTIVE; DIRECTION_IMPROVEMENT; OPTIONAL_DIVERSITY_PRESERVATION
```
