# Final metric hierarchy

This hierarchy is frozen before the future primary final-cohort outcome is opened.

## Primary

### Structural

- **Validity3D overall** — per-record four-component conjunction, analyzed with the molecule as the paired statistical unit.

### Physical

- **GFN2-xTB median delta-E** relative to matched Source — xTB 6.7.1 single-point, no geometry optimization.

The two primary endpoints cover different domains. Neither may be replaced by a more favorable component or summary after outcome access.

## Source preservation

- **Raw Source Displacement RMS** in Å. This is explicitly an intervention-magnitude/source-preservation endpoint, not an aligned RMSD and not a generic improvement endpoint.

## Secondary

- PoseBusters overall;
- Reference RMSD;
- Bond raw MAE in Å;
- Angle-cosine raw MAE (dimensionless);
- xTB fraction `delta-E < 0`;
- xTB 5% trimmed mean;
- xTB arithmetic mean;
- matched MMFF94s optimization success/failure rate when STEP 2+ authorizes that baseline.

## Diagnostic/supplementary

- all V3D components;
- all selected PoseBusters components;
- xTB P90/P95/P99 and `>+25/>+50/>+100 kcal/mol` counts;
- Kabsch-aligned Source RMSD;
- tau and per-atom movement distributions, finite-coordinate rate and Restricted cap activity;
- internal post objective and direction improvement;
- Reliability, Adaptive BA and predicted heteroscedastic-scale distributions;
- transition/discordance and molecule-level effect-distribution tables.

## Claim scope

The hierarchy supports only `ETFlow-source fixed-topology one-to-one local conformer refinement`. Reference RMSD is included to show global-conformer relation/source preservation; it does not turn the task into global conformer recovery. Predicted sigma is a heteroscedastic scale, not calibrated uncertainty.

```text
PRIMARY_METRICS = VALIDITY3D_OVERALL; GFN2_XTB_MEDIAN_DELTAE
SOURCE_PRESERVATION_METRIC = RAW_SOURCE_DISPLACEMENT_RMS
SECONDARY_METRICS = POSEBUSTERS_OVERALL; REFERENCE_RMSD; BOND_RAW_MAE; ANGLE_COSINE_RAW_MAE; XTB_LOWER_ENERGY_FRACTION; XTB_5PCT_TRIMMED_MEAN; XTB_ARITHMETIC_MEAN; MMFF94S_SUCCESS_RATE
DIAGNOSTIC_METRICS = V3D_COMPONENTS; POSEBUSTERS_COMPONENTS; XTB_TAILS; KABSCH_ALIGNED_SOURCE_RMSD; MOVEMENT_DISTRIBUTIONS; INTERNAL_POST; DIRECTION_IMPROVEMENT; RELIABILITY_BA_SIGMA_DISTRIBUTIONS; TRANSITIONS_AND_EFFECT_DISTRIBUTIONS
PRIMARY_SECONDARY_METRICS_FROZEN = YES
```
