# Molecule-level and statistical plan

## Required paired effects

For Restricted–Source, Unrestricted–Source and MMFF94s–Source, report molecule-level:

- V3D rate change over that molecule's records and win/tie/loss;
- PB rate change and paired discordances;
- median/mean record delta-energy with the global robust xTB summaries;
- raw Source displacement and Reference RMSD change;
- continuous Bond/Angle error changes.

Restricted–Unrestricted remains a planned Pareto contrast, not a protected-outcome selection mechanism. Distributions must show whether aggregate gains are broad or driven by a small subset.

## Statistical levels

- **Seed level:** show every development seed, mean ± sample SD, and matched-seed formulation effects. Do not attach strong inferential claims to `n=3`.
- **Molecule level:** paired molecule-cluster bootstrap confidence intervals and effect distributions. This is the principal final-test uncertainty level.
- **Record level:** descriptive histograms/quantiles only; two records from one molecule are not independent observations.
- **xTB:** median primary with a molecule-cluster bootstrap CI, fraction below zero, trimmed mean and tail counts. Arithmetic mean secondary.
- **Binary validity:** rate difference plus molecule-level win/tie/loss/discordances.
- **p-values:** not mechanically required when primary contrasts, effects and CIs are preregistered.

## Stratification

- **MUST HAVE:** Source-quality/initial-failure stratum (initial V3D/PB pass state and Source local defect), because the method targets correctable local defects.
- **HIGH VALUE:** rotatable-bond/flexibility and heavy-atom/size strata, kept descriptive unless powered and preregistered.
- **OPTIONAL:** ring-containing molecules and rare tail case studies.

Limit subgroup count, publish denominators and treat small strata as exploratory. Existing seed307 quintiles are development evidence, not confirmatory subgroup proof.

```text
MOLECULE_LEVEL_ANALYSIS_STATUS = PARTIAL__FINAL_PAIRED_DISTRIBUTIONS_MISSING
STATISTICAL_STATUS = DESIGN_READY__FINAL_COHORT_ANALYSIS_NOT_RUN
PSEUDOREPLICATION_CONTROL = MOLECULE_CLUSTERING_REQUIRED
```
