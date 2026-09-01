# Metric and statistical completeness

## Metric coverage

```text
METRIC_COVERAGE = MINOR_GAPS
```

Existing metrics cover the central task well: local geometry validity, PoseBusters integrity, continuous Bond/Angle error, Source locality, Reference RMSD, GFN2-xTB single-point energy, movement distributions, and component-level transitions.

Material remaining gaps:

- uncertainty calibration is not characterized; NLL and noncollapse do not prove calibration;
- final candidate runtime/throughput/memory is not matched to conventional baselines;
- molecule-level win/loss and effect distributions against Source should be primary supporting plots, not only aggregate means/pass rates;
- failure rates should be stratified by size, flexibility, Source quality, and rare tail events after multiseed.

Coverage/diversity is outside scope for a one-to-one refiner. Chirality, ring, connectivity, aromaticity, and clashes are already represented in PoseBusters/Validity3D components; adding redundant aggregate metrics is unnecessary.

## Statistical completeness

```text
STATISTICAL_COMPLETENESS = INCOMPLETE_PENDING_MULTISEED_AND_UNBIASED_FINAL_EVALUATION
STATISTICAL_BLOCKERS_BEFORE_SUBMISSION = MATCHED_MULTISEED_COMPLETION; UNBIASED_FINAL_COHORT; SEED_AND_COHORT_UNCERTAINTY_SEPARATED
```

- Seed307 alone cannot distinguish a stable formulation effect from initialization/training noise.
- Three matched seeds are a reasonable minimum for this expensive paired design. Report each seed, mean ± seed SD, and the distribution of matched-seed Restricted-minus-Unrestricted differences; do not present `n=3` as high-powered inference.
- Molecule-cluster bootstrap is appropriate because each molecule contributes two records. Record-level independent bootstrap would be pseudo-replication.
- Seed variability and within-seed molecule bootstrap answer different questions and must remain separate.
- For binary endpoints, show molecule-level paired discordances as well as rate differences.
- For xTB, retain median as primary, fraction below zero, trimmed mean, quantiles, and explicit tail counts. Arithmetic mean must remain secondary because of catastrophic positive tails.
- Molecule-level aggregation is needed wherever both records from one molecule enter an endpoint. Per-record descriptive distributions may still be shown if dependence is stated.
- Multiple-comparison correction is not the main issue if endpoints and primary contrasts are preregistered. It becomes relevant only if many subgroup/metric contrasts are promoted as confirmatory.
- After formulation selection, the final unbiased cohort must not be used for further model or threshold tuning.

