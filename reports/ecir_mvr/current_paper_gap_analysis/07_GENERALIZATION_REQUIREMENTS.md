# Generalization requirements

## Current boundary

`ARTIFACT_FACT`: complete evidence supports molecule-disjoint DEV performance for unseen molecules from the same ETFlow Source distribution. Cross-upstream behavior and protected outcomes are unknown.

## Claim levels

### CLAIM_LEVEL_A — ETFlow post-generation refinement

Minimum evidence:

1. matched multiseed completion and integrity pass;
2. exact final formulation frozen before outcome read;
3. unbiased ETFlow molecule-disjoint test/authorized protected cohort;
4. Source and matched conventional refinement baselines;
5. main validity, geometry, locality, xTB, runtime, and tail metrics.

Cross-upstream evaluation is not mandatory for this explicitly ETFlow-scoped claim.

### CLAIM_LEVEL_B — transferable refinement across upstream generators

Minimum evidence:

All Level A evidence, plus zero-shot application of the same frozen checkpoint and inference constants to at least two materially different Source generators such as DiTMC and AvgFlow, with identity-matched records and no upstream-ID feature. Report per-upstream effects and failures; pooled-only reporting is insufficient.

### CLAIM_LEVEL_C — general conformer refinement framework

Minimum evidence:

All Level B evidence, plus broader chemical/source-quality coverage, a protected or genuinely untouched large cohort, strong conventional and learned baselines, subgroup/failure analysis, and evidence that fixed topology plus Bond/Angle-only actions remain useful outside ETFlow-like local defects.

```text
GENERALIZATION_COMPLETENESS = ADEQUATE_ONLY_FOR_DEVELOPMENT_LEVEL_CLAIM_A
CROSS_UPSTREAM_REQUIRED_FOR_LEVEL_A = NO
CROSS_UPSTREAM_REQUIRED_FOR_LEVEL_B_OR_C = YES
```

The paper should choose a claim level based on available evidence rather than expand the experiments to rescue an overbroad title.

