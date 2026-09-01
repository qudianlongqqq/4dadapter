# Reproducibility closure

## Forward reproducibility

From this freeze onward, an independent researcher can identify:

- code and operating-point definitions: `01`, `02` and `06`;
- exact configs and TRAIN/DEV manifest identities: `01` and `06`;
- all six authoritative endpoint checkpoints and load status: `08`;
- V3D, PoseBusters, RMSD and xTB evaluators/definitions: `03` and `06`;
- metric hierarchy and outcome-blind molecule-level statistical plan: `04` and `05`;
- environment and external dependency identities: `07`, `ENVIRONMENT_LOCK_MINIMAL.txt` and `06`;
- immutable repository snapshot: Git tag `final-development-freeze-step1`;
- artifact integrity: `SHA256SUMS.txt`.

No large checkpoint or external reference cache is stored in Git; each is bound by stable path and SHA256. Reproduction therefore also requires access to those hash-matching external files.

```text
FORWARD_REPRODUCIBILITY = PASS
REPRODUCIBILITY_AFTER_FREEZE = PASS
```

## Historical reproducibility

The exact seed307 executed runner source was not preserved, although its executed SHA was recorded and its current runner differs in bytes. Core model modules, configs, checkpoints and completed results are hash-bound, so this is a provenance limitation rather than evidence of invalid scientific results.

```text
HISTORICAL_SEED307_RUNNER_PROVENANCE = PARTIAL
HISTORICAL_REPRODUCIBILITY = PARTIAL
```

Forward PASS does not overwrite historical PARTIAL.

## Scope and guards

This closure performed checkpoint metadata load tests only. It did not run model inference, coordinate generation, V3D, PoseBusters, xTB, MMFF94s, training, protected evaluation or final-cohort identity work.

```text
PROTECTED_OUTCOME_READ = NO
NO_NEW_SCIENTIFIC_EXECUTION = YES
STEP_2_STARTED = NO
```
