# Post-multiseed resolved and open gaps

> Audit date: 2026-09-01. Evidence is restricted to completed development artifacts and protected-cohort metadata. No protected outcome was read.

## Resolved after multiseed

- **Random-seed robustness:** all six matched formulation/seed runs completed at step 17,500 on GPU and passed the integrity audit.
- **V3D direction:** Unrestricted exceeded Restricted at seeds 307, 331 and 353 by `+0.0030`, `+0.0048` and `+0.0028`; seed summaries are `0.5645 ± 0.0017` versus `0.5681 ± 0.0011`.
- **PB stability:** essentially tied (`0.93233 ± 0.00012` versus `0.93240 ± 0.00000`).
- **Reference RMSD stability:** practically tied; neither formulation demonstrates global reference recovery.
- **xTB central tendency:** Unrestricted had a lower median delta-energy at every matched seed, while both formulations retained rare positive-energy tails.
- **Unrestricted movement stability:** small bulk movement replicated. Median tau was about `0.0059–0.0061 Å`, P99 about `0.0092–0.0099 Å`, no record exceeded `0.05 Å`, and all coordinates were finite.
- **Training-budget question:** the prior 22,500-step capacity extension did not justify replacing step 17,500.

## Still open after multiseed

- An untouched, molecule-disjoint, same-upstream primary final cohort has not been identified and frozen for the current method.
- Current authoritative source, runners, configs, manifests, evaluators, environments, checkpoints and results remain uncommitted/untracked as a release unit.
- The multiseed `source_rmsd` field is actually unaligned proposal displacement; the final metric vocabulary must be corrected before a protocol freeze.
- A matched MMFF94s optimization baseline is absent for the exact final cohort and task.
- Runtime/resource reporting and molecule-level effect distributions are not yet paper-ready.
- Predictive sigma must either receive a calibration analysis or be described conservatively as an action-weighting signal, not calibrated uncertainty.

## Newly exposed after multiseed

- The formulation difference is a stable trade-off, not mere seed noise: Unrestricted improves V3D/xTB central tendency but moves farther from Source and has slightly heavier extreme movement/energy tails.
- Selecting a single formulation from development outcomes would be an unnecessary post-hoc choice. A predeclared two-operating-point presentation is scientifically cleaner.
- Existing protected assets do not supply a clean primary endpoint as-is: Formal contains TRAIN overlap; the large holdout is cross-upstream and not project-level untouched.

```text
RESOLVED_AFTER_MULTISEED = SEED_ROBUSTNESS; FORMULATION_STABILITY; SMALL_MOVEMENT_REPLICATION; V3D_PB_XTB_CONSISTENCY; STEP17500_ENDPOINT
STILL_OPEN_AFTER_MULTISEED = PRIMARY_PROTECTED_COHORT; RELEASE_PROVENANCE; METRIC_NAME_FREEZE; MATCHED_MMFF94S; RUNTIME_AND_DISTRIBUTION_REPORTING; SIGMA_CLAIM_BOUNDARY
NEWLY_EXPOSED_AFTER_MULTISEED = STABLE_PARETO_TRADEOFF; NO_JUSTIFIED_SINGLE_WINNER; EXISTING_PROTECTED_ASSETS_NOT_PRIMARY_AS_IS
```
