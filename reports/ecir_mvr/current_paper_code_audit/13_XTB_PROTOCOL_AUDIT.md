# xTB protocol audit

## Verified protocol

- Executable/version: xTB 6.7.1.
- Theory level: GFN2-xTB (`--gfn 2`).
- Operation: frozen-coordinate single-point energy only.
- Geometry optimization: NO. The command rejects `--opt`, `--ohess`, and `--md`; every result records `geometry_optimization_performed=False`.
- Native energy: Hartree.
- Conversion: `deltaE_kcal_mol=(E_proposal-E_source)*627.509474`.
- Timeout: 180 s.
- Execution: four workers, one thread per xTB process for the completed restricted diagnostic.

Evidence: `scripts/run_sixs_j1r1_full_joint_xtb_dev.py:61-65,398-464,565-591,701-748`; `XTB_PROTOCOL.json`; `XTB_FINAL_STATUS.json`.

## Failure and nonfinite handling

Failures are classified as timeout, nonzero exit, nonfinite energy, parse failure, or other and written to a failure table. Numerical summaries use paired rows with successful finite energies. This is a protocol-based engineering-validity filter, not a filter on favorable/unfavorable deltaE.

For completed seed307 Restricted, Source, comparator, and proposal each have 5,000/5,000 successes. Unrestricted also has 5,000 successful finite rows. Therefore no record was excluded from the authoritative seed307 comparisons.

The multiseed finalizer is coded to require `all_xtb_success_5000` for integrity. Its outputs are incomplete and were not audited as results.

## Robust statistics

The implementation calculates median, fraction below zero, 5% and 10% trimmed means, quantiles, positive tails, mean, and molecule-cluster bootstrap. The project designates median as the primary location statistic and mean as secondary.

```text
XTB_PROTOCOL_STATUS = PASS_FOR_COMPLETED_SEED307
XTB_VERSION = 6.7.1
GFN_LEVEL = GFN2_XTB
GEOMETRY_OPTIMIZATION = NO
OUTCOME_DEPENDENT_EXCLUSION = NO_EVIDENCE
ACTUAL_SEED307_EXCLUSIONS = 0
MULTISEED_XTB_STATUS = INCOMPLETE
```


