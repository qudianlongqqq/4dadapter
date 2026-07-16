# MCVR Medium Bond Metric Audit

Decision: **METRIC_IMPLEMENTATION_CORRECT**

## Exact recalculation

| Quantity | Value |
|---|---:|
| Upstream bond outlier rate | 0.262031877853 |
| Accepted-model bond outlier rate | 0.235871849317 |
| Absolute delta (model - upstream) | -0.026160028536 |
| Relative improvement | 0.099835290081 |
| Relative improvement percent | 9.9835290081% |

The formal definition is `(upstream - accepted_model) / upstream`; the unrounded float is compared directly with `0.10`.

## Aggregation contract

| Level | Rule |
|---|---|
| bond_within_record | each unique undirected bond has equal weight within its record |
| record_within_molecule | arithmetic mean; one or two validation records per molecule |
| molecule_within_all | arithmetic mean; each of 500 molecules has equal weight |
| bootstrap | paired molecule resampling with replacement |
| missing | none; paired pivots drop missing molecules fail-closed in this audit |
| identity_and_fallback | included unchanged; target fallback affects target diagnostics only |
| threshold_equality | distance > 0.0 is outlier; exact lower/upper equality is normal |
| ring_bonds | unique bonds appear once in bond_outlier_rate; ring rate is a subset diagnostic, not an added duplicate |
| bond_count_weighting | no cross-molecule bond-count weighting in the formal Gate |
| floating_comparison | unrounded float64 relative improvement compared directly with 0.10 |

Record-equal relative improvement is `0.100430902062`; it is not the preregistered Gate aggregation.

## Implementation checks

| Check | Result |
|---|---|
| formula_exact | PASS |
| matches_gate_value | PASS |
| matches_source_summary_upstream | PASS |
| matches_source_summary_model | PASS |
| all_500_molecules_present | PASS |
| no_missing_metric_values | PASS |
| paired_molecule_rows_complete | PASS |
| selected_checkpoint_identity | PASS |
| test_records_zero | PASS |
| protected_file_unchanged | PASS |

Exact threshold equality is normal because the implementation uses `distance > 0.0`.
Ring bonds are present once in the unique-bond list; the ring metric is a subset and is not added to the ordinary bond count.
The paired bootstrap resamples molecules, not records or bonds. No validation value is missing.
