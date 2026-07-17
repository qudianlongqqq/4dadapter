# MCVR Stage D Bond Oracle

Decision: **PASS**

The D0 oracle solves `J^T (J J^T + lambda I)^-1 r` globally over unique undirected bonds. Corrections are translation-free and then pass through the frozen trust and deterministic acceptance rules.

| Metric | Value |
|---|---:|
| Accepted bond relative improvement | 0.624611165102 |
| Minimal Target available improvement | 0.620714682376 |
| Target recovery upper bound | 1.006277413498 |
| RMSD delta | 0.001250363614 |
| Total validity delta | -0.457631539130 |
| Angle rate delta | -0.001458556241 |
| Ring rate delta | -0.077421497107 |
| High-flex torsion change | 0.010012537367 |
| Clean identity | 1.000000000000 |
| Numerical failure fraction | 0.000000000000 |

## Five-way comparison

| Method | Bond rate | Bond magnitude | Total validity | Angle rate | Ring rate | RMSD | Displacement |
|---|---:|---:|---:|---:|---:|---:|---:|
| upstream | 0.262031877853 | 1.247488950711 | 0.793988849439 | 0.031130727267 | 0.153503645435 | 1.321787232537 | 0.000000791880 |
| minimal_target | 0.099384844019 | 0.404742402372 | 0.328646139220 | 0.025619312335 | 0.076074067544 | 1.323031248949 | 0.008490385357 |
| bond_oracle_raw | 0.098230507993 | 0.403471679103 | 0.336204194769 | 0.029630504358 | 0.076082148328 | 1.323038823292 | 0.007196460342 |
| bond_oracle_trusted | 0.098230507993 | 0.403471679103 | 0.336204194769 | 0.029630504358 | 0.076082148328 | 1.323038823292 | 0.007196460342 |
| bond_oracle_accepted | 0.098363841333 | 0.404697937262 | 0.336357310309 | 0.029672171026 | 0.076082148328 | 1.323037596151 | 0.007184130252 |

The chemical-metric noninferiority margin is fixed at `0.005`; severe clash and chirality must not increase.

## Gate

| Condition | Result |
|---|---|
| 01_accepted_bond_relative_improvement_ge_25pct | PASS |
| 02_target_recovery_ge_40pct | PASS |
| 03_rmsd_delta_le_0p003 | PASS |
| 04_angle_not_clearly_worse | PASS |
| 05_ring_not_clearly_worse | PASS |
| 06_clash_not_worse | PASS |
| 07_chirality_not_worse | PASS |
| 08_high_flex_validity_improves | PASS |
| 09_high_flex_torsion_controlled | PASS |
| 10_clean_identity_ge_90pct | PASS |
| 11_numerical_failure_lt_1pct | PASS |

No training, test evaluation, seed43/44, 20k, or 100k run was performed.
Stage D1 is permitted only when this decision is PASS.
