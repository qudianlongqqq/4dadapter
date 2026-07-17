# MCVR Stage D Pilot Report

Decision: **STAGE_D_NO_ADDED_VALUE**

Gate result: **17/20**. D1-B clears the absolute 12% bond threshold and improves total validity over V4, but does not meet the registered recovery floor and slightly worsens the V4 angle and ring rates.

D1-A selected step 2000; D1-B selected step 2000.

| Metric | Value |
|---|---:|
| D1-B bond relative improvement | 0.122723971416 |
| Model-to-target recovery | 0.197713981803 |
| D1-B validity delta vs V4 | -0.034822519482 |
| RMSD mean delta vs upstream | 0.000924050953 |
| Newly broken bonds D1-B / V4 | 177 / 192 |
| Cancellation ratio | 0.110143123833 |
| Solver failure fraction | 0.000000000000 |
| Angle-rate delta vs V4 | 0.000822072097 |
| Ring-rate delta vs V4 | 0.001092642277 |

## Proposal stages

| Method | Bond rate | Bond magnitude | Total validity | RMSD | Displacement |
|---|---:|---:|---:|---:|---:|
| Upstream | 0.262031877853 | 1.247488950711 | 0.793988849439 | 1.321787232537 | 0.000000791880 |
| V4 selected | 0.235871849317 | 1.045735421844 | 0.709340872537 | 1.322532642487 | 0.001952111438 |
| D1-A accepted | 0.230071955249 | 0.933301738482 | 0.679145358991 | 1.322632104728 | 0.002272030687 |
| D1-B raw | 0.279717920728 | 1.049819574781 | 0.755682144975 | 1.325351388074 | 0.009155865789 |
| D1-B trust-clipped | 0.279717920728 | 1.049819574781 | 0.755682144975 | 1.325351388074 | 0.009155865789 |
| D1-B safety-gated | 0.235161671676 | 0.908803501019 | 0.678721473491 | 1.322914387368 | 0.002942466177 |
| D1-B accepted | 0.229874285165 | 0.922581976359 | 0.674518353055 | 1.322711283490 | 0.002410810924 |
| Minimal Target | 0.099384844019 | 0.404742402372 | 0.328646139220 | 1.323031248949 | 0.008490385357 |

The raw and trust-clipped rows are identical because the selected validation trajectories did not require trust clipping. Learned safety attenuation and deterministic acceptance produced most of the deployed-stage difference. Cartesian and bond effects had mean bond-subspace cosine `0.615783922064`; stagewise cancellation was `0.004285714286` and the transition cancellation ratio was `0.110143123833`.

## Training resources

| Run | Wall time (s) | Active optimizer (s) | Validation (s) | Checkpoint I/O (s) | Steps/s | Samples/s | GPU util mean | Peak card MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1-A | 850.203 | 413.203 | 260.453 | 0.545 | 12.101 | 96.756 | 13.248% | 3672 |
| D1-B | 834.609 | 405.690 | 255.298 | 0.733 | 12.325 | 98.548 | 12.347% | 3413 |

Both runs completed all 5,000 optimizer steps from step 0. Every registered checkpoint and learning rate is recorded in `checkpoint_comparison.csv` and the two `lr_history.csv` files; every original and bond-specific loss is recorded in each run's `metrics.csv`.

## Gate

| Condition | Result |
|---|---|
| 01_bond_relative_improvement_ge_12pct | PASS |
| 02_bond_vs_v4_paired_ci_improves | PASS |
| 03_bond_magnitude_vs_v4_not_worse | PASS |
| 04_model_to_target_recovery_ge_0p22 | FAIL |
| 05_total_validity_vs_v4_not_worse | PASS |
| 06_angle_vs_v4_not_worse | FAIL |
| 07_ring_vs_v4_not_worse | FAIL |
| 08_newly_broken_bonds_not_above_v4 | PASS |
| 09_cancellation_ratio_le_20pct | PASS |
| 10_rmsd_mean_delta_le_0p003 | PASS |
| 11_rmsd_ci_upper_le_0p005 | PASS |
| 12_mat_p_mat_r_limits | PASS |
| 13_cov_p_cov_r_no_material_drop | PASS |
| 14_high_flex_validity_improves | PASS |
| 15_high_flex_torsion_controlled | PASS |
| 16_clean_identity_ge_90pct | PASS |
| 17_clash_chirality_not_worse | PASS |
| 18_unseen_validity_accuracy_pass | PASS |
| 19_improvement_not_single_source | PASS |
| 20_solver_numerical_failure_lt_1pct | PASS |

No test, seed43/44, 20k, or 100k execution occurred.

## Verification

Targeted tests: `24 passed`.

Full repository tests: `369 passed`, `0 failed` (baseline: 353 passed).

Experimental test records read: `0`.
