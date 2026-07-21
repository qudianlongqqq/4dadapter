# MCVR external refinement FULL10K

Status: `MCVR_EXTERNAL_REFINEMENT_FULL10K_COMPLETED`

| Method | accepted | weighted_bac_delta | bond_delta | angle_delta | active_angle_delta | ring_delta | clash_delta | chirality_preserved | mean_displacement | rmsd | MAT_P | MAT_R | COV_P | COV_R | conformer_diversity | duplicate_conformer_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FROZEN_D1 | 0.9898 | -0.018951378 | -0.0079371342 | -0.00010625842 | -0.00010625842 | -0.00071669825 | -1.4730475e-09 | 1 | 0.00024253871 | 1.3300683 | 1.3300683 | 2.0556104 | 0.4911 | 0.13908534 | 2.3884046 | 0.0028 |
| GFN2_XTB | 0.9989 | -0.1834679 | -0.097053083 | -0.0030739586 | -0.0030739586 | -0.012145697 | -2.2031385e-07 | 1 | 0.27327426 | 1.2918229 | 1.2918229 | 2.0484351 | 0.5173 | 0.14453438 | 2.3770978 | 0.0042 |
| MATCHED_D1_12P5K | 0.9828 | -0.18533894 | -0.095874859 | -0.0011171409 | -0.0011171409 | -0.014139173 | -3.3256038e-11 | 1 | 0.0021724476 | 1.3300795 | 1.3300795 | 2.0555986 | 0.4912 | 0.13907423 | 2.3883625 | 0.0028 |
| MMFF94S | 0.9951 | 0.056056931 | 0.03397041 | 0.0046508518 | 0.0046508518 | 0.021105285 | -1.6593984e-07 | 1 | 0.8003517 | 1.3823086 | 1.3823086 | 2.0841156 | 0.4609 | 0.1313469 | 2.3226845 | 0.006 |
| RAW | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1.330059 | 1.330059 | 2.0556088 | 0.4911 | 0.13907868 | 2.3884066 | 0.0028 |
| V5_B | 0.9938 | -0.025686073 | -0.0075727843 | -9.0221271e-05 | -9.0221271e-05 | -0.00068191487 | -1.0092001e-08 | 1 | 0.0003446533 | 1.3300735 | 1.3300735 | 2.0556147 | 0.4911 | 0.13907423 | 2.3884103 | 0.0028 |
| V7 | 0.9918 | -0.020691975 | -0.0079819111 | -0.00011846062 | -0.00011846062 | -0.001074258 | -9.2172237e-09 | 1 | 0.00028934278 | 1.3300694 | 1.3300694 | 2.0556111 | 0.4911 | 0.13908534 | 2.3884027 | 0.0028 |
| V8_FULL_12P5K | 0.9862 | -0.20140865 | -0.093121296 | -0.0044310814 | -0.0044310814 | -0.019807461 | -3.7915966e-09 | 1 | 0.0034744402 | 1.3302337 | 1.3302337 | 2.0556156 | 0.491 | 0.13909685 | 2.3882748 | 0.0028 |

## Frozen questions

### 1_mcvr_vs_mmff94s_composite_geometry

Yes. V8 has lower weighted BAC (-0.20140865 vs 0.056056931), lower angle/ring deltas, and lower RMSD (1.3302337 vs 1.3823086).

### 2_mcvr_faster_than_gfn2_xtb

Yes on observed wall time: V8 was 12.70x faster per record than the two-worker xTB run. Hardware differs, so this is an operational comparison, not algorithm-normalized CPU timing.

### 3_xtb_physical_optimization_and_movement

GFN2-xTB lowered its own native energy on successful records and moved atoms much more (0.27327426 A vs V8 0.0034744402 A). Native xTB energy is not compared numerically with MMFF or neural methods.

### 4_mmff_xtb_effect_on_rmsd_mat_cov_diversity

MMFF worsened RMSD versus Raw (1.3823086 vs 1.330059) and increased duplicate rate to 0.006; xTB RMSD is 1.2918229, with diversity 2.3770978 and duplicate rate 0.0042.

### 5_mcvr_small_movement_global_conformation

Yes. V8 mean displacement is 0.0034744402 A while MAT-P/MAT-R are 1.3302337/2.0556156 and diversity remains 2.3882748.

### 6_failure_and_coverage

MMFF success/fallback = 9951/10000 and 49/10000; xTB = 9989/10000 and 11/10000. No record was dropped.

### 7_clash_power

Only 20 natural records were applicable for active clash, so clash inference is explicitly treated as low-power when its CI crosses zero.

### 8_mcvr_gain_concentration

Yes. V8's main unified-evaluator gains remain weighted BAC (-0.20140865), angle (-0.0044310814), and ring (-0.019807461), with only 0.0034744402 A mean movement.

Native MMFF94s and GFN2-xTB energies are retained only within their own method and are never cross-compared.

All external failures use all-record deployment semantics with bitwise Source fallback.
