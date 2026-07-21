# V8 Full 12.5K unified formal-large baseline comparison

All methods use the same ordered 10K validation records and frozen evaluator. Non-applicable records are excluded from metric/cohort denominators.

| Method | Acceptance | Weighted BAC | Bond | Angle | Active angle | Clash | Ring | Chirality | Mean disp. | Max disp. | Solver fail | RMSD | MAT-P | MAT-R | COV-P | COV-R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1.330059 | 1.330059 | 2.0556088 | 0.4911 | 0.13907868 |
| D1 | 0.9898 | -0.018951378 | -0.0079371342 | -0.00010625842 | -0.00010625842 | -1.4730475e-09 | -0.00071669825 | 1 | 0.00024253871 | 0.000817971 | 0 | 1.3300683 | 1.3300683 | 2.0556104 | 0.4911 | 0.13908534 |
| V5-B | 0.9938 | -0.025686073 | -0.0075727843 | -9.0221271e-05 | -9.0221271e-05 | -1.0092001e-08 | -0.00068191487 | 1 | 0.0003446533 | 0.0016609988 | 0 | 1.3300735 | 1.3300735 | 2.0556147 | 0.4911 | 0.13907423 |
| V7 | 0.9918 | -0.020691975 | -0.0079819111 | -0.00011846062 | -0.00011846062 | -9.2172237e-09 | -0.001074258 | 1 | 0.00028934278 | 0.0012866911 | 0 | 1.3300694 | 1.3300694 | 2.0556111 | 0.4911 | 0.13908534 |
| V8 Full 5K | 0.9849 | -0.18371444 | -0.079698529 | -0.00365297 | -0.00365297 | -3.7832124e-09 | -0.017683495 | 1 | 0.00385252 | 0.019761536 | 0 | 1.32962 | 1.32962 | 2.0551456 | 0.4912 | 0.13927583 |
| V8 Full 12.5K | 0.9862 | -0.20140865 | -0.093121296 | -0.0044310814 | -0.0044310814 | -3.7915966e-09 | -0.019807461 | 1 | 0.0034744402 | 0.020309386 | 0 | 1.3302337 | 1.3302337 | 2.0556156 | 0.491 | 0.13909685 |

## Primary paired results (V8 12.5K minus baseline)

| Comparison | Metric | Mean | Median | Bootstrap 95% CI | W/T/L | Applicable | Status |
|---|---|---:|---:|---|---:|---:|---|
| V8 Full 12.5K minus Source | accepted | -0.0138 | 0 | [-0.0162, -0.0116] | 0/9862/138 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus Source | weighted_bac_delta | -0.20140865 | -0.18196725 | [-0.20388055, -0.19890303] | 9862/138/0 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus Source | bond_delta | -0.093121296 | -0.090909094 | [-0.094174706, -0.092081817] | 9392/608/0 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus Source | active_angle_delta | -0.0077412323 | 0 | [-0.0080140864, -0.0074670965] | 2624/3100/0 | 5724 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus Source | ring_delta | -0.052665411 | -0.047619049 | [-0.054436804, -0.050913274] | 2519/1235/7 | 3761 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus Source | rmsd | 0.00017471468 | 0.00013878942 | [0.00016183883, 0.00018757015] | 3300/139/6561 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus D1 | accepted | -0.0036 | 0 | [-0.006, -0.0012] | 54/9856/90 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus D1 | weighted_bac_delta | -0.18245727 | -0.16966941 | [-0.18468938, -0.18019208] | 9834/51/115 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus D1 | bond_delta | -0.085184162 | -0.085106387 | [-0.086201059, -0.084166546] | 9144/736/120 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus D1 | active_angle_delta | -0.0075555957 | 0 | [-0.0078289175, -0.0072815904] | 2594/3103/27 | 5724 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus D1 | ring_delta | -0.050759805 | -0.047619049 | [-0.052494713, -0.049018123] | 2473/1251/37 | 3761 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus D1 | rmsd | 0.00016535367 | 0.00012528896 | [0.00015256474, 0.00017807501] | 3490/51/6459 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus V5-B | accepted | -0.0076 | 0 | [-0.0098, -0.0055] | 21/9882/97 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus V5-B | weighted_bac_delta | -0.17572258 | -0.16589875 | [-0.17780317, -0.1736283] | 9821/44/135 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V5-B | bond_delta | -0.085548512 | -0.085714288 | [-0.086562355, -0.084537575] | 9161/731/108 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V5-B | active_angle_delta | -0.007583613 | 0 | [-0.0078573091, -0.0073070043] | 2594/3110/20 | 5724 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V5-B | ring_delta | -0.05085229 | -0.047619049 | [-0.052593975, -0.049125461] | 2479/1257/25 | 3761 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V5-B | rmsd | 0.00016021349 | 0.00012290478 | [0.00014805273, 0.00017208952] | 3503/44/6453 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus V7 | accepted | -0.0056 | 0 | [-0.008, -0.0033] | 42/9860/98 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus V7 | weighted_bac_delta | -0.18071668 | -0.16904416 | [-0.18291028, -0.178485] | 9823/43/134 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V7 | bond_delta | -0.085139385 | -0.085106386 | [-0.086155184, -0.084118465] | 9144/733/123 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V7 | active_angle_delta | -0.007534278 | 0 | [-0.0078066192, -0.007260711] | 2593/3101/30 | 5724 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V7 | ring_delta | -0.049809102 | -0.047619049 | [-0.051590827, -0.048030331] | 2454/1258/49 | 3761 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V7 | rmsd | 0.00016430448 | 0.00012460351 | [0.00015163174, 0.00017676807] | 3491/43/6466 | 10000 | SIGNIFICANT_V8_12P5K_worse |
| V8 Full 12.5K minus V8 Full 5K | accepted | 0.0013 | 0 | [-0.0004, 0.003] | 46/9921/33 | 10000 | NOT_SIGNIFICANT |
| V8 Full 12.5K minus V8 Full 5K | weighted_bac_delta | -0.01769421 | -0.0026971796 | [-0.018907609, -0.016493335] | 5496/2305/2199 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V8 Full 5K | bond_delta | -0.013422767 | 0 | [-0.01407718, -0.012766215] | 4375/4529/1096 | 10000 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V8 Full 5K | active_angle_delta | -0.001359384 | 0 | [-0.0016067402, -0.0011147681] | 1193/3798/733 | 5724 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V8 Full 5K | ring_delta | -0.0056473448 | 0 | [-0.0068842372, -0.0043941045] | 639/2781/341 | 3761 | SIGNIFICANT_V8_12P5K_better |
| V8 Full 12.5K minus V8 Full 5K | rmsd | 0.00061372289 | 0.00055718422 | [0.00060339303, 0.00062448548] | 630/105/9265 | 10000 | SIGNIFICANT_V8_12P5K_worse |

## Conclusions

**A — proven:** V8 Full 12.5K improves weighted BAC, bond, active-angle, and ring metrics relative to Source on the same records. Acceptance and movement trade-offs remain explicit in the table.

**B — not yet proven:** frozen D1 is older and is not a strict matched control. No causal attribution to the V8 constraint module is made before the matched D1-only 12.5K exposure-control run.
