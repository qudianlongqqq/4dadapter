# MCVR external refinement SMOKE100

| Method | Success | Fallback | Accept | Weighted BAC | Bond | Angle | Active angle | Ring | Clash | Chirality | Mean disp. | RMSD | MAT-P | MAT-R | COV-P | COV-R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1.3402555 | 1.3353902 | 2.3893465 | 0.53571429 | 0.083907885 |
| MMFF94S | 0.98 | 0.02 | 0.98 | 0.019696168 | 0.021770897 | 0.0036350905 | 0.0036350905 | 0.029996062 | -4.8680562e-06 | 1 | 0.79687787 | 1.3770536 | 1.3734626 | 2.4011833 | 0.5255102 | 0.087353054 |
| GFN2_XTB | 1 | 0 | 1 | -0.2219532 | -0.10128647 | -0.0054257983 | -0.0054257983 | -0.019196563 | -4.8680562e-06 | 1 | 0.30015169 | 1.2888164 | 1.283238 | 2.384158 | 0.58673469 | 0.093188128 |
| V8_FULL_12P5K | 1 | 0 | 0.98 | -0.20110391 | -0.08899805 | -0.0042096478 | -0.0042096478 | -0.018856613 | -1.0702308e-07 | 1 | 0.003652611 | 1.3403682 | 1.3355092 | 2.3892779 | 0.53571429 | 0.083907885 |
| MATCHED_D1_12P5K | 1 | 0 | 0.98 | -0.21143188 | -0.099723853 | -0.00051685306 | -0.00051685306 | -0.025293259 | -1.2070377e-08 | 1 | 0.0024281129 | 1.34026 | 1.3353974 | 2.3893543 | 0.53571429 | 0.083907885 |

All methods use the same ordered frozen Source records and evaluator. External failures remain in the all-record result via bitwise Source fallback.

Native MMFF94s and GFN2-xTB energies are retained only within their own method and are not compared across methods.
