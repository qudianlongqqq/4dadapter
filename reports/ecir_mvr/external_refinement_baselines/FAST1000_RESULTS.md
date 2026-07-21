# MCVR external refinement FAST1000

| Method | Success | Fallback | Accept | Weighted BAC | Bond | Angle | Active angle | Ring | Clash | Chirality | Mean disp. | RMSD | MAT-P | MAT-R | COV-P | COV-R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1.3550721 | 1.3555674 | 2.3847055 | 0.47531513 | 0.077075669 |
| MMFF94S | 0.995 | 0.005 | 0.995 | 0.051178263 | 0.032488719 | 0.0047371568 | 0.0047371568 | 0.018046523 | -2.2782651e-08 | 1 | 0.79277999 | 1.415952 | 1.419466 | 2.3984382 | 0.43644958 | 0.07022301 |
| GFN2_XTB | 0.999 | 0.001 | 0.999 | -0.1846501 | -0.098820573 | -0.0031576404 | -0.0031576404 | -0.013759724 | -4.7138948e-07 | 1 | 0.28494727 | 1.3189954 | 1.3199119 | 2.3783042 | 0.50157563 | 0.080397045 |
| V8_FULL_12P5K | 1 | 0 | 0.986 | -0.20311826 | -0.094136319 | -0.0043258924 | -0.0043258924 | -0.020460883 | 1.2249395e-08 | 1 | 0.003415492 | 1.3552499 | 1.3557456 | 2.3846448 | 0.47531513 | 0.077040655 |
| MATCHED_D1_12P5K | 1 | 0 | 0.979 | -0.18917689 | -0.097578323 | -0.0010406779 | -0.0010406779 | -0.015707915 | 2.8788636e-09 | 1 | 0.0021742531 | 1.3550981 | 1.3555952 | 2.3846872 | 0.47531513 | 0.077075669 |

All methods use the same ordered frozen Source records and evaluator. External failures remain in the all-record result via bitwise Source fallback.

Native MMFF94s and GFN2-xTB energies are retained only within their own method and are not compared across methods.
