# Torsion internal calibration

Decision: **TORSION_NO_GO**

| seed | partition | fixed κ | uniform NLL | single VM | K3 VM | Source>Reference | active | min occupancy | duplicated |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 211 | dev_a | 7.224 | 1.8379 | 7.0535 | 1.4826 | 0.502 | 0.062 | 0.305 | 0.063 |
| 211 | dev_b | 7.224 | 1.8379 | 7.0772 | 1.4571 | 0.510 | 0.057 | 0.306 | 0.060 |
| 223 | dev_a | 7.243 | 1.8379 | 7.2116 | 1.4666 | 0.534 | 0.066 | 0.309 | 0.062 |
| 223 | dev_b | 7.243 | 1.8379 | 7.1903 | 1.4542 | 0.506 | 0.056 | 0.307 | 0.070 |
| 239 | dev_a | 7.137 | 1.8379 | 7.0612 | 1.4798 | 0.530 | 0.067 | 0.322 | 0.091 |
| 239 | dev_b | 7.137 | 1.8379 | 7.0408 | 1.4608 | 0.514 | 0.055 | 0.314 | 0.108 |

Checks: `{"active_fraction_bounded": true, "better_than_uniform": true, "k3_not_worse_than_single": true, "no_component_collapse": true, "no_mode_duplication": true, "source_selectivity": false}`

All κ estimation and selection used TRAIN residuals plus DEV_A/DEV_B likelihood only. External metrics were locked.
