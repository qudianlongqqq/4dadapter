# Post-hoc BA scale calibration

The exact historical BA anchor uses the frozen typed DRCSR/reference scales. Changing them would invalidate the incremental BA→BA+C comparison. This audit measures neural-mean residual calibration but does not modify coordinate inference.

The neural network does not output sigma. Consequently sigma inflation cannot occur. `train_residual_gamma` is the robust MAD scale of `(q_ref-mu_neural)/sigma_frozen`; values different from one quantify calibration mismatch rather than a trainable uncertainty.

| seed | partition | group | gamma | raw z std | calibrated z std | raw |z|<1 | calibrated |z|<1 |
|---:|---|---|---:|---:|---:|---:|---:|
| 173 | train | bond | 0.2480 | 6.4901 | 26.1670 | 0.954 | 0.637 |
| 173 | train | angle | 0.6822 | 1.7049 | 2.4991 | 0.761 | 0.636 |
| 173 | dev_a | bond | 0.2480 | 15.7939 | 63.6784 | 0.951 | 0.626 |
| 173 | dev_a | angle | 0.6822 | 2.1041 | 3.0842 | 0.755 | 0.630 |
| 173 | dev_b | bond | 0.2480 | 0.4897 | 1.9745 | 0.953 | 0.635 |
| 173 | dev_b | angle | 0.6822 | 1.7311 | 2.5375 | 0.765 | 0.636 |
| 181 | train | bond | 0.2432 | 6.4903 | 26.6913 | 0.955 | 0.635 |
| 181 | train | angle | 0.6710 | 1.6892 | 2.5173 | 0.766 | 0.635 |
| 181 | dev_a | bond | 0.2432 | 15.7946 | 64.9550 | 0.953 | 0.629 |
| 181 | dev_a | angle | 0.6710 | 2.0879 | 3.1115 | 0.764 | 0.632 |
| 181 | dev_b | bond | 0.2432 | 0.4854 | 1.9962 | 0.954 | 0.629 |
| 181 | dev_b | angle | 0.6710 | 1.7055 | 2.5417 | 0.771 | 0.637 |
| 193 | train | bond | 0.2498 | 6.4906 | 25.9847 | 0.954 | 0.636 |
| 193 | train | angle | 0.6892 | 1.7105 | 2.4817 | 0.758 | 0.637 |
| 193 | dev_a | bond | 0.2498 | 15.7953 | 63.2357 | 0.952 | 0.631 |
| 193 | dev_a | angle | 0.6892 | 2.1117 | 3.0638 | 0.754 | 0.632 |
| 193 | dev_b | bond | 0.2498 | 0.4847 | 1.9403 | 0.954 | 0.633 |
| 193 | dev_b | angle | 0.6892 | 1.7382 | 2.5219 | 0.762 | 0.636 |

No MVT coordinate, Cartesian delta, PB, xTB, formal test or frozen holdout was accessed.
