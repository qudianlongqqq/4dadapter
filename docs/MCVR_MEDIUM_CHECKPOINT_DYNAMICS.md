# MCVR Medium Checkpoint Dynamics

All checkpoints are diagnostic controls. The formal V4 result remains selected step 1500.

| Run | Step | LR | Bond relative | Raw gain | Clip loss | Safety loss | Acceptance loss | Validity delta | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | 500 | 0.00020000 | 0.070985846261 | 0.000712847959 | 0.000000000000 | -0.001058904417 | -0.016828802221 | -0.045370851097 | 0.011773166984 |
| V4 | 1000 | 0.00019877 | 0.054821026980 | 0.006633514207 | 0.000000000000 | 0.000919021353 | -0.008650363792 | -0.038362212983 | 0.011352899522 |
| V4 | 1500 | 0.00019512 | 0.099835290081 | 0.015637512829 | 0.000000000000 | 0.001256273858 | -0.011778789565 | -0.084647979007 | 0.010782461651 |
| V4 | 2000 | 0.00018915 | 0.065479972625 | 0.017727475222 | 0.000000000000 | 0.004361873146 | -0.003792238113 | -0.063811800894 | 0.010489572369 |
| V4 | 3000 | 0.00017096 | 0.017010414296 | -0.116462838545 | -0.000022727311 | -0.076243145727 | -0.044654236309 | -0.008719775924 | 0.011218030289 |
| V4 | 5000 | 0.00011743 | 0.047952523729 | 0.024886299986 | 0.000000000000 | 0.013363233484 | -0.001042023338 | -0.054253913238 | 0.010541574400 |
| V4 | 7500 | 0.00004904 | 0.061284290959 | 0.029396552410 | 0.000000000000 | 0.014305591539 | -0.000967476971 | -0.064979491237 | 0.010619925946 |
| V4 | 10000 | 0.00002000 | 0.044598405625 | 0.040779955111 | 0.000000000000 | 0.029277084816 | -0.000183333680 | -0.053515294620 | 0.010688559029 |
| V3_best_overall | 2000 | 0.00020000 | 0.098274898912 | 0.014883836620 | 0.000000000000 | -0.004093486331 | -0.006773833357 | -0.090128445406 | 0.010376569018 |
| V3_formal | 10000 | 0.00020000 | 0.062951244215 | 0.036934049122 | 0.000000000000 | 0.021195177462 | -0.000756361075 | -0.074067661332 | 0.010288463179 |
| Run_A_Stage2b | 3000 | 0.00020000 | 0.122983856215 | 0.010314678341 | 0.000000000000 | -0.016208373822 | -0.005702638626 | -0.116643924555 | 0.007524595894 |

The diagnostic bond-rate peak is step `1500`; the preregistered validity-based formal selection remains step `1500`.
From step 1500 to 10000, raw gain changed `0.015637512829 -> 0.040779955111` while safety-gate loss changed `0.001256273858 -> 0.029277084816`. The late decline is therefore not raw-proposal degradation; it is dominated by stronger learned safety attenuation.
Trust-clipping loss remained negligible and acceptance loss was non-positive at both points, so clipping and deterministic acceptance did not cause the late decline.
V4 diversity minimum/maximum ratio is `0.890972869283`; this does not indicate mode collapse under the frozen diversity criterion.
A classical overfitting claim is not supported because this audit has no matched training-set performance curve; the observed validation dynamics are attributable to stage behavior, not labeled as overfitting.
The Pearson correlation between registered LR and accepted bond improvement is `0.294101517128`; the non-monotonic curve does not support a simple LR-only explanation.
These diagnostics do not authorize V5 or checkpoint reselection.
