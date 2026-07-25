# Frozen DRCSR versus LSGO neural mean

This is the valid same-identity comparison. Variant A is the untouched DRCSR typed-context median/scale objective; Variant B changes only μ to a neural continuous-context predictor and retains the frozen DRCSR σ.

| evidence | DRCSR A | LSGO B (3-seed median/mean) | conclusion |
|---|---:|---:|---|
| DEV_A held-out joint NLL | 61.056975 | 60.483366 | B better by 0.573609 |
| DEV_B held-out joint NLL | -2.096982 | -2.253617 | B better by 0.156635 |
| xTB median ΔE | -0.475170 | -0.804526 ± 0.004977 | B substantially stronger |
| xTB improved fraction | 86.83% | 93.61% ± 0.38% | B stronger |
| xTB p95 | 0.090186 | worst 0.000000 | B tail safer |
| PB overall | 89.833% | 89.833% ± 0 | tied; no transfer |

Neural μ therefore improves both held-out likelihood and external energy direction relative to handcrafted DRCSR buckets. It does not improve PB, and B uses frozen statistical σ; this is not evidence for learned uncertainty.
