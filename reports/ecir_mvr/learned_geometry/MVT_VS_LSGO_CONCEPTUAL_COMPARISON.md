# MVT versus LSGO conceptual comparison

| property | MVT | LSGO |
|---|---|---|
| Bond target | handcrafted/reference-statistical objective | learned conditional μ |
| Angle target | handcrafted/reference-statistical objective | learned conditional μ |
| uncertainty | fixed/artificial | C attempted learned σ; failed, B retains frozen σ |
| BAC global weights | manually set component weights | no fixed BAC weights; equal group aggregation |
| coordinate teacher | yes | no |
| ~40-step solver | yes | no |
| real Source error state | manually defined objective terms | standardized learned likelihood (B μ + frozen σ) |
| update | iterative coordinate optimizer | one-step 0.003 Å gradient |
| external physics training | no | no |

LSGO removes handcrafted coordinate targets/teachers and manual BAC component weights, not all inductive bias. It still chooses Bond/Angle primitives, Gaussian likelihood, architecture, sigma floor, ridge, trust budget, step count and frozen ring handling.
