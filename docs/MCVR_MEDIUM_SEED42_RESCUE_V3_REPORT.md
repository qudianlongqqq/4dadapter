# MCVR Medium Seed42 Rescue V3 Final Report

Decision: **MEDIUM_SEED42_FAIL**

Monitoring audit: **POST_CLIP_THRESHOLD_SELF_TRIGGER**. Trust clipping mathematics was not changed.

## Completion and segmented timing

| Item | Value |
|---|---:|
| Completed optimizer steps | 20000 / 20000 |
| Training status | COMPLETED |
| Stop reason | none |
| V3 added training wall seconds | 2124.937 |
| V3 added active optimizer seconds | 1287.337 |
| V2+V3 total training wall seconds | 2469.578 |
| V2+V3 active optimizer seconds | 1447.128 |
| V2+V3 validation seconds | 303.126 |
| V3 pipeline wall seconds | 2262.375 |
| Mean optimizer steps/s (cumulative) | 13.8205 |
| Mean examples/s (cumulative) | 110.5058 |
| Estimated 100k active hours | 2.010 |
| Resume checkpoint | E:\3dconformergenerationcode\4dadapter\logs_ecir_mvr\medium\run_a_seed42_rescue_v2_20k\checkpoints\last.ckpt |
| Resume step/reason | 2450 / POST_CLIP_THRESHOLD_SELF_TRIGGER |
| Downtime seconds | 2662.626 |

## Checkpoint selection

Formal-policy selected step: **10000** (`5b312915516c258429a980a4c2341391853c4bf6a4948b17346f4b7927d2a6b1`).

Best overall step: **2000**; early: `True`.

| Step | Segment | Formal | Validity delta | RMSD delta | MAT-P | MAT-R | Identity |
|---:|---|---|---:|---:|---:|---:|---:|
| 1000 | V2 | False | -0.015900 | 0.000167 | 0.000167 | 0.000127 | 1.000000 |
| 2000 | V2 | False | -0.090117 | 0.000700 | 0.000700 | 0.000995 | 1.000000 |
| 3000 | V3 | False | -0.070379 | 0.000517 | 0.000517 | 0.000848 | 1.000000 |
| 5000 | V3 | True | -0.062390 | 0.000338 | 0.000338 | 0.000548 | 1.000000 |
| 10000 | V3 | True | -0.074069 | 0.000556 | 0.000556 | 0.000860 | 1.000000 |
| 15000 | V3 | True | -0.063421 | 0.000355 | 0.000355 | 0.000590 | 1.000000 |
| 20000 | V3 | True | -0.073478 | 0.000508 | 0.000508 | 0.000567 | 1.000000 |

## Final Gate metrics

| Metric | Upstream | V3 accepted | Delta |
|---|---:|---:|---:|
| Total validity | 0.793989 | 0.719921 | -0.074068 |
| RMSD | 1.321787 | 1.322344 | 0.000556 |
| MAT-P | 1.321787 | 1.322344 | 0.000556 |
| MAT-R | 2.375986 | 2.376846 | 0.000860 |
| COV-P | 0.482000 | 0.482000 | 0.000000 |
| COV-R | 0.068843 | 0.068776 | -0.000067 |

High-flex validity: `0.781884 -> 0.702904`.
Unseen validity: `1.550886 -> 1.368889`.
Clean identity: `1.000000`.
Gate conditions: **26/27**.

Seed43/44 permitted: **no**. Generated commands: `[]`.
Seed43/44 were not executed. 100k and test evaluation were not run.

## V3 interval timing

| Step end | Interval seconds | Active seconds | Steps/s | Examples/s |
|---:|---:|---:|---:|---:|
| 3000 | 120.125 | 42.401 | 12.9714 | 103.6768 |
| 4000 | 115.344 | 74.245 | 13.4689 | 107.6975 |
| 5000 | 157.812 | 75.054 | 13.3237 | 106.5366 |
| 6000 | 115.922 | 80.477 | 12.4259 | 99.3576 |
| 7000 | 108.625 | 75.921 | 13.1716 | 105.3200 |
| 8000 | 107.656 | 75.590 | 13.2293 | 105.7812 |
| 9000 | 104.422 | 73.566 | 13.5932 | 108.6915 |
| 10000 | 152.797 | 75.783 | 13.1956 | 105.5118 |
| 11000 | 103.891 | 73.539 | 13.5982 | 108.7314 |
| 12000 | 101.015 | 71.074 | 14.0698 | 112.5025 |
| 13000 | 100.516 | 71.229 | 14.0392 | 112.2576 |
| 14000 | 100.500 | 71.391 | 14.0074 | 112.0029 |
| 15000 | 145.281 | 71.585 | 13.9694 | 111.6994 |
| 16000 | 101.329 | 71.058 | 14.0730 | 112.4715 |
| 17000 | 100.469 | 71.225 | 14.0400 | 112.2640 |
| 18000 | 100.172 | 71.076 | 14.0694 | 112.4993 |
| 19000 | 100.016 | 70.785 | 14.1273 | 112.9618 |
| 20000 | 144.156 | 71.338 | 14.0178 | 112.0861 |
