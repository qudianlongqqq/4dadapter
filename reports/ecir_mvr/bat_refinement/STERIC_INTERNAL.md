# BA+C internal steric gate

Decision: **STERIC_INTERNAL_GO**

| partition | records | active | BA violations | BA+C violations | penetration reduction | new catastrophic | BA+C objective Δ vs Source | RMS | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev_a | 432 | 0.0139 | 3 | 3 | 87.0747% | 0 | -0.191308 | 0.00291059 | 0.000% |
| dev_b | 432 | 0.0278 | 12 | 12 | 12.3978% | 0 | -0.181811 | 0.00291233 | 0.000% |

The gate is based on penetration magnitude because a 0.003 Å micro-step can reduce overlap continuously without necessarily crossing the binary boundary in one step. Binary violation totals remain fully reported.

Checks: `{"ba_objective_improves_from_source": true, "dev_a_penetration_improves": true, "dev_b_penetration_improves": true, "fallback_reasonable": true, "no_new_catastrophic": true, "reference_stationary": true, "rms_trust": true, "topology_chirality_safe": true}`

No PB, xTB, MVT, formal test, or frozen holdout was accessed.
