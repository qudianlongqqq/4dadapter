# Trust-region audit

The primary Source correction remains a single direct-gradient step with graph RMS capped at 0.003 Å and per-atom displacement capped at 0.03 Å. No 0.01/0.02 Å rescue was evaluated.

Across the frozen three-seed BA+C internal cohort:

| partition | records | mean RMS (Å) | p95 RMS (Å) | fallback | new catastrophic |
|---|---:|---:|---:|---:|---:|
| DEV_A | 432 | 0.002911 | 0.003000 | 0% | 0 |
| DEV_B | 432 | 0.002912 | 0.003000 | 0% | 0 |

Reference stationarity used the historical 0.001 Å diagnostic budget. Median and p95 movement were 0.001 Å for all three seeds; chirality and ring non-regression were 100%. This is a budget-bounded stationarity diagnostic, not an exact zero-gradient claim.

Formal test reads=0; frozen holdout reads=0; PB/xTB access=false.
