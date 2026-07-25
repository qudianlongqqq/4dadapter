# Fresh prospective GFN2-xTB single-point evaluation

All 4,200 jobs completed successfully with xTB 6.7.1/GFN2. Geometry optimization was disabled. ΔE is paired to the same frozen Source in kcal/mol.

| method | mean ΔE | median ΔE | improved | p95 | p99 | max harmful |
|---|---:|---:|---:|---:|---:|---:|
| BA seed173 | -0.9457 | -0.9038 | 93.67% | 0 | 0 | +0.0645 |
| BA seed181 | -0.9430 | -0.9060 | 93.17% | 0 | 0 | +0.0683 |
| BA seed193 | -0.9405 | -0.9043 | 93.67% | 0 | 0 | +0.0740 |
| BA mean ± sample SD | -0.9430 ± 0.0026 | -0.9047 ± 0.0012 | 93.50% ± 0.29% | 0 | 0 | +0.0689 ± 0.0048 |
| BA+C seed173 | -0.9762 | -0.9165 | 99.17% | -0.3258 | -0.0882 | +0.0645 |
| BA+C seed181 | -0.9749 | -0.9171 | 99.00% | -0.3285 | -0.0995 | +0.0683 |
| BA+C seed193 | -0.9723 | -0.9128 | 99.17% | -0.3284 | -0.1158 | +0.0740 |
| BA+C mean ± sample SD | -0.9745 ± 0.0020 | -0.9154 ± 0.0023 | 99.11% ± 0.10% | -0.3276 ± 0.0015 | -0.1012 ± 0.0139 | +0.0689 ± 0.0048 |

Fresh BA therefore replicates the historical negative-median, majority-improved, safe-tail result.

The BA+C energy row is not a clean clash-barrier attribution. Relative to exact frozen BA, BA+C coordinates differ on 44/46/44 records for seeds 173/181/193; only 12 per seed are steric-active, while 32/34/32 inactive differences arise because the combined solver backtracks BA candidates that the exact historical BA safety guard rejects. This confounding was discovered after coordinate freeze and is reported without retuning. PB remains unchanged, so these xTB gains cannot establish steric validity.

Failure/timeout/nonfinite = 0/0/0 for every condition. Formal test reads=0; frozen holdout reads=0.
