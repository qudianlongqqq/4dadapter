# LSGO source-fidelity report

| method      | seed | records | rms_mean_A | rms_p95_A | rms_max_A | max_atom_A | fallback  | mode_switch | topology_preserved | chirality_preserved | diversity_retention |
| ----------- | ---- | ------- | ---------- | --------- | --------- | ---------- | --------- | ----------- | ------------------ | ------------------- | ------------------- |
| A-G         | nan  | 600     | 0.002845   | 0.003     | 0.003     | 0.0177493  | 0.0516667 | 0.00166667  | 1                  | 1                   | 0.999404            |
| B-G_seed173 | 173  | 600     | 0.002815   | 0.003     | 0.003     | 0.0176185  | 0.0616667 | 0.00166667  | 1                  | 1                   | 0.999316            |
| B-G_seed181 | 181  | 600     | 0.00282    | 0.003     | 0.003     | 0.0175151  | 0.06      | 0.00166667  | 1                  | 1                   | 0.999244            |
| B-G_seed193 | 193  | 600     | 0.00281    | 0.003     | 0.003     | 0.0176214  | 0.0633333 | 0.00166667  | 1                  | 1                   | 0.999207            |

All conditions remain within the frozen 0.003 Å graph-RMS budget. B-G mode switching is 0.167% (≤0.5%), topology/chirality preservation is 100%, and mean pairwise diversity retention is 99.926%. Fallback is an exact Source no-op, not deletion.
