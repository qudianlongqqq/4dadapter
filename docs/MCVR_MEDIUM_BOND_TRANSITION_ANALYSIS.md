# MCVR Medium Bond Transition Analysis

Transitions use unique undirected bonds per validation record; ring bonds are flagged as a subset without duplicate counting.

| Comparison | Normal->normal | Outlier->normal | Outlier->outlier | Normal->outlier |
|---|---:|---:|---:|---:|
| raw_proposal | 21270 | 1535 | 9150 | 655 |
| accepted | 21733 | 1268 | 9417 | 192 |
| minimal_target | 21645 | 6295 | 4390 | 280 |

The accepted model's new/repaired ratio is `0.151419558360`.

## Largest new-outlier environments

| Dimension | Environment | New | Repaired | Net repaired |
|---|---|---:|---:|---:|
| ring | non_ring | 192 | 1173 | 981 |
| aromatic | non_aromatic | 192 | 1200 | 1008 |
| branch | branched | 192 | 1267 | 1075 |
| bond_type | SINGLE | 190 | 1168 | 978 |
| heteroatom | carbon_only | 189 | 1162 | 973 |
| flexibility | rotatable_ge_6 | 154 | 995 | 841 |
| flexibility | rotatable_3_5 | 33 | 243 | 210 |
| flexibility | rotatable_le_2 | 5 | 30 | 25 |
| heteroatom | heteroatom | 3 | 106 | 103 |
| bond_type | DOUBLE | 2 | 32 | 30 |
