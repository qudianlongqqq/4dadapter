# LSGO-BA v2 Full307 result

Decision: **NO_GO** (development feasibility only; extra-training exposure confound remains). Learned tau is non-collapsed and is a non-dominated Validity–RMS point among the evaluated intervention arms, but the frozen overall GO gate fails: PoseBusters changes from 0.9330 to 0.9328 and rollback changes from 5.9800% to 8.8800%, exceeding the pre-frozen exact PB non-regression and +1 percentage-point rollback constraints. No additional seed or energy stage is authorized.

```csv
method,records,molecules,V3D,PB,mean_graph_rms,median_graph_rms,p95_graph_rms,rollback_rate,accept_rate
Raw,5000,2500,0.4254,0.9326,0.0,0.0,0.0,0.0,1.0
Original-v1-003,5000,2500,0.4934,0.933,0.0028206,0.0029999999999999988,0.003000000000000027,0.0598,0.9402
V2-network-fixed003,5000,2500,0.493,0.933,0.0028205999999999995,0.0029999999999999988,0.003000000000000027,0.0598,0.9402
V2-network-fixed005,5000,2500,0.5202,0.9326,0.004543999999999999,0.0049999999999999975,0.005000000000000027,0.0912,0.9088
V2-learned,5000,2500,0.5132,0.9328,0.003907960918562882,0.00410538166761399,0.005754838231951008,0.0888,0.9112
```

Learned tau mean/median/P10/P25/P75/P90/P95: 0.0043516 / 0.0042483 / 0.0032534 / 0.0037062 / 0.0048807 / 0.0054919 / 0.0059586 Å. Collapse-zero=False; collapse-ceiling=False; validity–displacement Pareto point=True; PB non-regression=False; rollback gate=False; preregistered strict GO gate=False. No xTB, ORCA, docking, formal test, or frozen holdout was accessed.
