# Minimum paper table and figure plan

## Main tables

1. **Protected final results:** Source, Restricted, Unrestricted and MMFF94s; V3D, PB, raw Source displacement, Reference RMSD, xTB median delta-energy and MMFF failure rate. Include paired CIs or concise win/tie/loss evidence.
2. **Development stability and trade-off:** all three seeds plus mean ± seed SD and matched Restricted–Unrestricted effects; movement and xTB-tail summaries.
3. **Efficiency:** parameters, neural latency/throughput/memory, preprocessing boundary, MMFF94s runtime/success and xTB single-point runtime. Add xTB optimization only if actually run.

V3D/PB component details, Bond/Angle MAE, full xTB quantiles/tails and subgroup tables can be supplementary; a fourth main table is unnecessary unless venue space permits.

## Main figures

### Must have

1. **Method overview:** Source → primitive predictions/weights → rigid-projected normalized update → two movement operating points.
2. **Pareto figure:** structural validity/energy versus raw Source displacement for Source, Restricted, Unrestricted and MMFF94s.
3. **Molecule-level paired effect distributions:** V3D/PB discordances and xTB delta-energy, with explicit adverse-tail inset.

### Should have

- Compact movement-distribution/tail plot or integrate it into the Pareto figure.
- Runtime/throughput comparison if not fully conveyed by Table 3.

### Optional

- A small set of predeclared qualitative molecules illustrating a repaired defect and an adverse-tail case.
- Diversity-preservation diagnostic.

```text
MINIMUM_MAIN_TABLES = 3
MINIMUM_MAIN_FIGURES = 3
```
