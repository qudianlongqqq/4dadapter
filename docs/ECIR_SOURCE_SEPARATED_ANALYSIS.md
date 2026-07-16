# ECIR source-separated analysis

The frozen 5k checkpoint was evaluated on the same 100 validation records, split into 50 ETFlow and 50 Cartesian-100k conformers (25 molecules per source). Set metrics were computed per molecule before source aggregation.

## All-flex summary

| Source | gate mean | aligned displacement (Å) | bond Δ | angle Δ | torsion Δ | ring Δ | RMSD Δ (Å) | MAT-P Δ | MAT-R Δ | improved/worsened |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cartesian 100k | 0.10243 | 0.14842 | -0.07904 | -0.05172 | -0.03391 | -0.06724 | +0.03762 | +0.04829 | +0.04153 | 100% / 0% |
| ETFlow formal | 0.03614 | 0.00651 | -0.00146 | +0.00010 | -0.00080 | -0.00249 | +0.00072 | +0.00072 | +0.00104 | 88% / 12% |

`COV-P` is unchanged for both sources. `COV-R` is unchanged for Cartesian and changes by -0.00133 for ETFlow. “Improved” is a diagnostic sign test on the unweighted sum of the five persisted internal errors with a `1e-6` near-unchanged band; because the terms have mixed units it is not a formal validity score.

## High-flex (`rotatable >= 6`)

- Cartesian (15 molecules): RMSD `+0.04896 Å`, MAT-P `+0.05669`, MAT-R `+0.05091`, aligned displacement `0.17216 Å`.
- ETFlow (2 molecules): RMSD `+0.00283 Å`, MAT-P `+0.00283`, MAT-R `+0.00344`, aligned displacement `0.00885 Å`.

## Required answers

1. **Does ECIR mainly improve extreme Cartesian input? Yes.** Nearly all large internal-metric gains are from Cartesian records.
2. **Does it over-repair normal ETFlow input? Not at the current four-step setting.** ETFlow movement is small, although 12% of ETFlow molecules worsen under the mixed-unit diagnostic and angle error changes slightly in the wrong direction.
3. **Which source causes RMSD/MAT degradation? Cartesian 100k.** Its RMSD/MAT deltas are roughly 40–67 times the ETFlow deltas.
4. **Which source causes high-flex degradation? Cartesian 100k.** The directional effect is unambiguous; the ETFlow high-flex sample is only two molecules and should not be overinterpreted.

The per-conformer, per-molecule, and subset tables are in `diagnostics/ecir_mvr/root_cause/`.
