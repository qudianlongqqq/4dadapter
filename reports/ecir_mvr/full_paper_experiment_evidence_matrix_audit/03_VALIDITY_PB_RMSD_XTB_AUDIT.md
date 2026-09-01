# Validity, RMSD and xTB protocol audit

## Validity3D

The external worker is pinned to GenBench3D commit `0926bc6614509aa10ccf6f69da0405d4be6af6b3`, adapter SHA256 `6ebc450b...`, and frozen LigBoundConf reference/cache hashes. Parameters are `q_value_threshold=0.001`, steric-clash safety ratio `0.75`, maximum ring-plane distance `0.1`, torsions excluded and hydrogens not considered by V3D.

Per record, V3D is the conjunction of bond geometry, angle geometry, aromatic-ring geometry and intramolecular steric-clash validity. Each component and overall are averaged across exactly 5,000 records; seed summaries are then mean ± sample SD across three seeds. Final inference must retain molecule-cluster uncertainty because there are two records per molecule.

```text
V3D_ROLE = PRIMARY_STRUCTURAL_VALIDITY_METRIC
V3D_COMPONENTS_REQUIRED = YES
```

## PoseBusters

The stable external environment uses PoseBusters `0.6.5`, `mol_fast.yml`, one worker and a frozen selected-column conjunction. Current components are:

- molecule loaded, sanitization and InChI convertibility;
- all atoms connected and no radicals;
- bond lengths and bond angles;
- internal steric clash;
- aromatic-ring flatness, non-aromatic-ring non-flatness and double-bond flatness.

No chirality component is present in the selected current protocol; the paper must not imply otherwise. PB overall is the per-record logical AND of all selected components, followed by a rate over the frozen denominator.

V3D and PB overlap on bonds, angles, clashes and some ring geometry. They are not duplicates: V3D uses learned reference-geometry distributions and its four-part conjunction; PB additionally checks parsing/chemical graph integrity, radicals, connectivity and several planarity conditions.

```text
POSEBUSTERS_ROLE = SECONDARY_CHEMICAL_AND_GEOMETRIC_INTEGRITY_BATTERY
REDUNDANCY = PARTIAL_NOT_COMPLETE
BOTH_REQUIRED = YES__V3D_PRIMARY_PB_SECONDARY
```

## RMSD vocabulary

1. **Raw Source displacement RMS:** direct proposal-minus-Source Cartesian RMS, no Kabsch, Å. This is the quantity mislabeled `source_rmsd` in the multiseed summary.
2. **Kabsch-aligned Source RMSD:** fixed-order proposal-to-Source RMSD after translation and proper SO(3) rotation, Å.
3. **Reference RMSD:** fixed-order explicit-all-atom float64 Kabsch RMSD to the nearest member of the frozen Reference ensemble, Å.

Atom correspondence checks identity, atom count, atomic numbers and order. Hydrogens are explicit and included for the Kabsch Reference calculation. Reflections are rejected. No symmetry-equivalent atom permutation search is performed. The paper main table should contain raw Source displacement and Reference RMSD; aligned Source RMSD is diagnostic/supplementary. The near-zero change in Reference RMSD is consistent with a tiny local refiner and must not be marketed as global conformer recovery.

## xTB

Completed development evidence uses xTB `6.7.1`, GFN2-xTB single-point energy, no geometry optimization, delta-energy in kcal/mol versus the exact matched Source, and 5,000/5,000 successful records for every formulation/seed. Median is primary, followed by fraction below zero, 5% trimmed mean, P90/P95/P99 and explicit >+25/+50/+100 kcal/mol tail counts; arithmetic mean is secondary.

Final matched energy evaluation should include Source, Restricted, Unrestricted and MMFF94s-optimized geometry. GFN2-xTB geometry optimization is a different baseline and remains optional unless superiority to quantum optimization is claimed.
