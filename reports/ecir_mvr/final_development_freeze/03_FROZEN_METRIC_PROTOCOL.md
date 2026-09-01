# Frozen metric protocol

## RMSD vocabulary

### Raw Source Displacement RMS

For proposal coordinates `x_prop` and matched Source coordinates `x_source` with `N` atoms:

```text
sqrt((1/N) * sum_i ||x_prop_i - x_source_i||^2)
```

No Kabsch alignment is performed. Unit: Å. This is the quantity historically stored as `source_rmsd` in training/DEV tables. Future tables must use the full name **Raw Source Displacement RMS**.

### Kabsch-aligned Source RMSD

Fixed-order proposal-to-Source RMSD after centering and proper SO(3) Kabsch rotation. Reflections are rejected. Unit: Å. This is diagnostic/supplementary and must not be merged with Raw Source Displacement RMS.

### Reference RMSD

- Correspondence: same molecule identity, atom count, atomic numbers and atom order.
- Atoms: explicit all atoms; hydrogens included.
- Arithmetic: float64.
- Alignment: centering plus proper SO(3) Kabsch rotation; reflections rejected.
- Ensemble aggregation: minimum RMSD across the frozen Reference conformer ensemble.
- Symmetry-equivalent atom permutation search: NO.
- Unit: Å.

## Validity3D

```text
VALIDITY3D_DEPENDENCY = GenBench3D commit 0926bc6614509aa10ccf6f69da0405d4be6af6b3
ADAPTER_SHA256 = 6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450
Q_VALUE_THRESHOLD = 0.001
STERIC_CLASH_SAFETY_RATIO = 0.75
MAXIMUM_RING_PLANE_DISTANCE = 0.1
INCLUDE_TORSIONS = NO
CONSIDER_HYDROGENS = NO
```

Selected components are exactly:

1. `bond_geometry_valid`;
2. `angle_geometry_valid`;
3. `aromatic_ring_valid`;
4. `intramolecular_steric_clash_valid`.

Per-record overall is the logical AND of those four components. The record-level rate is descriptive. Per-molecule aggregation is the mean of that molecule's record-level binary values (success fraction). The primary paired statistical unit is the molecule.

Denominator and failures: start from the complete frozen eligible record manifest. An unreadable/missing molecule, evaluator failure or missing required component is not silently deleted; it is recorded as failure for the V3D overall endpoint and in the failure table. Component selection cannot change after outcomes are opened.

## PoseBusters

```text
POSEBUSTERS_VERSION = 0.6.5
CONFIG = installed mol_fast.yml
CONFIG_SHA256 = e263264cfbfd9c3764856c093a3a27e6c656826c388851600ee1eadf3ab09cf8
MAX_WORKERS = 1
CHUNK_SIZE = 50
CHIRALITY_COMPONENT_SELECTED = NO
```

The selected binary components are exactly:

1. `mol_pred_loaded`;
2. `sanitization`;
3. `inchi_convertible`;
4. `all_atoms_connected`;
5. `no_radicals`;
6. `bond_lengths`;
7. `bond_angles`;
8. `internal_steric_clash`;
9. `aromatic_ring_flatness`;
10. `non-aromatic_ring_non-flatness`;
11. `double_bond_flatness`.

Per-record PB overall is the logical AND of all selected components after missing component values are treated as false. Per-molecule aggregation is the mean record-level pass value. The denominator is the complete frozen eligible record manifest. Parse/evaluator/missing-component failures remain failures; no outcome-dependent row deletion is permitted.

## xTB energy protocol

```text
XTB_VERSION = 6.7.1
METHOD = GFN2-xTB (--gfn 2)
TASK = FROZEN-COORDINATE SINGLE-POINT ENERGY
GEOMETRY_OPTIMIZATION = NO
NATIVE_UNIT = HARTREE
REPORT_UNIT = KCAL/MOL
HARTREE_TO_KCAL_MOL = 627.509474
DELTA_E = (E_proposal - E_source) * 627.509474
NEGATIVE_DELTA_E = PROPOSAL_LOWER_ENERGY
```

Primary reported xTB statistics are median delta-E, fraction `delta-E < 0`, and 5% trimmed mean. Tail statistics are P90/P95/P99 and counts with `delta-E > +25`, `> +50`, and `> +100 kcal/mol`. Arithmetic mean is secondary.

Timeout, nonzero exit, nonfinite energy and parse failure are engineering failures, recorded with record identity. Primary statistics use successful finite matched Source/candidate pairs and must always report success/failure counts against the frozen denominator. No row may be excluded based on the sign or magnitude of delta-E. Failure sensitivity must be reported if any matched pair is missing.

```text
METRIC_VOCABULARY_FROZEN = YES
SOURCE_RMSD_NAMING_RESOLVED = YES
V3D_PROTOCOL_FROZEN = YES
POSEBUSTERS_PROTOCOL_FROZEN = YES
XTB_PROTOCOL_FROZEN = YES
```
