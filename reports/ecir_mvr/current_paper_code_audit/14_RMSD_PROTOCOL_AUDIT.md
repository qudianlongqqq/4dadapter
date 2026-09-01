# RMSD protocol audit

## Reference RMSD

Reference RMSD is fixed-order, explicit-all-atom, float64 Kabsch RMSD to the nearest member of the frozen Reference conformer ensemble.

- Correspondence checks: same molecule identity, atom count, atomic numbers, and atom order.
- Alignment: translation + proper SO(3) rotation; reflections rejected.
- Hydrogens: explicit and included.
- Symmetry: no atom-permutation/symmetry search.
- Ensemble aggregation: minimum RMSD across references.
- Units: Å.

Evidence: `scripts/run_sixs_current_final_evidence_completion.py:140,181-215,358-383`, `etflow/commons/kabsch_utils.py:24-62`, and `CORRESPONDENCE_AUDIT.json` (PASS).

## Source RMSD has two definitions

1. **Training/DEV summary Source RMSD**: raw, unaligned Cartesian RMS displacement (sqrt{operatorname{mean}_i|Delta x_i|^2}). Evidence: restricted runner `:914-931`; unrestricted runner equivalent. Authoritative reported means are 0.0055252578 Å Restricted and 0.0060600029 Å Unrestricted.
2. **Evidence table Source RMSD**: fixed-order Kabsch proposal-to-source RMSD. The Restricted mean is approximately 0.005525526 Å. Evidence: `02_MATCHED_SOURCE_REFERENCE_RMSD.csv`.

These values are close because rigid modes are projected, but they are not definitionally identical. A paper must name the protocol and should not merge the two columns under one ambiguous label.

## Limitations

Fixed atom order is audit-protected, but symmetry-equivalent atom permutations are not searched. This is a protocol limitation, not evidence of an implementation error.

```text
RMSD_PROTOCOL_STATUS = PASS_WITH_NAMING_QUALIFICATION
REFERENCE_CORRESPONDENCE = PASS
REFERENCE_ENSEMBLE_AGGREGATION = MINIMUM
SYMMETRY_PERMUTATION = NO
SOURCE_RMSD_NAMING_COLLISION = YES
```


