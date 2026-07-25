# Clash historical audit

## Implementations and semantics

- Formal MVT uses a handcrafted squared penetration term with coefficient 2.0, alongside its BA/ring/chirality/anchor terms and 40-step optimizer. It is not a learned clash distribution.
- `etflow/ecir/bac_constraints.py::sparse_clash_edges` excludes graph-distance 1–2 and 1–3 pairs by default, uses a universal 1.0 Å allowed contact and 2.0 Å search cutoff, and is therefore not atom-type calibrated.
- The frozen LSGO catastrophic guard excludes topology distance ≤2 and rejects a candidate if any remaining pair worsens beyond a 0.50 fraction of summed RDKit van der Waals radii. This is a safety fallback, not an improving force.
- PoseBusters 0.6.5 `mol_fast` uses RDKit distance-geometry bounds with `set15bounds=true`, `scaleVDW=true`, triangle smoothing and `threshold_clash=0.3`; hydrogens are ignored. Its discrete outcome is not numerically equivalent to the project's internal penetration score.

## What the frozen external evidence establishes

On the old exposed 600-record LSGO cohort, Source and every BA seed had identical Internal steric clash pass rate `91.6667%`, Overall `89.8333%`, pass→fail=0 and fail→pass=0. Thus failures were already present in Source and BA neither created nor resolved any PB-classified clash. BA was not designed to resolve nonbonded overlap: its only clash mechanism was a reject/fallback guard.

The frozen PB output records molecule-level pass/fail, not the atom pair that triggered RDKit bounds. Therefore the exact dominant element-pair types cannot be truthfully recovered from that schema alone. A separate historical diagnostic may rank internal vdW penetrations on the already exposed coordinates, but those ranks are descriptive and are forbidden from choosing BAT thresholds.

## Fixed exclusions and atom handling for BAT

- explicit Bond (1–2) and Angle (1–3) pairs: excluded;
- 1–4 pairs: retained with a reduced safe factor because they are torsion-coupled;
- other heavy-atom pairs: retained;
- ring atoms: treated by the same topology-distance rule, while ring closure/geometry remains a separate hard guard;
- hydrogens: excluded from the primary soft objective to match the stable heavy-atom geometry pipeline and PB's configured clash check; atom identity and all coordinates remain preserved;
- radii: RDKit periodic-table van der Waals radii, frozen before external evaluation.

No PoseBusters result is used to set these rules.

