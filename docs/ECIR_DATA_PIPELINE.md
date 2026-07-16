# ECIR data pipeline

## Error atlas

`scripts/build_conformer_error_atlas.py` accepts one cache or a JSON list of
heterogeneous sources. Each source declares its real coordinate field (for
example `x_init` or `x_cart`), checkpoint, NFE, solver, seed and split paths.
Selection is molecule-capped and targets are written outside the formal source
cache. Parquet rows retain the original path, coordinate field and target path.

The current atlas contains:

- train: 500 records / 250 molecules, split evenly between formal ETFlow and
  Cartesian-100k rollout;
- val: 100 records / 50 molecules, split evenly between the same sources;
- test: 100 ETFlow records / 50 molecules, held out and unused for selection.

Atlas identity:
`aa0db9d67d57cc2077557fac76270bbf1322f295d76852b2d3d310d309f2e985`.

## Real targets

MMFF94s is attempted with harmonic position anchors. A target is accepted only
when coordinates are finite, Kabsch drift stays within the configured basin
threshold and force-field energy does not increase. Optimization status,
energies, steps and drift are retained even when the maximum iteration count is
reached. Unsupported MMFF may use UFF, but UFF is always a separate population;
unsupported energy is null, never zero.

If relaxation is rejected, multi-reference soft coupling computes RMSD,
torsion and internal-geometry costs. One aligned reference is sampled from the
soft distribution. Cartesian coordinates are never averaged.

## Mixed training records

`ECIRMixedDataset` uses configurable initial ratios:

- real error: 0.50;
- structured synthetic error: 0.35;
- clean identity: 0.15.

Synthetic modes are torsion, multi-torsion, angle, bond strain, clash, ring,
mixed and zero. Metadata records affected bonds/atoms, amplitude, exact
correction direction and pre/post metrics. Clean samples have identical input
and target.

Splits are molecule-defined. Test is not consumed by training, threshold choice
or go/no-go selection.
