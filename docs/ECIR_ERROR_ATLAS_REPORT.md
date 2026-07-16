# ECIR conformer error atlas

Energies are reported only as paired deltas or per-heavy-atom values. MMFF94s and UFF are separate populations.

| split | records | molecules | aligned RMSD | clash score | MMFF coverage |
|---|---:|---:|---:|---:|---:|
| train | 500 | 250 | 1.160323 | 0.000002 | 1.000 |
| val | 100 | 50 | 1.278707 | 0.000000 | 1.000 |
| test | 100 | 50 | 1.238603 | 0.000000 | 1.000 |

Atlas identity: `aa0db9d67d57cc2077557fac76270bbf1322f295d76852b2d3d310d309f2e985`

## Real sources found

| source | train+val records | aligned RMSD | bond error | angle error | torsion error | relaxation energy drop | relaxation RMSD |
|---|---:|---:|---:|---:|---:|---:|---:|
| formal ETFlow upstream | 300 | 1.0274 | 0.00818 | 0.01509 | 0.60545 | 10.61 | 0.1278 |
| Cartesian teacher 100k rollout | 300 | 1.3327 | 0.13977 | 0.09598 | 0.88987 | 1347.76 | 0.2561 |

The Cartesian rollout population exposes a real and substantially different
internal-geometry error mode. These values are paired within each molecule;
absolute force-field energies are not pooled across molecules.

All 600 train+val records have MMFF94s parameters. One Cartesian record failed
the restrained target acceptance check and used a single-reference soft target;
the remaining 599 used accepted restrained relaxation. Only 10/300 upstream
records and 0/300 Cartesian records reached RDKit convergence status within 50
steps, so `optimization_success` remains distinct from a basin-safe accepted
short relaxation target.

The Cartesian Stage 2 cache exposes only its selected reference, while 90% of
the sampled upstream validation records contain multiple references. This
limitation is explicitly retained in the parquet schema.
