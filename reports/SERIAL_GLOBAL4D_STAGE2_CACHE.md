# Serial Global4D Stage 2 Cache

Status: **COMPLETED**. Scientific status remains **ORACLE_PASS**; Phase A has
not started, so this is not `DECISION_B`.

The old diagnostic check used a fixed absolute tolerance of `1e-5`. It now
uses exactly:

```python
math.isclose(persisted, recomputed, rel_tol=1e-6, abs_tol=1e-4)
```

This change applies only to persisted floating-point RMSD diagnostics. Hard
identities, indices, hashes, shapes, selected reference tensors, teacher,
config, manifest, and sampling identities remain exact and fail closed.

The formerly failing record at manifest index 4557 retained the same
`x_init_hash` and selected reference index 7. Its persisted value
`575.6069946289062` and recomputed value `575.6070556640625` differ by
`6.103515625e-05` (relative delta `1.0603615026849413e-07`). The effective
tolerance is `5.756070556640625e-04`, so its status is now
`PASS_NUMERICALLY_CLOSE`.

## Field semantics audit

Twenty records selected with seed 42 were compared against five quantities:
the persisted field, current recomputation, standard per-atom RMSD, squared
error sum, and mean squared coordinate error. The sampled manifest indices
were:

`204, 217, 244, 260, 712, 767, 839, 912, 1143, 1791, 1828, 1905, 2006,
2253, 3456, 4139, 4467, 4597, 4837, 4931`.

For all 20, the current value exactly matched the standard formula
`sqrt(mean(sum((x_ref_aligned - x_init)^2, dim=-1)))`; maximum formula delta
was zero and maximum persisted/recomputed float32 drift was `4.7684e-7`.
All selected reference IDs matched their selected indices. Therefore the field
is standard per-atom RMSD in the stored coordinate unit.

For the extreme record, the standard RMSD is `575.6070556640625`, squared
error sum is `18,554,114.0`, and mean squared coordinate error is
`110,441.15625` across 56 atoms. The large value is a real extreme coordinate
discrepancy in that source record, not a different historical formula. It is a
diagnostic naming/scale concern only; neither the formal source cache nor any
training labels were rebuilt.

## Resume and completion

All 4,557 existing files formed the exact prefix `00000000.pt` through
`00004556.pt`. Every record passed schema/target validation and matched the
frozen manifest order, sample and molecule IDs, `x_init_hash`, graph sizes,
flexibility cohort, teacher identity, sampling identity, target-time schedule,
and manifest identity. There were no duplicates or temporary files.

The build therefore safely resumed at the 4,558th record and generated the
remaining 444 records. It did not delete or rebuild the partial cache. A final
independent audit validated all 5,001 records, the manifest identity, and the
training DataLoader completion gate.

- Pilot manifest canonical SHA256:
  `28ae322407592a69bf41ace35a12ae86c63338c56e2d37e2e2ce3dec9879e6b0`
- Teacher sampling identity SHA256:
  `4a0461db7f3069ebb44d4d0cfc3ff5cc7faad46a19abc4f717c3955d02ceaccf`
- Cache manifest SHA256:
  `89fb9a125f6710c27b748aff33aafa03b4fbf921e0c1c827fda05c816e0a28cb`
- Final cache identity SHA256:
  `3511ccce3422dcb945f3cc3cccce7d6718960198d60b6e2b0f7d9dc0e54f13e8`
- Raw `train_manifest.json` SHA256:
  `486f496d39c64145998d28d4f78a4b51836e4a1788369491ac1068f3e0f0b08f`

The builder wrote `COMPLETED.json` atomically. The train dataset now refuses
to initialize without a valid marker whose record count and identity match the
cache. No record was skipped or replaced, and the formal source cache and
frozen pilot manifest were not modified.

Fix commit: `2b73e01cac02fa9280d779e383b1a5974255f1a7`.

Resume/COMPLETED gate commit: `e12ac7cd751d9694db1eb12b01339a13dca77107`.
