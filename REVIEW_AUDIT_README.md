# PINA reviewer-gap audit

This additive patch targets frozen commit `282bdae52bf5a7211673f685676cd8f8d5d14230`.
It does not change any default model, checkpoint, frozen dataset or historical result.

Updated 2026-11-08: begin with `REVIEW_EXPERIMENT_INVENTORY_CN.md` to avoid
repeating completed work. For the existing Linux/5090 server, use
`SERVER_HANDOFF_CN.md` and the `PINA_ASSET_ROOT` / `PINA_GENBENCH_REPO`
environment variables. The Windows commands below describe the original local
execution. The nine focused tests include asset-path relocation tests.

Use the existing local environments and asset paths declared in the scripts.
Intermediate jobs live beside the worktree under `review_gap_cache`; results go to the chosen output directory.

```powershell
$out = 'F:\codexdata\2026-11-06\prediction-is-not-action-reliability-guided\outputs\audit_v1'
$modelPython = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
$validityPython = 'E:\miniconda\envs\external-validity\python.exe'
& $modelPython -m pytest tests/test_review_interventions.py tests/test_review_reference_validation.py -q
& $modelPython scripts/audit_review_gaps.py prepare --out $out --molecules 256
& $validityPython scripts/audit_review_gaps.py evaluate --out $out
& $modelPython scripts/audit_review_gaps.py aggregate --out $out
& $modelPython scripts/audit_review_gaps.py census --out $out
& $modelPython scripts/audit_review_gaps.py guard --out $out
& $validityPython scripts/audit_review_gaps.py evaluate --out (Join-Path $out 'guard_variant')
& $modelPython scripts/audit_review_gaps.py guard_summary --out $out
& $modelPython scripts/audit_reference_lineage.py $out
& $modelPython scripts/audit_raw_reference_control.py $out
& $validityPython scripts/audit_review_gaps.py evaluate --out (Join-Path $out 'reference_control')
& $modelPython scripts/audit_raw_reference_control.py $out --summarize
& $validityPython scripts/plot_review_audit.py $out
```

The model environment's Matplotlib exits with a native DLL error on this host; plotting succeeds in `external-validity`. No numerical thresholds were changed to resolve this environment issue.

Interpretation boundaries:

- Fixed-budget comparisons reuse frozen directions, not retrained fixed-budget networks.
- Sigma-only is a coefficient sensitivity test using seed 307; it is not a fixed-sigma retraining result.
- Bootstrap clusters are molecules. The three trained seeds are fixed for those intervals.
- The new guard uses no Reference, V3D or energy oracle. Caps were evaluated separately, not selected into the original model.
- Standardized torsion-matched RDKit references must not be labelled raw GEOM coordinates.
- Stereo checks cover Source-specified tetrahedral and E/Z geometry only.
- This is exploratory reanalysis of a previously evaluated cohort. No new training was performed.
