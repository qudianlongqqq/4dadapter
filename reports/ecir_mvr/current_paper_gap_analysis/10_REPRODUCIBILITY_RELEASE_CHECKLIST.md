# Reproducibility and final release checklist

## MUST_HAVE

- clean immutable Git commit for all paper-path source;
- exact executed Restricted/Unrestricted runners, not merely current equivalents;
- frozen configs and command lines;
- TRAIN/DEV/final-test manifests with identity/order/hash and split lineage;
- selected step-17,500 checkpoints plus optimizer/scheduler/RNG recovery state or a documented minimal inference checkpoint;
- Python, Torch, CUDA, RDKit, PoseBusters, xTB and OS/environment lock information;
- evaluator source and external-environment identity;
- per-record scientific results needed to reproduce aggregates;
- protocol definitions for V3D, PB, Source RMSD, Reference RMSD, and xTB;
- SHA256 manifest covering code snapshot, configs, manifests, checkpoints, evaluator, results, and audits;
- a final audit that maps every paper table/figure number to an artifact hash;
- protected/final test access log proving no post-outcome model changes.

## RECOMMENDED

- container/conda lock plus installation instructions;
- deterministic seed and CUDA caveat documentation;
- smoke test with expected hashes/tolerances;
- inference-only example and resource profile;
- archived stdout/stderr and recovery continuity report;
- public data preparation script where licensing permits;
- machine-readable schema descriptions for all tables.

## OPTIONAL

- full optimizer checkpoints for every historical ablation;
- legacy Sigma-v2/MCVR/BAT branches in the main release;
- all intermediate development caches;
- bitwise identity across different GPU/RDKit builds where tolerance equivalence is documented.

```text
REPRODUCIBILITY_COMPLETENESS = PARTIAL
CURRENT_BLOCKING_PROVENANCE_GAPS = UNTRACKED_CURRENT_PAPER_PATH; EXECUTED_RUNNER_BYTES_NOT_ARCHIVED
```

