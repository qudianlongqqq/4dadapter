# D1-B Formal Runtime Fix Audit

## Decision

`PENDING_LINUX_EXHAUSTIVE_SCAN`

The reported training failure is a runtime adapter failure on a standalone
mapped `[H+]` component. The previous adapter assumed every explicit hydrogen
had one heavy-atom parent. That assumption is invalid for a disconnected ionic
component. The fix is generic and contains no sample ID, dataset index, atom
index, or molecule-specific exception.

Windows does not contain the reported Linux source and target payload for
dataset index 46546. Therefore this audit does not claim independent inspection
of atom 73, does not claim that all 160,000 records are runtime-ready, and does
not emit `D1B_FORMAL_RUNTIME_READY`. The Linux exhaustive scan is the authority
for those conclusions.

## Adapter Semantics

- Bonded explicit hydrogens retain the existing heavy-parent mapping path.
- Degree-zero atoms never enter a heavy-parent hydrogen bucket.
- A disconnected atom is matched by exact atomic number, optional cache formal
  charge, optional isotope, semantic atom map when one exists, and exact graph
  component topology.
- Consecutive `0..N-1` cache IDs remain positional identities only. They verify
  cache/x_init/x_ref ordering and are never treated as RDKit atom-map numbers.
- A single uniquely typed disconnected ion can be mapped without guessing.
- Multiple indistinguishable disconnected atoms without semantic identity fail
  closed. Complete positive semantic maps permit exact one-to-one mapping.
- Atom count, ordered atomic numbers, bond endpoints, bond types, topology
  signature, formal charge metadata, isotope metadata, and component count are
  validated. Coordinates are not changed.

## Runtime Validation

`scripts/validate_ecir_mvr_formal_runtime.py` performs a deterministic CPU-only
scan of the complete train and val manifests through source PT loading, target
PT loading, strict target payload validation, `adapt_formal_cache_record`,
`MCVRMixedDataset.__getitem__`, topology construction, and PyG `Data` creation.
It requires exactly 150,000 train and 10,000 val pairs, verifies exact unique
pairing, collects every record failure, records all disconnected/ionic runtime
observations, supports identity-bound progress resume, and atomically writes the
JSON and Markdown reports. No test manifest is named or read.

The runtime report is bound to the base config, immutable source/target asset
identities, current adapter SHA, all runtime-path code SHAs, and current git
commit. Preflight requires this report before querying the GPU. The finalizer
requires the same report to be bound into the formal64 preflight. Formal
training requires both `D1B_FORMAL_TARGETS_READY` and
`D1B_FORMAL_RUNTIME_READY` before CUDA initialization or output creation.

## Target Rebuild Decision

No formal target was modified or rebuilt. The supplied Linux evidence locates
the exception in runtime adaptation, not target construction. A runtime-only
fix is sufficient in unit tests, but its sufficiency for the real data remains
conditional on the exhaustive Linux scan. There is currently no evidence that
a targeted or full minimal-target rebuild is required. Any source/target
identity, atom-order, charge, component, or coordinate mismatch will keep the
runtime gate closed instead of silently repairing an asset.

## Windows Verification

- Related targeted suite: `152 passed, 20 warnings`.
- Full suite: `557 passed, 23 warnings`.
- Ruff: passed.
- Compileall: passed.
- No formal train/val/test payload was read on Windows.
- No GPU preflight or training was started.
- Existing performance numerical-equivalence tests remain passing.

## Linux Execution Order

Run these only after the adapter/readiness changes are committed on Linux.

1. Exhaustive CPU runtime validation, including safe resume:

```bash
python scripts/validate_ecir_mvr_formal_runtime.py \
  --config configs/ecir_mvr_formal_large_d1b_base.yaml \
  --resume
```

The command must finish with `D1B_FORMAL_RUNTIME_READY`. Its fixed outputs are
`reports/ecir_mvr/D1B_FORMAL_RUNTIME_VALIDATION.json` and
`reports/ecir_mvr/D1B_FORMAL_RUNTIME_VALIDATION.md`.

2. Formal64 preflight on the same commit:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/preflight_ecir_mvr_formal_large.py \
  --config configs/ecir_mvr_formal_large_d1b_base.yaml \
  --gpu-index 1 \
  --allow-shared-gpu \
  --target-effective-batch 64
```

3. Finalize the formal64 config from the new runtime and preflight evidence:

```bash
python scripts/finalize_ecir_mvr_formal64_config.py \
  --base-config configs/ecir_mvr_formal_large_d1b_base.yaml \
  --runtime-report reports/ecir_mvr/D1B_FORMAL_RUNTIME_VALIDATION.json \
  --preflight-report reports/ecir_mvr/formal64_preflight/D1B_FORMAL_PREFLIGHT.json \
  --output reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml
```

4. After manual confirmation, restart formal64 training from step 0:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_ecir_mvr_medium_rescue_v2.py \
  --config reports/ecir_mvr/D1B_FORMAL_RECOMMENDED_CONFIG.yaml \
  --data_audit /home/aidd4090v2/Experiment/qdl/data/4dadapter/ecir_mvr/formal_large/statistics/validation.json \
  --device cuda:0
```

The failed run produced no checkpoint. `--resume_checkpoint` and controller
resume are forbidden for this restart; it must begin at optimizer step 0 with
the frozen 64 x 25,000 budget.
