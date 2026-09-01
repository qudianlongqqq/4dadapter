# Final development provenance freeze checklist

The authoritative development implementation is still untracked as a coherent release unit. `git status --short` reports the current J1-R1 source, configs, runners, evaluators, artifacts and reports as untracked. Their files exist and can be hashed, but filesystem presence is not immutable provenance.

## Freeze before protected evaluation

- [ ] Freeze the two formulation identities and explicitly prohibit outcome-dependent winner selection.
- [ ] Commit/tag exact model source and shared geometry/VJP/rigid-projection code.
- [ ] Commit exact Restricted and Unrestricted configs and training/evaluation runners.
- [ ] Freeze TRAIN/DEV/final manifests, record ordering, molecule identities and overlap audits.
- [ ] Hash step-17,500 checkpoints, recovery states and per-run RNG/environment manifests.
- [ ] Freeze V3D, PoseBusters, RMSD, xTB and baseline evaluators, including external Python/xTB versions.
- [ ] Export CUDA/PyTorch/RDKit/PoseBusters/xTB/OS/driver dependency inventories.
- [ ] Freeze completed multiseed tables, component tables, xTB tables and integrity audit.
- [ ] Freeze this audit and all final statistical/claim definitions.
- [ ] Generate a root SHA256 manifest and immutable release tag/archive.

## Recoverability boundary

Historical seed307 execution bytes cannot be proven retrospectively beyond surviving files, logs, hashes and behavioral equivalence audits; do not claim a historical commit that did not exist. From the current state onward, exact files, checkpoints, manifests, environments and outputs can be fully archived and verified.

Representative current hashes are recorded in `FINAL_POST_MULTISEED_STATUS.json`, but the authoritative freeze requires committing or otherwise immutably archiving the entire dependency closure.

```text
FINAL_DEVELOPMENT_FREEZE_REQUIRED_NOW = YES
PROVENANCE_FREEZE_READY = NO
HISTORICAL_RUNNER_PROVENANCE_FULLY_REPAIRABLE = NO
FORWARD_REPRODUCIBILITY_CAN_BE_COMPLETE = YES
```
