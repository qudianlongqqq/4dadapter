# Protected-test identity audit

> Only manifests, preregistration, identity and leakage metadata were inspected. Metric, energy and outcome-summary files were not opened.

## Formal

- Pre-existing legacy Formal split and preregistration: **yes**.
- Current manifest describes a small canonical Formal cohort of about 100 molecules / 23,882 records.
- TRAIN identity overlap: **7 molecules**; DEV overlap: **0** according to the existing outcome-blind preflight metadata.
- Upstream/source: ETFlow lineage, but its frozen checkpoint/protocol belongs to an older LSGO BA method, not the current J1-R1 full-joint pair.
- Because of TRAIN overlap, small size and method/protocol mismatch, it is not a valid untouched primary test for the current paper as-is. It could be secondary context, or be identity-refrozen before outcome access only if a clean access history is independently established.

## Large holdout

- Pre-existing, deterministic and outcome-blind selected: **yes**.
- Size: **783 molecules / 1,566 records**, two records per molecule.
- TRAIN overlap: **0**; DEV overlap: **0** according to the frozen leakage audit.
- Upstream/source: **DiTMC**, not ETFlow.
- The protocol was frozen for an older softplus-v2 checkpoint set, not the current final pair.
- **569/783 identities** had appeared in a previously opened DiTMC confirmatory cohort, so it is not project-level untouched. It remains useful as a frozen cross-upstream secondary holdout after updating only the predeclared current-method identity/protocol, without reading its outcomes.

## Decision

Neither existing asset is an adequate unique one-shot primary final test as-is. The primary final cohort should be an outcome-unseen, zero-TRAIN/DEV-overlap, same-ETFlow molecule cohort frozen for both predeclared formulations. Large holdout can serve as secondary cross-upstream robustness evidence.

```text
FORMAL_VALID_AS_PROTECTED_TEST = NO_AS_PRIMARY_AS_IS
LARGE_HOLDOUT_VALID_AS_PROTECTED_TEST = YES_AS_SECONDARY_ONLY
FORMAL_ROLE = SECONDARY_OR_REFREEZE_CANDIDATE__NOT_PRIMARY_AS_IS
LARGE_HOLDOUT_ROLE = SECONDARY_CROSS_UPSTREAM_HOLDOUT
TRAIN_FORMAL_OVERLAP = 7_MOLECULES
DEV_FORMAL_OVERLAP = 0_MOLECULES
TRAIN_LARGE_HOLDOUT_OVERLAP = 0_MOLECULES
DEV_LARGE_HOLDOUT_OVERLAP = 0_MOLECULES
PROTECTED_FINAL_OUTCOME_READ = NO
```
