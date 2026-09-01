# Git provenance status

## Pre-freeze state

```text
REPOSITORY = E:/3dconformergenerationcode/4dadapter-lsgoba-musigma-reliability-factorial
BRANCH = experiment/sixs-musigma-reliability-factorial
PRE_FREEZE_HEAD = 9be3633a0af930b29704a29d432327340872de9b
TRACKED_MODIFICATIONS = NONE
UNTRACKED_CONTENT = PRESENT
```

The untracked content contains current paper code/reports as well as large checkpoints, generated artifacts and temporary xTB working directories. Because there were no tracked modifications, a precise whitelist can be committed safely without mixing unrelated or large generated content.

## Freeze policy

The freeze commit includes:

- active core source introduced by the final method;
- Restricted/Unrestricted configs and small run-local frozen configs;
- active runners/evaluators/statistical finalizer;
- the DEV identity manifest;
- small authoritative status/config/audit artifacts;
- the current code audit, gap analysis, post-multiseed audit, evidence-matrix audit and this freeze package.

It explicitly excludes:

- all `.phase1_xtb_mmff_sp`, `.sixs_final_multiseed_xtb_sp` and `.unrestricted_xtb_sp` temporary trees;
- large checkpoint binaries and generated model/evaluator artifact trees;
- stdout/stderr/traceback logs and large per-record tables not needed in Git;
- unrelated historical experiment directories.

Excluded authoritative checkpoints and external dependencies are bound by stable path, byte size, SHA256 and metadata in the release/checkpoint manifests.

The immutable symbolic reference is `refs/tags/final-development-freeze-step1`. It resolves to the exact final commit without requiring the commit to contain a self-referential copy of its own SHA.

## Historical provenance boundary

Seed307 historical executed runner bytes cannot be reconstructed from the current repository snapshot:

```text
RESTRICTED_SEED307_EXECUTED_RUNNER_SHA256 = c8b336e9b117f30230cf63ab7be6e8dec41030981fd14266997d52c73d845dcf
RESTRICTED_CURRENT_FROZEN_RUNNER_SHA256 = 81bb200a7118ea27c3fe1b80f11bf14e3245d30514cfdbeeb750b16f9f0796dc
UNRESTRICTED_SEED307_EXECUTED_RUNNER_SHA256 = eee8c6426e5c7af2b77fda979925b615bad9d564f605cb98da1ca2b407dba1b3
UNRESTRICTED_CURRENT_FROZEN_RUNNER_SHA256 = 37f932565651e159ffc4047446b2d3d14d5397785ba365ce9797ff039352eff1
HISTORICAL_SEED307_RUNNER_PROVENANCE = PARTIAL
```

Core modules, configs, checkpoint metadata and result artifacts remain hash-bound. The historical limitation is retained and not rewritten as PASS.

```text
GIT_FREEZE_STATUS = PASS_BY_EXPLICIT_WHITELIST_COMMIT_AND_TAG
FORWARD_PROVENANCE_FROZEN = YES
```
