# Repository identity

## Git snapshot

| field | value | evidence type |
|---|---|---|
| REPOSITORY_ROOT | `E:/3dconformergenerationcode/4dadapter-lsgoba-musigma-reliability-factorial` | GIT_FACT |
| GIT_BRANCH | `experiment/sixs-musigma-reliability-factorial` | GIT_FACT |
| HEAD_COMMIT | `9be3633a0af930b29704a29d432327340872de9b` | GIT_FACT |
| HEAD_COMMIT_TIME | `2026-08-16T14:01:55+08:00` | GIT_FACT |
| HEAD_COMMIT_MESSAGE | `Add unified Softplus multiseed development evaluation` | GIT_FACT |
| WORKTREE_CLEAN | NO | GIT_FACT |
| STAGED_FILES | 0 | GIT_FACT |
| MODIFIED_TRACKED_FILES | 0 | GIT_FACT |
| UNTRACKED_PATHS | 50 top-level status entries | GIT_FACT |

The current paper-path implementation, configs, runners, artifacts, reports, and audit scripts are untracked at HEAD. Relevant examples include `etflow/ecir/j1r1_full_joint.py`, `etflow/ecir/j1r1_full_joint_unrestricted.py`, `etflow/ecir/musigma_reliability.py`, both current configs, and the seed307 result directories.

## Version relation

**CODE_VERSION.** HEAD above plus the dirty/untracked worktree snapshot. Core current hashes are:

- restricted module `5e20d34ef6a8b92f40a22da02148d7451b2476fd0dfb9b06f864081c6d968765`;
- unrestricted module `22044fc13f7d54074c179b7029f44a4f7d533a6f8cb64cda61e1a4e25c3d0b77`;
- direct mu/sigma/reliability module `058e3da07308a7178e267230e85acd774ce95d9007eec3d05764b64f4ba7cb7d`;
- current restricted runner `81bb200a7118ea27c3fe1b80f11bf14e3245d30514cfdbeeb750b16f9f0796dc`;
- current unrestricted runner `37f932565651e159ffc4047446b2d3d14d5397785ba365ce9797ff039352eff1`.

**RESULT_VERSION.** Seed307 execution is identified by artifact hashes, not a Git commit:

- restricted config `3cf2db...`, module `5e20d3...`, executed runner `c8b336...`, checkpoint `7c2bb67a...`;
- unrestricted config `7c5c8a...`, module `22044f...`, executed runner `eee8c642...`, checkpoint `f63e9d79...`.

The current config and core-module hashes match the execution records. Both current runner hashes differ from the executed runner hashes. The later runners contain multiseed parameterization, but the exact byte-level executed runner sources are not represented by HEAD and are not available as a source snapshot.

**AUDIT_VERSION.** This read-only snapshot, identified by the hashes in `FINAL_AUDIT_STATUS.json` and `SHA256SUMS.txt`.

```text
CODE_RESULT_VERSION_MISMATCH = YES
CODE_RESULT_VERSION_MATCH = NO
MISMATCH_SCOPE = GIT_COMMIT_IDENTITY_AND_EXECUTED_RUNNER_BYTES
CORE_MODULE_AND_CONFIG_HASH_MATCH = YES
SCIENTIFIC_RESULT_INVALIDATED_BY_THIS_FACT = NOT ESTABLISHED
EXACT_REPRODUCIBILITY_IMPACT = HIGH
```

The mismatch is a provenance/reproducibility finding. It is not evidence that frozen metrics were calculated incorrectly.


