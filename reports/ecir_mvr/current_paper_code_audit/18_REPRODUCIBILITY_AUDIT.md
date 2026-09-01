# Reproducibility audit

| element | status | evidence |
|---|---|---|
| frozen TRAIN/VAL/DEV split | PASS | hashes exist and match disk |
| seed | PASS for seed307 | checkpoint/config/status |
| config | PASS | exact SHA256 matches execution record |
| core model module | PASS | exact SHA256 matches execution record |
| executed runner source | FAIL | hash recorded but current file differs and no committed/archive source snapshot |
| checkpoint | PASS | exact SHA256 and metadata |
| optimizer/scheduler/recovery | PASS | checkpoint states and runner logic |
| RNG recovery | PASS | Python/NumPy/Torch/CUDA/generator states |
| environment | PARTIAL | CUDA version/device explicit for Unrestricted; Restricted exact execution environment linkage incomplete |
| evaluator | PARTIAL | artifacts and some versions/hashes present; no Git commit for all current evaluator code |
| result artifacts | PASS | hashes in final statuses |
| Git commit identity | FAIL | current paper path is untracked at HEAD |
| audit/report hashes | PASS | this audit manifest |
| seed331/353 | INCOMPLETE | active replication, no completed integrity audit |

```text
REPRODUCIBILITY_STATUS = PARTIAL
BYTE_EXACT_CORE_MODEL_RECONSTRUCTION = POSSIBLE
BYTE_EXACT_SEED307_RUNNER_RECONSTRUCTION = NOT VERIFIED
SCIENTIFIC_INPUT_IDENTITY = PASS
RESULT_ARTIFACT_IDENTITY = PASS
COMMIT_LEVEL_PROVENANCE = FAIL
```

The strongest remediation would be archival/version-control work, but this audit performs no fixes.


