# Restricted versus Unrestricted implementation diff

## Shared verified identity

Both use the same prepared/source payload hashes, TRAIN/VAL/DEV manifests, 3-layer 128D backbone, Bond/Angle mu heads, direct J1 sigma, beta 0.5, R1 Reliability, Adaptive BA, analytic primitive descent, rigid projection, graph-RMS normalization, belief/post coefficients, AdamW recipe, learning rates, weight decay, batch size, gradient clipping, scheduler horizon, 17,500-step endpoint, and seed307 initial geometry checkpoint.

## EXPECTED_DIFFERENCES

| aspect | Restricted | Unrestricted |
|---|---|---|
| total loss | belief + post + 0.4079342 move | belief + post |
| tau | 0.010*sigmoid(raw) | softplus(raw) |
| tau finite bound | 0.010 Å | none |
| initial output bias | logit for 0.003/0.010 | inverse-softplus for 0.003 |
| per-atom cap | 0.03 Å | none |
| proposal | capped source+tau*d | exact source+tau*d |
| rollback | none in scientific proposal; diagnostic safety flag only | none |
| movement loss | mean((tau/0.010)^2) | none |

The unrestricted magnitude hidden layers retain the same architecture/standard initialization; only the final parameterization/bias changes as required.

## UNEXPECTED_DIFFERENCES / operational asymmetries

- The Restricted runner accepts `cuda/cpu/auto` device modes; Unrestricted is GPU fail-closed. Current multiseed orchestration uses the verified CUDA environment, so this is an operational guard asymmetry, not evidence of differing executed compute for the ongoing run.
- Seed307 Restricted did not preserve a standalone GPU verification artifact; Unrestricted did.
- Current runner bytes differ from the seed307 executed runner hashes for both branches. This is a provenance issue affecting both, not a discovered formulation-specific scientific change.
- No extra hidden mask, alternative normalization, evaluator cohort, loss coefficient, optimizer recipe, or data-ordering difference was found in the current scientific paths.

```text
RESTRICTED_IMPLEMENTATION_VERIFIED = YES
UNRESTRICTED_IMPLEMENTATION_VERIFIED = YES
UNEXPECTED_SCIENTIFIC_DIFFERENCE_FOUND = NO_EVIDENCE
MULTISEED_FAIRNESS_DESIGN = MATCHED
MULTISEED_FAIRNESS_EXECUTION = INCOMPLETE
MULTISEED_FAIRNESS_RISK = INCOMPLETE_PROVENANCE_NOT_A_DETECTED_MISMATCH
```


