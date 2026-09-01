# Multiseed results status

## Completed runs

| formulation | seed | status | scientific result |
|---|---:|---|---|
| Restricted | 307 | COMPLETE | frozen seed307 metrics available |
| Unrestricted | 307 | COMPLETE | frozen seed307 metrics available |
| Restricted | 331 | INCOMPLETE at snapshot | excluded |
| Unrestricted | 331 | INCOMPLETE | unavailable |
| Restricted | 353 | INCOMPLETE | unavailable |
| Unrestricted | 353 | INCOMPLETE | unavailable |

The pre-existing process snapshot showed Restricted seed331 training and its run-local status at step 2000. This is engineering progress only.

## Permitted conclusion

```text
MULTISEED_STATUS = INCOMPLETE
THREE_SEED_MEAN_SD = NOT AVAILABLE
MATCHED_SEED_DIFFERENCE_331 = NOT AVAILABLE
MATCHED_SEED_DIFFERENCE_353 = NOT AVAILABLE
PARETO_NEAR_TIE = SEED307_ONLY
REPLICATED_OR_MIXED_OR_NOT_REPLICATED = NOT YET CLASSIFIABLE
MULTISEED_INTEGRITY = NOT YET AUDITED
```

No partial loss, checkpoint, or DEV metric is used for formulation selection.


