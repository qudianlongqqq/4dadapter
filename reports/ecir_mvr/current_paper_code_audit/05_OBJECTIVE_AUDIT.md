# Exact objective audit

## Belief/J1

For each primitive:

[
operatorname{NLL}(y,mu,sigma)=	frac12((y-mu)/sigma)^2+logsigma.
]

J1 uses

[
L_{belief}=operatorname{MolMean}left[
operatorname{NLL}cdotoperatorname{stopgrad}(sigma^2)^{0.5}
ight].
]

`MolMean` first averages primitives within Bond and Angle, uses 0.5/0.5 family aggregation per molecule, then averages molecules. Evidence: `etflow/ecir/musigma_reliability.py:239-253,256-285`. `beta=0.5` is present in both configs and runner calls.

## Post loss

[
L_{post}=operatorname{MolMean}left[
((q^{prop}_{bond}-q^{ref}_{bond})/operatorname{stopgrad}sigma_{bond})^2,,
((q^{prop}_{angle}-q^{ref}_{angle})/operatorname{stopgrad}sigma_{angle})^2
ight].
]

Evidence: `musigma_reliability.py:449-466`. Reference labels enter training loss only, not inference features.

## Restricted

[
L_{total}=L_{belief}+L_{post}+0.40793421960700144 L_{move}
]
[
L_{move}=operatorname{mean}[(	au/0.010)^2].
]

Evidence: `run_sixs_j1r1_full_joint_adaptive_ba_movement.py:274-290` and restricted config. Tau is `0.010*sigmoid(raw)`; proposal is capped at 0.03 Å per atom. The cap is in forward geometry, not an extra loss. The movement coefficient is recorded as a TRAIN-only 5%-of-initial-median-post-loss rule; its derivation was not recomputed here.

## Unrestricted

[
L_{total}=L_{belief}+L_{post}.
]

Evidence: unrestricted runner `:167-179`. Tau is `softplus(raw)`, initialized by inverse softplus at 0.003 Å; there is no finite tau upper bound, movement regularizer, per-atom cap, or rollback.

## Gradient routing

| module | belief | post | move (restricted) |
|---|---|---|---|
| backbone | primary | microscopic/indirect at initialization | no |
| Bond/Angle mu heads | yes | detached/no | no |
| sigma heads | yes | detached/no | no |
| Reliability | no | yes | no |
| Adaptive BA | no (detached input boundary) | yes | no |
| magnitude | no (detached input boundary) | yes | yes |

The later forensic audit supersedes the earlier global common-backbone `alpha_grad≈2.255e12` recommendation: effective shared modules were empty at scientific initialization and belief/post primarily update specialized module sets. Evidence: `sixs_objective_module_gradient_forensic_audit/FINAL_STATUS.json`.

```text
LAMBDA_BELIEF = 1.0
LAMBDA_POST = 1.0
BETA_NLL_BETA = 0.5
BETA_ORIGIN = HISTORICAL_DESIGN
OBJECTIVE_IMPLEMENTATION_VERIFIED = YES
GLOBAL_ALPHA_GRAD_AS_TRAINING_COEFFICIENT = INVALIDATED
OBJECTIVE_RATIO_SENSITIVITY_PRIORITY = MEDIUM
```


