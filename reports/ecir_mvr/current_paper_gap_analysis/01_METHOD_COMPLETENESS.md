# Method completeness

## Decision

```text
METHOD_COMPLETENESS = MOSTLY_COMPLETE
ARCHITECTURAL_BLOCKER_FOUND = NO
PIPELINE_OVERLY_ENGINEERED = PLAUSIBLE_REVIEWER_CONCERN
```

The current method is an end-to-end inference procedure, not a collection of disconnected heads. Its causal chain is:

`graph + Source coordinates -> graph representation -> Bond/Angle target means and scales -> Source defects -> source-conditioned Reliability -> graph-conditioned Bond/Angle allocation -> analytic primitive Jacobian-transpose accumulation -> rigid-mode removal -> graph-RMS normalization -> learned magnitude -> Proposal coordinates`.

That chain is complete enough to state as a paper method. The main gaps are evidentiary: some modules have not been independently isolated in the final formulation, and several constants remain human scientific design choices.

## Module-by-module closure

| operation | defined in current implementation | role in causal chain | duplication / legacy assessment | gap |
|---|---|---|---|---|
| Input | YES | frozen topology/categories plus Source geometry | no Reference at inference | no material gap |
| Graph representation | YES | encodes local molecular context for all learned heads | shared rather than duplicated | exact executed runner provenance is incomplete |
| Bond/Angle mu | YES | predicts graph-conditioned target primitive geometry | distinct physical families | necessity of each family is not isolated in final formulation |
| sigma | YES | heteroscedastic predictive scale and inverse-variance action factor | not equivalent to Reliability | calibration is not established |
| Reliability | YES | source-conditioned primitive action gate | complements graph-only sigma; not merely duplicate by code semantics | independent final-formulation contribution is not fully isolated |
| Adaptive BA | YES | molecule-level allocation between Bond and Angle families | differs from primitive Reliability | earlier isolated evidence was neutral; final gain is confounded with full-joint training |
| Cartesian direction | YES | analytic sum of Bond/Angle primitive derivative contributions | current reports must not call it an explicit autograd VJP | no torsion primitive; scope must be stated |
| Rigid projection | YES | removes translation/rotation components | no duplicate module | no matched deletion ablation |
| Graph-RMS normalization | YES | separates direction from learned magnitude | no duplicate module | constrains expression to direction plus scalar magnitude |
| tau magnitude | YES | graph-conditioned action size | not a second direction controller | supported at seed307, not yet multiseed |
| Restricted guards | YES | bound/cap/regularization for tail control | optional formulation bundle, not core direction logic | their individual causal effects are not isolated |
| Proposal | YES | `x_source + tau*d_hat`, with Restricted cap | final inference output | complete |

## Constants

`PREVIOUS_AUDIT_FACT`: the completed integrity audit counted 9 human scientific constants for Restricted and 6 for Unrestricted. The material choices are beta-NLL beta, sigma-ratio bound, Reliability initialization, equal-family aggregation, belief/post coefficients, initial tau, and—Restricted only—tau bound, atom cap, and movement-calibration form.

This is enough for a reviewer to call the system engineered, but not enough to establish a defect. A defensible presentation should group constants by role (probabilistic head, family balancing, movement safety) and distinguish numerical guards from scientific choices.

## Necessary qualifications

- `CODE_FACT`: sigma and Reliability consume different information and have different semantics. Treating one as redundant with the other is unsupported.
- `ARTIFACT_FACT`: neither sigma, Reliability, Adaptive BA, nor tau collapsed on seed307.
- `INFERENCE`: the architecture is coherent, but coherence is not proof that every component is necessary.
- `UNKNOWN`: whether a simpler final model could match all endpoints across seeds has not been established.

