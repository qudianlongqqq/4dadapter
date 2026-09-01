# Theory / implementation alignment

| interpretation or claim | classification | safe wording / issue |
|---|---|---|
| Analytic primitive Jacobian-transpose accumulation | SAFE | the code explicitly accumulates analytic Bond/Angle derivative contributions |
| Explicit autograd VJP | UNSAFE | current scientific action code does not obtain the coordinate direction through an explicit autograd VJP |
| First-order local action | SAFE_WITH_QUALIFICATION | the action uses first derivatives of primitive geometry, but no convergence theorem or iterative optimizer is established |
| Gauss–Newton method | UNSAFE | no normal-equation solve, Hessian approximation, or Gauss–Newton convergence analysis is present |
| Formal optimization algorithm | UNSAFE | one learned proposal step is not automatically an optimizer |
| sigma as calibrated uncertainty | UNKNOWN | heteroscedastic predictive scale/NLL evidence exists; calibration evidence was not found |
| sigma as inverse-variance action factor | SAFE_WITH_QUALIFICATION | this is exact code semantics; statistical precision interpretation requires calibration qualification |
| Reliability as source-conditioned action reliability | SAFE_WITH_QUALIFICATION | it is a learned sigmoid source-conditioned gate; causal utility/probability calibration is not proven |
| Adaptive BA as learned family allocation | SAFE | two softmax weights allocate Bond/Angle families |
| Energy-improving method | SAFE_WITH_QUALIFICATION | empirical GFN2 single-point DeltaE improves for nearly all seed307 records; energy is not optimized or guaranteed |
| Rigid invariance | SAFE_WITH_QUALIFICATION | translation/rotation components are projected out numerically; do not imply all representation choices are invariant without exact tests |
| General conformer recovery | UNSAFE | Reference RMSD is essentially unchanged and no torsion primitive is used |

The theory should present a learned, source-conditioned, first-order local refinement map. It should not borrow stronger language from optimizer families unless the missing mathematical operations and guarantees are actually supplied.

