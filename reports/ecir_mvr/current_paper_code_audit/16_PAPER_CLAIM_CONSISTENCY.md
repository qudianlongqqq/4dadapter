# Paper claim consistency

PAPER_TEXT_AUDIT = NOT AVAILABLE

No manuscript, TeX, Word draft, or paper claim table was found. The only claim-oriented files are audit reports such as `21_GENERALIZATION_CLAIM_MATRIX.md` and `10_CLAIM_CORRECTION_AUDIT.md`. Therefore this file assesses safe/unsafe claim classes, not unseen manuscript wording.

| claim | classification | qualification/evidence |
|---|---|---|
| current method uses shared graph backbone, direct sigma, Reliability, Adaptive BA, and learned magnitude | SUPPORTED | executable modules and configs |
| J1 is beta-NLL with beta 0.5 and detached variance weighting | SUPPORTED | `belief_loss` |
| Restricted seed307 DEV V3D/PB = 0.564/0.9324 | SUPPORTED | frozen DEV artifacts |
| Unrestricted seed307 DEV V3D/PB = 0.567/0.9324 | SUPPORTED | frozen DEV artifacts |
| Restricted and Unrestricted are Pareto-near-tied | SUPPORTED_WITH_QUALIFICATION | seed307 development evidence only |
| Unrestricted is better | NOT_SUPPORTED | endpoint tradeoffs and multiseed incomplete |
| Restricted is better | NOT_SUPPORTED | endpoint tradeoffs and multiseed incomplete |
| method generalizes to unseen molecules | SUPPORTED_WITH_QUALIFICATION | same ETFlow source distribution only |
| cross-upstream generalization | NOT_SUPPORTED | not tested for current formulation |
| final unbiased test performance | NOT_SUPPORTED | DEV guided development |
| protected Formal/large-holdout performance | NOT_VERIFIABLE | outcomes not read |
| explicit autograd VJP implementation | OUTDATED_OR_INACCURATE | action uses analytic derivative accumulation |
| guaranteed improvement / monotonicity | NOT_SUPPORTED | zero fallback/cap and empirical outcomes do not provide guarantee |
| SOTA / universal / general-purpose | NOT_VERIFIABLE | no benchmark corpus or manuscript evidence |
| theoretical upper bound from tau=0.010 | NOT_SUPPORTED | human design bound, not theoretical bound |
| alpha_grad≈2.255e12 is a recommended objective coefficient | OUTDATED | invalidated by module-gradient forensic audit |

PAPER_CLAIM_RISKS = DEV_AS_TEST; SEED307_AS_MULTISEED; CROSS_UPSTREAM_OVERCLAIM; VJP_TERMINOLOGY; SOURCE_RMSD_AMBIGUITY; OBJECTIVE_ALPHA_SUPERSEDED


