# Paper narrative gaps

| reviewer-level question | status | current answer and gap |
|---|---|---|
| 1. What is the problem? | ANSWERED | local post-generation refinement of fixed-topology Source conformers |
| 2. What defects do Source conformers have? | ANSWERED | Bond/Angle deviations and validity failures are measured; source-quality quintiles exist |
| 3. Why not directly use MMFF/xTB optimization? | PARTIALLY_ANSWERED | historical baselines show cost/movement tradeoffs, but no exact current-candidate matched comparison exists |
| 4. Why learned refinement? | PARTIALLY_ANSWERED | current seed307 improves local validity and xTB with small movement; final multiseed/baseline evidence is pending |
| 5. Why mu/sigma? | PARTIALLY_ANSWERED | mu defines defects and sigma scales them; J1 factorial supports the path, but calibration/fixed-sigma action control is missing |
| 6. Why Reliability? | ANSWERED | source-conditioned gate is semantically distinct; R0/R1 factorial gives matched seed307 support, though not exact final architecture |
| 7. Why Adaptive BA? | NOT_ANSWERED | final model uses it, but early isolated evidence was neutral and final gain is confounded |
| 8. Why learned magnitude? | ANSWERED | fixed-versus-learned magnitude comparison and noncollapsed tau support it at seed307 |
| 9. Why can tiny movement matter? | PARTIALLY_ANSWERED | local Bond/Angle/validity/xTB transitions show effect; a clear geometric sensitivity explanation and matched plots are needed |
| 10. Why study Restricted and Unrestricted? | ANSWERED | they isolate a practical safety/capability bundle and expose rare tail versus fewer constants |
| 11. What does the method improve? | ANSWERED | local geometry/V3D and GFN2 single-point energy; PB is nearly saturated; Reference RMSD is not materially improved |
| 12. What is the cost? | NOT_ANSWERED | current-candidate runtime, memory, and matched compute baseline are missing |
| 13. When should it not be used? | PARTIALLY_ANSWERED | fixed topology/local-refinement limits are inferable but must be explicit; cross-upstream behavior is unknown |
| 14. What is the evidence boundary? | ANSWERED | same-upstream unseen-molecule seed307 DEV only; multiseed/protected/cross-upstream are pending or unknown |

The most important narrative correction is to make local refinement—not global Reference recovery—the central claim. The second is to describe sigma and Reliability by their code semantics rather than by stronger probabilistic or causal labels.

