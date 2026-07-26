# Historical inheritance audit

The audit inherits, but does not re-select from, the two prospective LSGO-B results:

- historical fresh external: median ΔE ≈ −0.8045 kcal/mol; 93.61% improved;
- BAT fresh replication: median ΔE ≈ −0.9047 kcal/mol; 93.50% improved; p95/p99 0; mean RMS ≈ 0.0028 Å; mode switch 0%; topology/chirality 100%.

The frozen BAT decision at base HEAD was `KEEP_LSGO_B`. The K3 torsion model learned Reference distributions (DEV NLL 1.454–1.483 versus uniform 1.8379), but its Source>Reference surprise fraction 0.502–0.534 failed source selectivity. The valid inherited statement is therefore “the Reference marginal-likelihood detector failed,” not “torsion is useless.”

Soft steric repair remains paused: it changed continuous penetration but produced zero fresh PoseBusters clash transitions. The unchanged hard steric guard remains a do-no-harm safety check.

The BAT branch/worktree was clean at `a2a4b5d65cd358caaba6185121bb7e62aac8d2ae`; this audit uses a separate branch/worktree. No historical result was modified.
