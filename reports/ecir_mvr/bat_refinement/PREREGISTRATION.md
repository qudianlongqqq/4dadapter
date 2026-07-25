# BAT formal preregistration

Status: **FROZEN**

- Eligible prospective candidate: **BA+C** (minimum validated mechanism).
- BA remains the exact three-seed frozen LSGO-B anchor; C is analytic and has no trainable parameters.
- BA+T and BAT+C are stopped because `TORSION_NO_GO` failed Source selectivity.
- One direct-gradient step; graph RMS ≤0.003 Å; atom cap ≤0.03 Å.
- Fresh external conditions are frozen as Source plus paired BA and BA+C for seeds 173/181/193.
- Coordinates will be frozen once before the first PB/xTB access. No post-external selection or retuning is allowed.

Formal test reads=0; frozen holdout reads=0.
