# Torsion model audit

The primary model is a 3-component von Mises mixture head over frozen BA node embeddings. It learns π and circular μ only; κ is fixed after internal calibration. Maximum primary parameter count: `83337` (<0.5M). BA is never fine-tuned. Decision: **TORSION_NO_GO**.

The one-seed learned-κ diagnostic did not hit its floor or ceiling (DEV κ range approximately 1.65–14.21), but it was diagnostic only and cannot override the failed Source-selectivity gate. No learned uncertainty route is promoted.
