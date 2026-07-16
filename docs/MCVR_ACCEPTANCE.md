# MCVR deterministic acceptance

Acceptance is label-free and never consults a reference RMSD. Both final-step
and best-of-trajectory modes require positive total validity gain and an
improved active local chemical mode. Candidates are rejected for increased
severe clash, chirality flip, stereocenter degeneracy, ring-bond or ring-
planarity degradation, molecule/atom trust excess, high-flex torsion excess,
or excessive uncertainty.

If no candidate passes, the exact input is returned. Metadata includes
`accepted`, `selected_step`, validity gain, displacement, torsion change,
uncertainty, `reject_reason`, and a `score_breakdown`.
