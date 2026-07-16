# ECIR-Flow versus the 4D refiners

## Previous role

The earlier family predicts per-bond or coupled 4D rates, optionally alongside
a Cartesian residual. Some historical training paths regress a Cartesian
residual projected or solved into Jacobian coefficients.

## ECIR role

ECIR combines:

- heterogeneous real upstream errors;
- controlled structured degradation;
- explicit bond/angle/torsion/ring/clash/chirality diagnostics;
- error-conditioned complete Cartesian refinement;
- heteroscedastic uncertainty;
- repair and identity gates;
- label-free trust regions;
- clean identity learning;
- a four-step teacher before any student distillation.

The default ECIR output is `delta_x in R^(N x 3)`. Global4D and Jacobians remain
unchanged in the repository, but are used only for corruption, diagnostic label
construction and ablation. ECIR does not ask a 4D head to fit the full Cartesian
error and does not average multiple reference coordinates.
