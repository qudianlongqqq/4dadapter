# MCVR loss

`MCVRLoss` reports nine independent terms plus their weighted total:

- Cartesian flow matching to the minimal-validity target;
- active-mode-only validity supervision;
- clean identity velocity;
- input anchor;
- affected-atom sparsity;
- inactive/high-flex torsion anchoring;
- six-mode error auxiliary classification;
- difficulty uncertainty regression;
- atom and molecule trust excess.

An inactive `active_mode_mask` contributes exactly zero validity-mode loss.
Synthetic targets are original clean coordinates and their affected atoms are
persisted. Real targets use the deterministic anomaly mask, while clean inputs
have exact identity targets.
