# MCVR model

`MCVRModel` is an equivariant Cartesian model whose final output is always an
`N x 3` velocity. Existing ECIR classes remain untouched for old-checkpoint
compatibility.

The model combines a shared Cartesian backbone with rigid/local and
torsion/flexible Cartesian heads. Graph context includes frozen deterministic
error features, the learned error embedding, uncertainty features, and
optional source metadata with dropout. Separate rigid, torsion, and global
safety gates form

`v_final = safety_gate * trust_clip(rigid_gate*v_rigid + torsion_gate*v_torsion)`.

Default torsion scale is 0.25 versus rigid scale 1.0; high-flex torsion scale
is 0.125. With no deterministic torsion excess the torsion gate is exactly
zero. Trust clipping limits norms without deleting Cartesian directions. No
Strict Global4D fusion or default four-dimensional output head is present.
