# MCVR Stage D2 Angle and Ring Damage

Angle damage records: `246`; ring-bond damage records: `36`.

Angle damage involving a ring bond: `0.170731707317`.

Adjacent same-direction bond changes: `0.752032520325`.

Wrong-sign predictions among angle damage: `0.174796747967`; ring damage: `0.111111111111`.

Non-ring-only angle damage fraction: `0.829268292683`; the local comparison mode is `v4_record_screen_upstream_local_reference`.

D0 did not show the same damage because it solved the complete target residual once as a globally consistent minimum-norm correction. D1-B repeatedly combines approximate learned residuals with a separately learned Cartesian branch, then applies nonlinear safety and acceptance.

Ring/non-ring quantitative results are in `ring_nonring_summary.csv`; local records are in `angle_ring_damage_records.csv`.
