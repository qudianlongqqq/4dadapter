# MCVR Medium Velocity Raw vs Clipped Audit

Decision: **POST_CLIP_THRESHOLD_SELF_TRIGGER**

The V2 stop value `0.0600000023841857` came from `v_final`, after trust clipping and the global safety gate. It was not a raw-velocity measurement.

| Metric | Value |
|---|---:|
| raw_velocity_atom_mean | 0.025976585224 |
| raw_velocity_atom_p95 | 0.059676438570 |
| raw_velocity_atom_max | 1.468416213989 |
| raw_velocity_graph_rms | 0.258345931768 |
| clipped_velocity_atom_mean | 0.015500654466 |
| clipped_velocity_atom_p95 | 0.059676438570 |
| clipped_velocity_atom_max | 0.120000004768 |
| clipped_velocity_graph_rms | 0.059970196337 |
| final_output_velocity_atom_mean | 0.014487337321 |
| final_output_velocity_atom_p95 | 0.059676274657 |
| final_output_velocity_atom_max | 0.119999997318 |
| final_output_velocity_graph_rms | 0.059970196337 |
| graph_clip_scale | 1.000000000000 |
| atom_clip_scale | 0.081720694900 |
| graph_clipped_fraction | 0.000000000000 |
| atom_clipped_fraction | 0.024128686637 |

The frozen trust clipping calculation is bitwise identical in the model, the legacy helper, and the audit reconstruction. The monitoring comparison, not the clipping mathematics, caused the stop.

The step2450 checkpoint is complete and strictly resumable: `736cbe38a44396ed6d4c0da0af017b7f7cd622d333b02a07d225d9d0bc2e7b1e`.
It contains model, optimizer, global step, RNG, sampler, timing, and frozen identities.

No test split, seed43/44, or 100k artifact was read or started.
