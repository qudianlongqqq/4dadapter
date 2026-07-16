# Cartesian 100k source audit

Twenty fixed Confirm30 validation molecules were audited with rollout lengths `0,1,2,4,8,10` and update scales `0.05,0.10,0.20,0.50,1.00`. The audit generated 600 rollout rows, 100 SDF snapshots (`0/1/2/4/10` at scale 0.5), and atom displacement tables. Test data was not used.

## Identity and implementation checks

- checkpoint SHA256: `600d312328b31ab85ba13183f4db0f37951054c753dfacc024b6aeed334f973e`; global step 100000.
- config SHA256: `2e72151e3f6a149526f31050c4eaef3a99653ab97d0a21a08d1525557b1c9714`.
- strict Cartesian model loading and checkpoint/config architecture comparison passed.
- molecule ID, sample ID, `x_init_hash`, checkpoint/config/manifest identity, shape, atom count/order, hydrogen count and coordinate scale passed for all 20 molecules.
- the formal cached `x_cart` was reproduced with maximum absolute coordinate delta `1.91e-6`.
- update-scale linearity maximum drift was `7.63e-6` (float32); scale is applied once, not twice.
- Kabsch alignment is used only for metrics and displacement diagnostics; aligned coordinates are not fed back into rollout.
- the Stage 2 cache does not persist selected reference index/ID. The exact persisted `x_ref_aligned` tensor is SHA-bound in this audit, but the missing upstream reference identifier is a provenance warning.

## Scale 0.5 trajectory

| steps | bond error | angle error | torsion error | ring error | aligned displacement (Å) | max atom displacement (Å) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.01396 | 0.02508 | 0.06089 | 0.01555 | 0.00000 | 0.00000 |
| 1 | 0.02844 | 0.03607 | 0.06471 | 0.03992 | 0.04806 | 0.05331 |
| 2 | 0.03276 | 0.03887 | 0.06624 | 0.04291 | 0.05627 | 0.08521 |
| 4 | 0.05427 | 0.05757 | 0.07708 | 0.05964 | 0.09618 | 0.16988 |
| 8 | 0.11707 | 0.08846 | 0.09210 | 0.09898 | 0.20416 | 0.35035 |
| 10 | 0.14518 | 0.09943 | 0.10313 | 0.11016 | 0.25511 | 0.44315 |

The model was trained with `t in [0, 0.25]`, while `FlexBondOptimizerLightningModule.refine` evaluates `t=step/(steps-1)`, reaching `1.0` for every rollout longer than one step. Thus later calls extrapolate far outside the trained time range. Internal distortion and displacement rise monotonically with rollout length even though nearest-reference RMSD happens to fall in this 20-molecule subset.

## Classification and decision

Classification: **B — single-step is materially safer, while multi-step rollout diverges in internal geometry.** The proximate cause is inference-time extrapolation beyond the trained time range, not data mapping, units, duplicate scale application, duplicate cache parsing, or checkpoint mismatch.

The formal 10-step Cartesian rollout must not continue to represent ordinary real-error input. Existing frozen data remain untouched for audit reproducibility; Stage C must exclude or sharply down-weight this source and stratify it by severity. No retraining is authorized yet.
