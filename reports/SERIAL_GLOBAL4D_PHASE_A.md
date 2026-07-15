# Serial Global4D Phase A

Phase A **PASS**. Both prescribed 2k experiments completed without NaN, OOM,
or gradient failure. Validation-only checkpoint selection chose auto-batch
step 1500, which had the highest positive-gain fraction among stable passing
checkpoints:

- internal MSE: `0.501709 < 0.506037` zero predictor
- internal cosine: `0.109482`
- positive / negative raw gain: `60% / 40%`
- predicted/target norm ratio: `0.119846`
- mean q norm: `0.030111`
- mean raw gain: `+0.984270`

The fair bs8 run also produced a passing checkpoint at step 1750. The auto96
run saw far more records per fixed 2k optimizer-step budget and took 797.6 s
versus 108.3 s for bs8; this is a throughput/capacity experiment, not a claim
that the two runs saw equal records.

Selected external checkpoint:
`E:/3dconformergenerationcode/serial_global4d_work/logs/phase_a_pilot_2k/bs_auto96/step001500.ckpt`
(SHA256 `d67179d2ffbad97180223291bf13ca589f0c257bb31b5a6f139e4a58580007b1`).

No rescue hyperparameter run was needed.
