# LSGO-BA formal-large inheritance audit

The formal method is the frozen `LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=false)` with a shared invariant message-passing encoder, neural Bond and cosine-Angle conditional-mean heads, and the frozen DRCSR Reference scale. It has 473,674 trainable parameters. Bond and Angle primitive Gaussian NLL means are aggregated equally. No Cartesian coordinate target, MVT teacher, learned sigma, Torsion, Ring objective, soft Clash, xTB or PoseBusters term is present.

Training inherits AdamW (`lr=3e-4`, `weight_decay=1e-6`), cosine scheduling, gradient clipping at 1.0, and uniform sampling of one Reference per sampled molecule. Formal effective batch is 64 and the frozen horizon is 12,500 optimizer steps.

Formal TRAIN contains 50,000 molecules / 150,000 Source identity records and VALIDATION contains 5,000 / 10,000; molecule overlap is zero. LSGO does not train against those Source coordinates: it consumes Reference local Bond/Angle geometry. Therefore 800,000 draws are 16.0 molecule-equivalent epochs; 5.33 is retained only as the historical `800,000/150,000` record-accounting convention.

The Reference scales remain frozen from 2,000 development TRAIN molecules (46,581 References), SHA `ca4203914a42314ebd4bcb985a4c1ef83691d1b16ad41a22d38286cba23124b3`. They are not refitted on formal-large.

Checkpoint payloads must contain model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, sampler generator, exposure and validation identities and must strict-resume. Selection is frozen to lowest full-validation joint BA NLL, then calibration error, then earlier step. External xTB/PoseBusters are prohibited for selection.

Formal test reads = **0**. Frozen holdout reads = **0**.
