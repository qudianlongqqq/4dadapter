# MCVR Stage 2b Run B Report

## Decision

Run B is **`RUN_B_HARMS`**. It remains non-inferior to upstream on all 22 frozen Run A conditions, but does not add measurable torsion-prior benefit and is worse than Run A on total validity, angle validity, ring safety, high-flex validity, and unseen validity. Active torsion repair is therefore not suitable as the main method for the medium run.

The final Gate 1 selection is Run A rigid-only. No Run C, 20k training, 100k training, or test evaluation was run.

## Frozen experiment

- Branch: `feat/ecir-mvr-progressive`
- Preparation commit: `295d788af660d736f8db328bd8e879d74eb64a7c`
- Run B config SHA256: `e273c8b026ba4db0b2526ad755da7e0da2bc5b94d74d65bd9619cefd3ced27d0`
- Seed: 42
- Train/validation molecules: 500/100
- Validation records/molecules: 130/100
- Teacher steps: 4
- Device: NVIDIA GeForce RTX 5080; CUDA 12.8; PyTorch 2.11.0+cu128; Python 3.11.15
- Test records read: 0

Frozen identities were unchanged:

- Minimal target: `6d73ccf9e1453134134ad27ba18bd3a1f8a2e76e49a72e0c464a7bd290f23ca7`
- Real source: `e61f8eb7d29b1693688f6a1735bc5d1d5460ba99dec31702098c5eca9a6e7f7c`
- Validity statistics: `66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3`
- Run A selected checkpoint: `ac3e7e3b1fa4189e8ccdfeb45ea7c799a7130c213aeed017c301218b71487070`

## Training outcome

Run B used the frozen conservative torsion settings: normal/high-flex torsion scales 0.10/0.05 rad, trust limits 0.035/0.020 rad, strict evidence and safety gating, and the unchanged Run A rigid branch and non-inferiority margins.

Training stopped legitimately at step 3000 after 400.558 seconds. The registered reason was `no_incremental_validity_with_displacement_growth`. Saved checkpoints at steps 1000, 2000, and 3000 were evaluated; steps 4000 and 5000 were intentionally not run after the early-stop condition fired.

| Step | Checkpoint SHA256 | Total-validity delta vs Run A | Torsion-prior delta | High-flex validity delta | Upstream/incremental accuracy gates |
|---:|---|---:|---:|---:|---|
| 1000 | `f57b05eedd77d77d385a37f76fa84f069a932f195eb9ab09851a31b1c3450820` | +0.071066 | +0.002341 | +0.135578 | pass/pass |
| 2000 | `88bbecbafdbf2796c99d7f7f594a570ce1dcfe4b3325ac9c8f8d4eecfd30e57b` | +0.065786 | -0.000337 | +0.122193 | pass/pass |
| 3000 | `6a43ceb2ed66e5c2906e378eaad6c537a8b88b2236938b2ffc2953025af0179a` | +0.014673 | 0.000000 | +0.032395 | pass/pass |

Lower validity scores are better. Step 3000 was the best eligible incremental checkpoint and was used for the final comparison.

The additional losses were finite and recorded. From first to last validation measurement: torsion-mode loss 0.040075 to 0.035831, torsion-anchor loss 0.0000170 to 0.0000126, torsion-gate sparsity loss 4.78e-7 to 9.84e-8, and high-flex torsion-trust loss remained 0. Total validation loss moved from 0.143566 to 0.130611.

## Run B versus upstream

Run B passed every frozen upstream condition. Paired molecule bootstrap results (Run B minus upstream; 1000 draws) included:

| Metric | Delta | 95% CI |
|---|---:|---:|
| Total thresholded validity | -0.063831 | [-0.083490, -0.044736] |
| Bond outlier rate | -0.022124 | [-0.028724, -0.015882] |
| Angle outlier rate | -0.000081 | [-0.000242, +0.000079] |
| Ring bond outlier rate | -0.001250 | [-0.002917, 0.000000] |
| Aligned RMSD | +0.000431 Å | [+0.000314, +0.000557] |
| MAT-P | +0.000431 Å | [+0.000314, +0.000557] |
| MAT-R | +0.000537 Å | [+0.000394, +0.000692] |
| COV-P / COV-R | 0 / 0 | [0, 0] / [0, 0] |

## Torsion behavior and safety

- Torsion gate mean: 1.8538e-7
- Torsion gate active fraction: 0.03
- Torsion velocity fraction: 8.5297e-8
- Accepted mean/p95 torsion change: 0.000173/0.000930 rad
- High-flex mean/p95 torsion change: 0.001116/0.004359 rad
- High-flex limits: mean <= 0.010 rad; p95 <= 0.030 rad
- Clean identity controls: 20/20 exact identity
- Severe clash and chirality deltas vs Run A: exactly zero

The gate was very sparse and its velocity contribution was negligible. It did not change the selected checkpoint's torsion-prior score. Relative to Run A, angle outlier rate had a strictly positive CI and two molecules worsened on ring bond outliers; that incremental ring regression makes the registered safety condition fail even though absolute severe clash and chirality remained zero.

Cartesian severe and non-ring groups contained zero records and are reported as unavailable. Clean controls are evaluated separately from the main source summary and remained 20/20 exact identity.

## Artifacts

The complete three-way result, molecule-paired bootstrap outputs, source/flexibility/torsion summaries, record metrics, molecule metrics, checkpoint comparison, and loss summary are under `diagnostics/ecir_mvr/stage2b/run_b/`.
