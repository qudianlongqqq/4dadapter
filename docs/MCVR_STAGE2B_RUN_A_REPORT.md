# MCVR Stage 2b Run A Report

## Decision

**RUN_A_PASS** on the frozen validation split. All 22 provisional Run A conditions passed. This result does not authorize automatic Run B, Run C, 20k, or 100k execution; `next_command` remains `null`.

The selected model is the step 3000 checkpoint, chosen only after it passed every accuracy non-inferiority gate and then achieved the largest total thresholded chemical-validity improvement among the five validated checkpoints.

## Identity and environment

- Branch: `feat/ecir-mvr-progressive`
- Preparation commit used for training: `27337cebb7250166a725ab6959dbdb56c6f420a1`
- Config SHA256: `aec111994c1a82db5627f26b8dfaba3ad890e810f4506596e7f97014bd894c85`
- Target gate identity: `6d73ccf9e1453134134ad27ba18bd3a1f8a2e76e49a72e0c464a7bd290f23ca7`
- Real-source identity: `e61f8eb7d29b1693688f6a1735bc5d1d5460ba99dec31702098c5eca9a6e7f7c`
- Validity identity: `66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3`
- GPU: NVIDIA GeForce RTX 5080; CUDA 12.8; PyTorch 2.11.0+cu128; seed 42
- Validation: 100 molecules / 130 records; test records read: 0

## Training and checkpoint selection

The run completed 5000 optimizer steps. Active training time was 676.173 seconds across the original and resumed segments. A numerical identity-diagnostic issue at step 2000 was corrected: clean candidates were exactly equal to their inputs, while self-Kabsch RMSD introduced sub-microångström residuals. Direct coordinate equality confirmed 20/20 unchanged controls at steps 1000 and 2000, after which the same model and optimizer state continued from step 2000.

All five checkpoints passed the frozen accuracy non-inferiority gate. Their total validity deltas were -0.009575, -0.034940, -0.078504, -0.058227, and -0.047827 at steps 1000 through 5000. Step 3000 was therefore selected.

- Selected checkpoint: `best_noninferior_validity.ckpt` / step 3000
- Selected checkpoint SHA256: `ac3e7e3b1fa4189e8ccdfeb45ea7c799a7130c213aeed017c301218b71487070`
- Last checkpoint SHA256: `76afae7f725d8395cb0b524367c725e7862f659e7ffe701adf0168e3db9cb89c`
- Train total loss: 0.205164 first, 0.131967 last, 0.103590 minimum
- Validation total loss: 0.140209 first, 0.118820 last/minimum

## Overall validation result

Molecule-level paired bootstrap used 1000 draws with seed 42. Run A accepted versus upstream:

| Metric | Delta | 95% CI |
|---|---:|---:|
| bond outlier rate | -0.023682 | [-0.031886, -0.016366] |
| bond outlier magnitude | -0.206987 | [-0.277623, -0.138999] |
| angle outlier rate | -0.000697 | [-0.001164, -0.000267] |
| ring bond outlier rate | -0.001998 | [-0.003964, -0.000417] |
| total thresholded validity | -0.078504 | [-0.104643, -0.054399] |
| aligned RMSD (Å) | +0.000521 | [+0.000357, +0.000697] |
| MAT-P (Å) | +0.000521 | [+0.000357, +0.000697] |
| MAT-R (Å) | +0.000706 | [+0.000486, +0.000947] |
| COV-P | 0.000000 | [0.000000, 0.000000] |
| COV-R | 0.000000 | [0.000000, 0.000000] |

Bond outlier rate improved by 10.72% relative to upstream, satisfying the fixed 10% core-validity condition. Severe clash and chirality error remained unchanged at zero. Diversity changed by -0.000805 (95% CI [-0.001143, -0.000516]), far below the fixed collapse tolerance.

## Source and flexibility slices

- ETFlow normal: total validity 0.294563 to 0.287699; RMSD 1.124615 to 1.124700 Å; COV unchanged; 38.57% accepted.
- Cartesian mild: total validity 1.783182 to 1.460378; RMSD 1.482337 to 1.484207 Å; COV unchanged; 95.45% accepted.
- Cartesian medium: total validity 2.619878 to 2.109823; RMSD 2.373591 to 2.376772 Å; COV unchanged; 100% accepted.
- Cartesian severe: unavailable in the frozen validation sources (0 molecules); no severe records were fabricated or substituted.
- Unseen Cartesian update scale 0.35: total validity 1.457912 to 1.212250; RMSD 1.541805 to 1.543344 Å; MAT/COV remained within the fixed non-inferiority limits.
- High-flex (at least 6 rotatable bonds): total validity 0.877791 to 0.736330; RMSD +0.000958 Å; high-flex torsion change 0.001484 versus 0.029430 for historical four-step ECIR.
- Clean validation-reference controls: 20/20 exact identity after deterministic acceptance.

## Acceptance and rigid-only checks

Record-level acceptance was 64.62%; molecule-level acceptance was 55.5%. Acceptance reduced validity-worsened fraction from 46% raw to 0%, and RMSD-worsened fraction from 97% raw to 53% accepted. It neither accepted nor rejected every candidate.

The torsion gate maximum and torsion velocity contribution maximum were both exactly zero. High-flex torsion changes were over an order of magnitude below historical ECIR. Improvements were driven by bond, angle, and ring validity rather than large torsional motion.

## Recommendation

The evidence supports `RUN_A_PASS`, but the workflow requires an explicit human decision before any Run B work. Do not start Run B automatically. The 20k and 100k gates remain closed, and `next_command` is `null`.
