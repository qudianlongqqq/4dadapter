# MCVR Medium Seed42 Report

## Outcome

The registered decision is **`MEDIUM_SEED42_FAIL`**. The authorized rigid-only medium run started from scratch but stopped at optimizer step 2000, before the first scheduled 5000-step validation checkpoint, because `velocity_norm_sustained_growth` fired. The run was not resumed. Seed43, seed44, test evaluation, and 100k training were not started.

## Frozen data and target provenance

- Train/validation molecules: 5000/500; overlap: 0
- Real-source records: 7500/700
- Validation severity records: normal 464, mild 134, medium 87, severe 15
- Validation ring/non-ring records: 685/15; high-flex records: 511
- Test paths opened/read: 0
- Validity statistics identity: `66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3`
- Medium real-source identity: `2b3e1f414e67f95395c9988bc1d03a5e60fefd2fa04a0b0723dc20273a28aa5d`
- Medium target identity: `5df1cace6af98279b6b9ab85d993a0a3f674210c3afd8b10a264268e26904dc1`
- Config SHA256: `eaf8d31a60a7f0cb07e7dc282067c43baa0c662f4a1f1ce06ba52e7f69dfdcd3`
- Preparation commit used for training: `9c2454d423e5b987d83861e6e34320aad7ac07f9`

The Stage C-equivalent target pilot and full target audit passed. Across 8200 target records, mean/p95 aligned displacement was 0.011582/0.035163 Å, maximum atom displacement was 0.085925 Å, high-flex mean maximum torsion change was 0.021974 rad, and mean validity gain was 0.630083. All 611 identity fallbacks and all 14 clean targets returned exact input coordinates.

## Training

The model was initialized from scratch with the frozen Run A architecture and loss. Torsion repair, torsion gate, torsion scale, high-flex torsion scale, and torsion velocity remained exactly zero.

Training ran for 309.223 seconds. The final five diagnostic windows were:

| Step | Total loss | Rigid gate | Safety gate | Velocity norm | Molecule displacement |
|---:|---:|---:|---:|---:|---:|
| 1800 | 0.123856 | 0.05816 | 0.45957 | 0.001196 | 0.000356 |
| 1850 | 0.131280 | 0.06537 | 0.43619 | 0.001827 | 0.000453 |
| 1900 | 0.114988 | 0.05166 | 0.29256 | 0.002151 | 0.000618 |
| 1950 | 0.116631 | 0.07284 | 0.31867 | 0.004522 | 0.001400 |
| 2000 | 0.133720 | 0.06783 | 0.27909 | 0.006158 | 0.001704 |

Velocity norm increased at all five points and the last value exceeded twice the first, satisfying the pre-registered early-stop rule. The only checkpoint is `last.ckpt` at step 2000, SHA256 `4a82898156ffddebfab9ad409d1cd68c4890f2bf7116ff20da63831daa9030e3`. It is an early-stop diagnostic checkpoint, not an eligible scheduled validation checkpoint or `best_noninferior_validity` checkpoint.

## Gate 2 evaluation of the stopped model

The same 700 validation records and 500 molecule aggregation were used for upstream, Stage B rescued, Stage 2b Run A, the medium raw/accepted model, and the minimal-target upper bound. All deltas below are accepted medium minus upstream.

| Metric | Upstream | Medium accepted | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Total thresholded validity | 0.793989 | 0.777601 | -0.016388 | [-0.019010, -0.014085] |
| Bond outlier rate | 0.262032 | 0.257236 | -0.004796 | [-0.005627, -0.004004] |
| Angle outlier rate | 0.031131 | 0.030729 | -0.000401 | [-0.000627, -0.000196] |
| Ring bond outlier rate | 0.153504 | 0.151181 | -0.002323 | [-0.003149, -0.001588] |
| Aligned RMSD | 1.321787 | 1.321806 | +0.000018 Å | [+0.000006, +0.000031] |
| MAT-P | 1.321787 | 1.321806 | +0.000018 Å | [+0.000006, +0.000031] |
| MAT-R | 2.375986 | 2.376003 | +0.000017 Å | [-0.000002, +0.000037] |
| COV-P | unchanged | unchanged | 0 | [0, 0] |
| COV-R | - | - | -0.000067 | [-0.000200, 0] |

The stopped model passed all accuracy margins, clean identity was 20/20, acceptance was 0.742857, and torsion gate/contribution maxima were exactly zero. Validity improved in ETFlow, Cartesian mild, medium, severe, high-flex, unseen scale 0.35, and non-ring slices. However, the largest registered core relative improvement was only 1.83%, below the required 10%, and the authorized 20k training was incomplete. These facts independently prevent a PASS.

Stage B rescued remained substantially stronger on aggregate validity (0.453955) than both Stage 2b Run A evaluated on the medium cohort (0.677345) and the stopped medium model (0.777601); this comparison is descriptive and does not change the frozen method selection.
