# FlexBond-4D: Bond-guided Flexible Conformer Optimizer

FlexBond-4D is a generator-agnostic secondary conformer optimizer. It consumes
a molecular graph and an upstream generated conformer, then predicts a light
EGNN Cartesian correction and (in the default hybrid mode) scalar bond-local
coefficients `q_b = [s, w1, w2, w3]`. A geometry-only Jacobian maps each `q_b`
back to Cartesian atom velocity.

The implementation deliberately does not reuse the full ETFlow Transformer.
ETFlow utilities are used only for topology featurization, rotatable-bond side
selection, cache interoperability, and benchmark compatibility.

Core idea: from atom-wise Cartesian refinement to bond-local internal-motion
correction.

## Commands

Build each split from an ETFlow packed sampled-output file that already contains
`pos_gen` and `pos_ref`:

```bash
python scripts/build_flexbond_init_cache.py \
  --init_path /path/to/train_samples.pkl \
  --output_dir data/flexbond_cache \
  --split train \
  --generator_name ETFlow \
  --generator_checkpoint /path/to/upstream.ckpt \
  --sample_seed 42 \
  --data_dir /path/to/GEOM
```

When references come from a separate file, both sides must expose an explicit
molecule id, SMILES, canonical SMILES, or scalar atom-map identity. List
positions are never accepted as cross-file identities. Schema-v2 caches also
store ordered topology signatures, bond annotations, Kabsch diagnostics, and
upstream provenance.

Export a label-free test cache and freeze the evaluation cohort before sampling:

```bash
python scripts/export_flexbond_inference_cache.py \
  --cache_dir data/flexbond_cache --split test \
  --output_dir data/flexbond_inference
python scripts/build_flexbond_eval_manifest.py \
  --cache_dir data/flexbond_inference --split test \
  --output eval_manifest.json
python scripts/check_flexbond_inference_no_labels.py \
  --cache_dir data/flexbond_inference --split test
```

If references are stored separately, add
`--reference_path /path/to/processed/train`; matching is by molecule id and then
exact ordered SMILES, never by list position.

Run data, Jacobian/least-squares, and SE(3) checks:

```bash
python scripts/check_flexbond_data_pairs.py \
  --cache_dir data/flexbond_cache --split train --num_samples 5
python scripts/check_flexbond_jacobian.py \
  --cache_dir data/flexbond_cache --split train
python scripts/check_flexbond_equivariance.py \
  --cache_dir data/flexbond_cache --split train
```

Run an individual 5,000-step smoke mode:

```bash
python scripts/train_flexbond_optimizer.py \
  --config configs/flexbond_optimizer_egnn.yaml \
  --mode cartesian_optimizer \
  --cache_dir data/flexbond_cache \
  --max_molecules 100 --max_steps 5000 \
  --output_dir logs_flexbond_optimizer/cartesian_smoke
```

Replace the mode with `flexbond4d_hybrid_optimizer` for the main method, or run
both plus checks, sampling, evaluation, and combined summaries with:

```bash
bash scripts/run_flexbond_optimizer_smoke.sh \
  data/flexbond_cache data/flexbond_inference
```

Each run writes `summary.md` and `summary.csv` under its evaluation directory;
the smoke root contains the combined versions. The 20k script is intentionally
manual and should only be invoked after smoke passes.

## Data contract

Each file under `data/flexbond_cache/{split}/` represents one upstream
conformer and stores molecular identity, ordered atomic numbers, graph and bond
features, rotatable-bond influence indices, `x_init`, all reference conformers,
the selected reference, its Kabsch-aligned coordinates, and provenance
metadata. Cache construction refuses positional pairing: records match by
molecule id or exact ordered SMILES.

For every `x_init`, the reference is selected by minimum Kabsch-aligned RMSD.
No symmetry permutation is attempted in this first prototype.

## Training versus inference

During hybrid training, `q_b_star` is obtained by a detached ridge
least-squares solve against the true residual. It is a training-time
pseudo-label only. Inference has no true conformer or target velocity and calls
the learned `q_b` head directly; the least-squares path is never invoked. The
primary objective is `MSE(v_final, u_t)`; `MSE(v_cart, u_t)` is logged separately.

## Equivariance

The backbone uses only scalar atom/edge features, distances, time embeddings,
and relative coordinate vectors. Cartesian outputs are scalar-weighted sums of
relative vectors. The bond frame is built from the bond axis and affected-side
local geometry, with degenerate frames skipped instead of falling back to a
laboratory axis. Thus the implemented velocity field is translation invariant
and SO(3)-equivariant. Use `scripts/check_flexbond_equivariance.py` to measure
mean and maximum errors for `v_cart`, `v_4d`, and `v_final`.

## Not Torsional Diffusion

Torsional Diffusion generates in a rotatable-angle torus using a torsion score
or SDE. FlexBond-4D starts from an existing Cartesian conformer and uses a
bond-local four-scalar correction as a geometric velocity mapping. It does not
implement a torsion SDE, reverse SDE, torus score matching, likelihood ODE, or
Hutchinson trace estimator. The first version trains with supervised flow
matching, not likelihood divergence.

## Current simplifications

- Euler integration only.
- Rotatable bonds only, with the deterministic smaller side affected.
- Per-bond ridge least squares and averaged overlapping atom corrections.
- No atom-symmetry matching during cache selection.
- The lightweight evaluator groups cache entries by source molecule and computes
  multi-conformer Kabsch COV/MAT without symmetry permutation; formal reporting
  should additionally run the existing RDKit/GEOM COV/MAT benchmark on packed
  outputs.
- No DMT-L or external Refiner integration in the first version.

## Safety boundary

All new logs use `logs_flexbond_optimizer/`. Cache building writes only to the
explicit `--output_dir`. None of these scripts modify legacy ETFlow logs,
checkpoints, summaries, configs, or training scripts.
