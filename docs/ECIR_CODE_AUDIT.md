# ECIR-Flow code audit

Audit date: 2026-07-16  
Branch: `feat/ecir-flow`  
Base commit: `3c7836fc3e715fb07fba929aa58623bca9b1cf88`

This document records the repository state before ECIR-Flow changes. No existing
Cartesian, Strict/Gated Global4D, or Serial Global4D implementation is replaced.
The untracked file `reports/global4d_profile_bundle_verification.json` is outside
ECIR scope and must remain unmodified and uncommitted.

## 1. Existing models

### Cartesian Adapter

The current Cartesian teacher is
`etflow.models.flexbond_optimizer.FlexBondOptimizerLightningModule` with
`mode=cartesian_optimizer`. Its equivariant trunk is
`LightEGNNRefinerBackbone` in
`etflow/models/components/light_egnn_refiner.py`.

`LightEGNNRefinerBackbone.encode()` returns:

- invariant atom states `h: [N, hidden_dim]`;
- equivariant Cartesian velocity `v_cart: [N, 3]`;
- per-atom sinusoidal time embedding `time_emb: [N, time_dim]`.

The legacy `forward()` additionally predicts a per-selected-bond four-vector
`q_b`. In Cartesian mode only the complete `v_cart_raw` is used.

The formal teacher audited by the preceding Serial work is:

`E:/3dconformergenerationcode/pretrained/cartesian_formal_large_seed42_pretrained/checkpoints/step100000.ckpt`

with its adjacent `config.resolved.yaml`. The 200k checkpoint also exists, but is
not the selected formal teacher.

### Strict/Gated Global4D

The coupled implementation is
`etflow.models.global_coupled_4d_flow.GlobalCoupled4DFlowLightningModule`.
Compatibility loading and architecture checks live in
`etflow/models/global4d_checkpoint.py`. Gated V2 remains available on its
existing branch/commit and is not deleted or made the default ECIR head.

### Serial Global4D

`etflow.serial_global4d.model.SerialGlobal4DResidualRefiner` has an invariant
EGNN trunk, a joint coefficient head and a graph gate. It has no Cartesian head;
its correction is `beta * gate * Jq`. Its Stage 2 cache, target materialization,
safe update and evaluation logic live under `etflow/serial_global4d/`.

For ECIR, all 4D/Jacobian paths are retained only for structured corruption,
diagnostics, internal-mode labels and ablations. ECIR's default prediction is a
complete Cartesian velocity, not `q` and not a pseudoinverse target.

## 2. Checkpoint loading

- Formal Cartesian loading: `load_frozen_cartesian_teacher()` in
  `etflow/serial_global4d/cache.py`; it requires `cartesian_optimizer`, freezes
  every parameter and uses eval mode.
- Lightning sampling entry points use `load_from_checkpoint()`.
- Formal ETFlow upstream generation performs strict `state_dict` loading in
  `scripts/generate_etflow_formal_large_upstream.py`.
- Serial checkpoints are plain dictionaries containing
  `model_state_dict`, optimizer state and training metadata; training and
  Confirm30 evaluation load them with `strict=True`.
- Global4D compatibility loading intentionally supports old checkpoint schemas;
  ECIR must not silently reuse that relaxed path for its own checkpoints.

## 3. Cache and identity contract

The formal source cache is
`E:/3dconformergenerationcode/dataset/flexbond_cache_formal_large`:

- train: 150,000 records;
- val: 10,000 records;
- test: 23,882 records.

It uses FlexBond cache schema 2.0. Records contain graph topology, atom order,
`x_init`, reference candidates, the selected aligned reference, generator
provenance and stable hashes. Important graph fields are:

- `atomic_numbers`, `node_attr`, `edge_index`, `edge_attr`;
- `bond_type`, `bond_is_aromatic`, `bond_is_in_ring`;
- `rotatable_bond_index`, `atom_bond_influence_index`;
- `x_init_hash`, topology signatures and ordered atom-map IDs.

`sample_id` is a conformer-record identity of the form
`<split>::<source_record_id>__genNNNN`. `source_mol_id`/`source_record_id` identify
the molecule. A molecule can therefore own multiple generated samples.

Reference pairing is fail-closed in `etflow/data/flexbond_cache_schema.py` and
`scripts/build_flexbond_init_cache.py`: explicit record IDs, dataset indices,
SMILES and atom maps are used; positional/index-only cross-file matching is not
allowed. Ambiguous SMILES are rejected. `x_init_hash` binds float32 coordinates
to ordered atomic numbers.

## 4. Reference conformers

Multiple references are already represented. The audited first formal train
record has `x_ref_candidates` shape `[27, 48, 3]`, selected index 14, and one
aligned selected reference `[48, 3]`. Selection records
`selected_reference_index`, `selected_ref_id` and diagnostic RMSD.

Consequently ECIR can implement multi-reference soft coupling without averaging
Cartesian coordinates. Single-reference records must be marked explicitly.

## 5. Manifests and splits

- Manifest identity and validation are implemented in
  `etflow/data/flexbond_eval_manifest.py`; identity is a path-independent
  canonical JSON SHA256 over ordered rows.
- The formal Confirm30 validation manifest is
  `E:/3dconformergenerationcode/pretrained/serial_global4d_validation/formal_large_val_confirm30.json`
  with 60 records / 30 molecules.
- The frozen Serial train pilot manifest is under
  `E:/3dconformergenerationcode/serial_global4d_work/pilot_manifests/` and owns
  the previously verified 5,001-record Stage 2 cache.
- Formal-large generation scripts expect complete train/val/test manifests under
  a `manifests/` output directory. Those full frozen JSON manifests are not
  tracked in this repository checkout.
- Formal-small preparation scripts generate `eval_manifest_formal_small.json`,
  but no frozen formal-small manifest is present in this checkout.

ECIR must split and aggregate by molecule, never by individual conformer. Test is
reserved for final reporting and leave-one-source-out evaluation, not model or
threshold selection.

## 6. RDKit and force-field support

The active CUDA environment contains RDKit 2026.03.4. Current repository force
field support is limited to optional `MMFFOptimizeMolecule()` use inside
`etflow/commons/covmat.py`; it does not provide the required restrained target
builder, MMFF94s coverage/status accounting, paired energy deltas, or controlled
UFF fallback. There is no repository UFF implementation.

ECIR therefore needs a new explicit force-field wrapper that:

- distinguishes MMFF94s, UFF fallback and unsupported cases;
- never records unsupported energy as zero;
- records convergence, energies, steps and coordinate drift;
- accepts/rejects relaxation targets using basin-preservation criteria.

The environment has pandas 3.0.3 but does not currently have `pyarrow`; parquet
atlas writing requires adding/installing a parquet engine before the atlas stage.

## 7. Sampling and evaluation entry points

- Cartesian: `scripts/sample_formal_large_cartesian.py`.
- Strict/Gated Global4D: `scripts/sample_global_coupled_4d_flow.py` and related
  sampling scripts.
- Serial validation: `scripts/evaluate_serial_global4d_confirm30.py`.
- Adapter/internal evaluation: `scripts/eval_flexbond_optimizer.py`.
- COV/MAT: `scripts/eval_cov_mat.py` and `etflow/commons/covmat.py`.
- Formal validation/reporting: `scripts/validate_formal_large.py` and
  `scripts/report_formal_large_final_test.py`.

Sample payloads are manifest-aware and retain ordered sample IDs, molecule IDs,
`x_init_hash` values and inference-cache provenance.

## 8. Internal motion and Jacobian code

Reusable components are:

- `etflow/commons/global_coupled_4d_topology.py`: joint topology and caching;
- `etflow/commons/global_coupled_4d_jacobian.py`: stretch/bending/torsion
  Jacobians, `Jq`, rate decomposition and first-order updates;
- `etflow/commons/flexbond_jacobian.py`: per-bond local frames, Jacobian
  application and legacy least-squares pseudo-labels;
- `etflow/commons/molecular_kinematics.py`: fragment/joint topology;
- `etflow/commons/rotatable_motion.py`: RDKit rotatable-bond sides and rigid
  motion decomposition;
- `etflow/serial_global4d/targets.py`: Stage 2 diagnostic target projection;
- `etflow/serial_global4d/safety.py`: label-free trust-region clipping,
  backtracking and rejection.

ECIR may reuse geometry/topology/Jacobian operations to generate corruptions and
compute `B_mode(x)v`. It must not use
`x_reference - x_upstream -> pseudoinverse(J) -> q_target` as its main target.

## 9. Audit conclusions and implementation constraints

1. The existing EGNN encoder is suitable for an error-conditioned Cartesian
   head and already exposes invariant node states plus equivariant vectors.
2. Formal caches contain multiple references and sufficient topology for bond,
   torsion and ring diagnostics, but reliable force-field construction requires
   rebuilding an RDKit molecule from SMILES plus strict atom-order validation.
3. Existing sample artifacts are heterogeneous in schema. The atlas must require
   explicit source descriptors and record missing provenance rather than infer it
   from filenames.
4. The existing formal source cache is read-only. ECIR atlas, targets,
   checkpoints and diagnostics must use `data/ecir_error_atlas`, `logs_ecir` and
   `diagnostics/ecir` exclusively.
5. Four-step ECIR teacher validation precedes any one-step student work.
6. Progressive Stage 1-4 gates are experiment configuration, not claimed domain
   standards. Failure at Stage 2 or Stage 3 must stop later automatic stages.
