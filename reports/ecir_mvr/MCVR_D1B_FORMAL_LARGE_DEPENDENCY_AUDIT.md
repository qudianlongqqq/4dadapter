# MCVR D1-B Formal-Large Dependency Audit

## Decision

`D1B_FORMAL_TARGETS_NOT_READY`

The existing `data/flexbond_cache_formal_large` asset is the standard
generated-conformer/reference-pair cache. It is not, by itself, a D1-B
Minimal Validity Target dataset. The reported
`check_flexbond_data_pairs.py` PASS only establishes atom-order, graph,
reference-set, and Kabsch consistency; that check does not inspect Minimal
Validity Target coordinates or identities.

The formal-large trainer configuration and launch command must not be prepared
until matched train and validation Minimal Validity Target assets have been
materialized and audited. Substituting `x_ref_aligned` or another selected
reference for the missing Minimal Validity Target would change the D1-B
training objective and is forbidden.

## Evidence

- The completed D1-B pilot resolved config pairs
  `data/ecir_mvr/medium/real_sources/{train,val}.parquet` with
  `data/ecir_mvr/medium/minimal_targets/{train,val}.parquet`.
- `MCVRMixedDataset` reads the offline target manifest, rejects any missing
  `sample_id`, loads `x_target` from each target payload, and uses
  `target_metadata.initial_validity` to construct the active-mode labels.
- `MCVRMixedDataset` explicitly documents that online minimal-target
  construction is forbidden.
- `MCVRLoss` consumes `x_target`, `active_mode_mask`,
  `affected_atom_mask`, and D1-B bond residual targets derived from
  `x_target`.
- The standard `FlexBondOptimizerDataset` returns `x_init`, selected/aligned
  references, graph topology, and flow-training fields. Its required cache
  schema does not include a Minimal Validity Target payload or its identity.
- `check_flexbond_data_pairs.py` calls `validate_cache_record` and reports
  atom order, graph size, reference count, rotatable bonds, Kabsch RMSD, and
  selected reference index. It does not validate Minimal Validity Targets.
- The Windows development checkout used for this audit does not contain
  `data/flexbond_cache_formal_large`, so the Linux payloads could not be
  inspected for undocumented extra fields. Even if extra fields exist, they
  are not established by the reported 5/5 check and cannot be treated as
  audited D1-B targets.
- The protected file
  `reports/global4d_profile_bundle_verification.json` remains SHA256
  `738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d`.

## Audit Answers

1. **D1-B 5k real training entry**: the pilot was executed through
   `scripts/train_ecir_mvr_medium_rescue_v2.py`. Stage D authorization and the
   D1-A/D1-B experiment names were added to that trainer. The completed run
   metadata records 5,000 optimizer steps, batch 8, and commit `e25440b`.
2. **Data format**: `MCVRMixedDataset` consumes Parquet source and target
   manifests. Source rows point to cached real conformer records; target rows
   map the same `sample_id` values to `.pt` payloads containing `x_target` and
   `target_metadata`.
3. **Target semantics**: real-error samples use the offline Minimal Validity
   Target. Synthetic samples use the uncorrupted reference as their clean
   coordinate target, and clean-identity samples use an identity target.
4. **Minimal Validity Target used**: yes. It is mandatory for the real-error
   portion of the D1-B mixture and is protected by a frozen target identity.
5. **Direct formal cache compatibility**: no. The standard formal-large cache
   is compatible as an upstream source/reference graph asset, but it cannot be
   passed directly to the D1-B trainer without a formal-large real-source view
   and matched offline Minimal Validity Targets.
6. **All D1-B fields present in formal cache**: not established and not
   provided by the standard schema. In particular, audited Minimal Validity
   `x_target`, target status/initial validity, active-mode labels, affected-atom
   labels, and deterministic error features are absent from the direct dataset
   interface.
7. **Additional asset required**: yes. A formal-large Minimal Validity Target
   cache plus train/validation source and target manifests, metadata, and
   identity hashes are required. The target builder must remain offline and
   test must not be read.
8. **Can Stage D model/loss semantics be preserved**: yes in code, provided the
   missing target assets are built with the same Minimal Validity Target
   definition. `MCVRModel`, `MCVRLoss`, explicit bond settings, zero torsion
   gate, optimizer, learning-rate schedule, clipping, and safety settings can
   then be reused without changing their scientific meaning.
9. **Split isolation**: build and audit independent train and validation source
   and target manifests. Training reads train only; scheduled validation reads
   val only; no code path may instantiate or enumerate the test split. Record
   `test_records_read: 0` in preflight and formal manifests.
10. **From-scratch training**: supported after the target gate passes. The
    formal run must instantiate a fresh `MCVRModel`; the 5k checkpoint may only
    be strict-loaded in a separate compatibility check and must never initialize
    either the 100-step preflight or the formal 25k run.

## Required Target Gate

Before implementing or enabling the formal runner, the Linux assets must pass
all of the following checks:

- Exactly 50,000 unique training molecules are represented in the formal
  real-source manifest.
- Every train and validation source `sample_id` has exactly one target-manifest
  row and a readable target payload.
- Every target payload contains finite `x_target` coordinates with the same
  atom count/order as its source plus complete `target_metadata`, including
  `target_status` and `initial_validity`.
- Source and target split identities match, train and validation are disjoint,
  and test is neither opened nor enumerated.
- Metadata records the Minimal Validity Target builder configuration and a
  stable aggregate identity SHA256.
- A small audit constructs actual `MCVRMixedDataset` batches and verifies every
  field consumed by `MCVRModel` and `MCVRLoss` without reference-target
  fallback.

## Budget Note

Once the target gate passes, the matched exposure is:

`8 effective batch x 200,000 optimizer steps = 1,600,000 records`

Therefore batch 64 with accumulation 1 maps to 25,000 optimizer steps. The
corresponding checkpoints are 6,250, 12,500, 18,750, and 25,000, with
validation every 625 optimizer steps. This derivation does not authorize a
run while the target gate is failing.

## Actions Intentionally Not Taken

- No D1-B formal-large config or runner was created.
- No 2-batch or 100-step smoke was started.
- No CUDA device was occupied.
- No train, validation, or test formal-large payload was read on Windows.
- No checkpoint, Stage D/F/G/H0 result, or historical decision was modified.
- No commit or push was performed.
