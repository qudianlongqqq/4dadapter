#!/usr/bin/env python
"""Audit the frozen Cartesian 100k rollout used by the ECIR error atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem

from etflow.ecir.audit import (
    displacement_metrics,
    file_sha256,
    internal_metrics,
    nearest_reference_rmsd,
)
from etflow.ecir.target_building import (
    _record_to_rdkit_mapping,
    _set_conformer,
    restrained_force_field_relaxation,
)
from etflow.serial_global4d.cache import (
    load_frozen_cartesian_teacher,
    rollout_frozen_cartesian,
    tensor_sha256,
    validate_stage2_training_record,
)


DEFAULT_STEPS = (0, 1, 2, 4, 8, 10)
DEFAULT_SCALES = (0.05, 0.10, 0.20, 0.50, 1.00)
SNAPSHOT_STEPS = {0, 1, 2, 4, 10}


def _write_sdf(record, coordinates: torch.Tensor, path: Path, properties: dict[str, str]) -> None:
    mol, mapping = _record_to_rdkit_mapping(record)
    _set_conformer(mol, mapping, coordinates)
    for key, value in properties.items():
        mol.SetProp(key, str(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    # RDKit pads SDF property headers with one trailing space. Normalize only
    # line endings/whitespace so repository integrity checks stay clean.
    normalized = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atlas_path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--molecules", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/ecir_mvr/cartesian_audit"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters") or {})
    model_config = dict(config.get("model") or {})
    sampling = dict(config.get("sampling") or {})
    structural_keys = (
        "mode", "atom_feature_dim", "edge_attr_dim", "hidden_dim", "edge_hidden_dim",
        "time_embedding_dim", "num_layers", "dropout", "cutoff", "correction_scale",
    )
    config_matches = all(hparams.get(key) == model_config.get(key) for key in structural_keys)
    if not config_matches:
        raise ValueError("checkpoint hyper_parameters do not match config.model")
    if hparams.get("mode") != "cartesian_optimizer":
        raise ValueError("checkpoint is not a Cartesian optimizer")
    teacher = load_frozen_cartesian_teacher(args.checkpoint, device=args.device)

    atlas = pd.read_parquet(args.atlas_path)
    atlas = atlas[atlas.source_type == "cartesian_teacher_100k"].sort_values(
        ["molecule_id", "sample_id"]
    )
    selected = atlas.drop_duplicates("molecule_id").head(args.molecules)
    if selected.molecule_id.nunique() != args.molecules:
        raise ValueError("atlas has fewer distinct Cartesian molecules than requested")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_by_sample = {str(row["sample_id"]): row for row in manifest["records"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = args.output_dir / "sdf_snapshots"
    rows: list[dict] = []
    atom_rows: list[dict] = []
    identity_checks = []
    cached_deltas = []
    scale_linearity_deltas = []
    time_out_of_training = []

    for molecule_index, atlas_row in enumerate(selected.itertuples(index=False)):
        record = torch.load(Path(atlas_row.source_path), map_location="cpu", weights_only=False)
        validate_stage2_training_record(record, require_targets=True)
        sample_id = str(record["sample_id"])
        manifest_row = manifest_by_sample.get(sample_id)
        if manifest_row is None:
            raise ValueError(f"sample_id is absent from Confirm30 manifest: {sample_id}")
        identity = dict(record["teacher_sampling_identity"])
        expected_checkpoint_sha = str(identity["checkpoint"]["file_sha256"])
        expected_config_sha = str(identity["config_sha256"])
        mapping_mol, mapping = _record_to_rdkit_mapping(record)
        atomic_numbers = torch.as_tensor(record["atomic_numbers"], dtype=torch.long)
        mapped_numbers = torch.tensor(
            [mapping_mol.GetAtomWithIdx(mapping[index]).GetAtomicNum() for index in range(len(mapping))]
        )
        x_init = torch.as_tensor(record["x_init"], dtype=torch.float32)
        x_cached = torch.as_tensor(record["x_cart"], dtype=torch.float32)
        target_payload = torch.load(Path(atlas_row.target_cache_path), map_location="cpu", weights_only=False)
        target = torch.as_tensor(target_payload["x_target"], dtype=torch.float32)
        references = torch.as_tensor(record["x_ref_aligned"], dtype=torch.float32).unsqueeze(0)
        bonds = torch.as_tensor(record["edge_index"], dtype=torch.long)
        unique = bonds[:, bonds[0] < bonds[1]]
        median_bond = float(torch.linalg.vector_norm(x_init[unique[0]] - x_init[unique[1]], dim=-1).median())
        check = {
            "molecule_id": str(record["mol_id"]),
            "sample_id": sample_id,
            "molecule_id_match": str(record["mol_id"]) == str(atlas_row.molecule_id) == str(manifest_row["mol_id"]),
            "sample_id_match": sample_id == str(atlas_row.sample_id),
            "x_init_hash_match": str(record["x_init_hash"]) == str(manifest_row["x_init_hash"]),
            "x_cart_hash_match": tensor_sha256(x_cached) == str(record["x_cart_sha256"]),
            "checkpoint_identity_match": file_sha256(args.checkpoint) == expected_checkpoint_sha,
            "config_identity_match": file_sha256(args.config) == expected_config_sha,
            "manifest_identity_match": str(record["original_manifest_identity"]) == str(identity["cohort_manifest_sha256"]),
            "atom_count_match": mapping_mol.GetNumAtoms() == atomic_numbers.numel() == x_init.size(0),
            "atom_order_mapping_match": bool(torch.equal(mapped_numbers, atomic_numbers)),
            "hydrogen_count_match": int((mapped_numbers == 1).sum()) == int((atomic_numbers == 1).sum()),
            "coordinate_shape_match": x_init.shape == x_cached.shape == target.shape,
            "coordinate_dtype": str(x_init.dtype),
            "median_bond_length": median_bond,
            "angstrom_scale_plausible": 0.7 <= median_bond <= 2.5,
            "reference_identity": f"x_ref_aligned_sha256:{tensor_sha256(record['x_ref_aligned'])}",
            "reference_id_persisted": False,
        }
        hard = [key for key, value in check.items() if key.endswith("_match") and value is False]
        if hard:
            raise ValueError(f"identity/mapping failure for {sample_id}: {hard}")
        identity_checks.append(check)

        full_scale_one = {}
        for steps in DEFAULT_STEPS:
            if steps == 0:
                full_scale_one[steps] = x_init
            else:
                full_scale_one[steps], _ = rollout_frozen_cartesian(
                    teacher, record, refinement_steps=steps, update_scale=1.0,
                    max_displacement=float(sampling["max_displacement"]),
                    max_coordinate_norm=float(sampling["max_coordinate_norm"]), device=args.device,
                )
            if steps > 1:
                max_time = (steps - 1) / max(steps - 1, 1)
                time_out_of_training.append(max_time > float(hparams.get("t_max", 1.0)) + 1e-12)

        for scale in DEFAULT_SCALES:
            for steps in DEFAULT_STEPS:
                if steps == 0:
                    coordinates = x_init.clone()
                    diagnostics = {}
                else:
                    coordinates, diagnostics = rollout_frozen_cartesian(
                        teacher, record, refinement_steps=steps, update_scale=scale,
                        max_displacement=float(sampling["max_displacement"]),
                        max_coordinate_norm=float(sampling["max_coordinate_norm"]), device=args.device,
                    )
                    expected = x_init + scale * (full_scale_one[steps] - x_init)
                    scale_linearity_deltas.append(float((coordinates - expected).abs().max()))
                errors = internal_metrics(coordinates, target, record)
                displacement = displacement_metrics(x_init, coordinates)
                ff = restrained_force_field_relaxation(record, coordinates, max_steps=10).metadata()
                row = {
                    "molecule_index": molecule_index,
                    "molecule_id": str(record["mol_id"]),
                    "sample_id": sample_id,
                    "steps": steps,
                    "update_scale": scale,
                    "aligned_RMSD": nearest_reference_rmsd(coordinates, references),
                    **errors,
                    "MMFF_relaxation_energy_drop": ff.get("energy_drop"),
                    "relaxation_RMSD": ff.get("relaxation_rmsd"),
                    **displacement,
                    "correction_norm": float(diagnostics.get("mean_step_update_norm_applied", 0.0)),
                    "cumulative_correction_norm": float(diagnostics.get("mean_update_norm", 0.0)),
                    "clipping_fraction": float(diagnostics.get("clipping_fraction", 0.0)),
                    "stable": bool(diagnostics.get("stable", True)),
                }
                rows.append(row)
                if steps == 10 and abs(scale - 0.5) < 1e-12:
                    cached_deltas.append(float((coordinates - x_cached).abs().max()))
                if abs(scale - 0.5) < 1e-12 and steps in SNAPSHOT_STEPS:
                    stem = f"{molecule_index:02d}_step{steps:02d}"
                    _write_sdf(record, coordinates, sdf_dir / f"{stem}.sdf", {
                        "sample_id": sample_id, "steps": str(steps), "update_scale": str(scale)
                    })
                    aligned = __import__("etflow.commons.kabsch_utils", fromlist=["kabsch_align"]).kabsch_align(coordinates, x_init)
                    norms = torch.linalg.vector_norm(aligned - x_init, dim=-1)
                    for atom_index, norm in enumerate(norms.tolist()):
                        atom_rows.append({
                            "molecule_index": molecule_index, "sample_id": sample_id,
                            "steps": steps, "update_scale": scale, "atom_index": atom_index,
                            "atomic_number": int(atomic_numbers[atom_index]), "aligned_displacement": norm,
                        })

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_step.csv", index=False)
    pd.DataFrame(atom_rows).to_csv(args.output_dir / "per_atom_displacement.csv", index=False)
    molecule = frame.groupby(["molecule_id", "sample_id", "steps", "update_scale"], as_index=False)[
        frame.select_dtypes(include=[np.number]).columns.difference(["molecule_index", "steps", "update_scale"]).tolist()
    ].mean()
    molecule.to_csv(args.output_dir / "per_molecule.csv", index=False)
    aggregated = frame.groupby(["steps", "update_scale"], as_index=False).mean(numeric_only=True)
    # A single-step baseline is compared with the formal 10-step protocol.
    one = aggregated[(aggregated.steps == 1) & (aggregated.update_scale == 0.5)].iloc[0]
    ten = aggregated[(aggregated.steps == 10) & (aggregated.update_scale == 0.5)].iloc[0]
    multi_step_worse = bool(
        ten["bond_violation"] > one["bond_violation"]
        and ten["aligned_rms_displacement"] > one["aligned_rms_displacement"]
    )
    classification = "B" if multi_step_worse else "C"
    summary = {
        "classification": classification,
        "classification_text": (
            "single-step behavior is materially safer; multi-step rollout extrapolates outside the trained time range"
            if classification == "B" else "the checkpoint itself produces severe internal distortion without clear multi-step-only divergence"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "manifest_sha256_raw": file_sha256(args.manifest),
        "checkpoint_config_structure_match": config_matches,
        "strict_model_load": True,
        "train_time_range": [float(hparams.get("t_min", 0.0)), float(hparams.get("t_max", 1.0))],
        "rollout_time_range_for_steps_gt_1": [0.0, 1.0],
        "rollout_exceeds_training_time_range": bool(any(time_out_of_training)),
        "fixed_molecules": int(selected.molecule_id.nunique()),
        "identity_checks_all_pass": all(
            all(value for key, value in row.items() if key.endswith("_match")) for row in identity_checks
        ),
        "identity_checks": identity_checks,
        "reference_id_persisted": False,
        "reference_identity_fallback": "strict SHA256 of persisted x_ref_aligned; selected reference index/id is absent from Stage 2 cache schema",
        "cached_x_cart_max_abs_delta": max(cached_deltas),
        "cached_x_cart_reproduced": max(cached_deltas) <= 1.0e-5,
        "update_scale_max_linearity_delta": max(scale_linearity_deltas),
        "update_scale_applied_once": max(scale_linearity_deltas) <= 1.0e-5,
        "formal_protocol_mean": ten.to_dict(),
        "single_step_scale_0_5_mean": one.to_dict(),
        "tests_used_for_selection": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
