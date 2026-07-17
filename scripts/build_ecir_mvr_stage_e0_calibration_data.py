#!/usr/bin/env python
"""Build the training-only Stage E0 per-bond calibration dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.bond_explicit import bond_length_residual
from etflow.ecir.audit import torsion_change_metrics
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.confidence_calibration import (
    CALIBRATION_DATA_SCHEMA, build_calibration_manifest, optimal_scale_targets,
    severity_weights, strict_load_frozen_model, validate_calibration_frame,
)
from etflow.ecir.geometry import bond_lengths
from etflow.ecir.mvr_dataset import deterministic_error_features
from etflow.ecir.run_a_evaluation import graph_data


def _source_coordinates(row) -> tuple[dict[str, Any], torch.Tensor]:
    record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
    if row.coordinate_path is not None and not pd.isna(row.coordinate_path):
        payload = torch.load(Path(row.coordinate_path), map_location="cpu", weights_only=False)
        coordinates = torch.as_tensor(payload[row.coordinate_key], dtype=torch.float32)
    else:
        coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
    return record, coordinates


def select_training_sources(
    source: pd.DataFrame, *, limit_molecules: int = 0, limit_records: int = 0
) -> pd.DataFrame:
    if set(source.split.astype(str).unique()) != {"train"}:
        raise ValueError("Stage E0 calibration fitting requires the train split only")
    source = source.sort_values(["molecule_id", "sample_id"]).copy()
    if limit_records:
        source = source.groupby("molecule_id", sort=True).head(1).head(int(limit_records))
    elif limit_molecules:
        identifiers = sorted(source.molecule_id.astype(str).unique())[:int(limit_molecules)]
        source = source[source.molecule_id.astype(str).isin(identifiers)]
    return source.reset_index(drop=True)


def load_training_items(
    source_path: str | Path, target_path: str | Path, validity,
    *, limit_molecules: int = 0, limit_records: int = 0,
) -> list[dict[str, Any]]:
    source = select_training_sources(
        pd.read_parquet(source_path),
        limit_molecules=limit_molecules, limit_records=limit_records,
    )
    targets = pd.read_parquet(target_path)
    if "split" in targets and set(targets.split.astype(str).unique()) != {"train"}:
        raise ValueError("Stage E0 calibration targets must be train-only")
    targets = targets.set_index("sample_id")
    items = []
    for row in source.itertuples(index=False):
        record, coordinates = _source_coordinates(row)
        target_row = targets.loc[row.sample_id]
        target_payload = torch.load(
            Path(target_row.target_cache_path), map_location="cpu", weights_only=False
        )
        target = torch.as_tensor(target_payload["x_target"], dtype=torch.float32)
        input_validity = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
        clean = all(float(input_validity[name]) <= 0.0 for name in (
            "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
            "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
            "stereocenter_degenerate_rate",
        ))
        active = torch.tensor([
            float(input_validity["bond_outlier_rate"] > 0),
            float(input_validity["angle_outlier_rate"] > 0),
            float(input_validity["ring_bond_outlier_rate"] > 0 or input_validity["ring_planarity_outlier_rate"] > 0),
            float(input_validity["clash_penetration"] > 0 or input_validity["severe_clash_rate"] > 0),
            float(input_validity["torsion_prior_outlier_score"] > 4.0), float(clean),
        ])
        items.append({
            "row": row, "record": record, "input": coordinates, "minimal_target": target,
            "data": graph_data(record, coordinates, row, active_mode_mask=active),
            "rotatable": int(record.get("num_rotatable_bonds", 0)),
        })
    return items


@torch.inference_mode()
def collect_calibration_rows(
    model, items, validity, *, device: torch.device, config: Mapping[str, Any],
    molecule_split: Mapping[str, str], batch_size: int = 16,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inference = config["inference"]
    calibration = config["calibrator"]
    schedule = torch.linspace(0.0, 1.0, int(inference["teacher_steps"])).tolist()
    for start in range(0, len(items), int(batch_size)):
        selected = items[start:start + int(batch_size)]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        target = torch.cat([item["minimal_target"] for item in selected]).to(device)
        ptr = batch.ptr.detach().cpu().tolist()
        for step, time_value in enumerate(schedule, start=1):
            current_cpu = current.detach().cpu()
            features, remaining = [], []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(
                    current_cpu[left:right], item["record"], baseline_coordinates=item["input"]
                )
                features.append(deterministic_error_features(
                    values, item["record"], str(item["row"].source_severity)
                ))
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                remaining.append(max(0.0, limit - float(changed)))
            output = model(
                batch, current, current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(features).to(device),
                torsion_trust_remaining=current.new_tensor(remaining),
            )
            if float(output["torsion_gate"].abs().max()) != 0.0 or float(output["v_torsion_contribution"].abs().max()) != 0.0:
                raise RuntimeError("Stage E0 observed a nonzero torsion branch")
            bonds = output["bond_indices"]
            bond_graph = batch.batch[bonds[0]]
            target_residual = bond_length_residual(current, target, bonds)
            predicted = output["bond_unattenuated_residual"]
            scales = optimal_scale_targets(predicted, target_residual, epsilon=float(calibration["epsilon"]))
            weights = severity_weights(
                target_residual,
                active_threshold=float(calibration["active_threshold"]),
                severe_threshold=float(calibration["severe_threshold"]),
                active_weight=float(calibration["active_weight"]),
                severe_weight=float(calibration["severe_weight"]),
            )
            edge_keep = batch.edge_index[0] < batch.edge_index[1]
            ring_mask = batch.bond_is_in_ring[edge_keep].to(torch.bool)
            for local, item in enumerate(selected):
                keep = bond_graph == local
                local_indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
                local_bonds = bonds[:, keep] - ptr[local]
                prepared = validity._prepare(item["record"])
                lengths = bond_lengths(current[ptr[local]:ptr[local + 1]], local_bonds)
                stats_by_bond = {
                    tuple(sorted(pair)): stat for pair, stat in zip(
                        prepared["bonds"].t().tolist(), prepared["bond_stats"].tolist()
                    )
                }
                molecule = str(item["row"].molecule_id)
                for bond_index, global_index in enumerate(local_indices.tolist()):
                    pair = tuple(sorted(local_bonds[:, bond_index].tolist()))
                    lower, upper = stats_by_bond[pair][:2]
                    length = float(lengths[bond_index])
                    residual = float(target_residual[global_index])
                    rows.append({
                        "schema_version": CALIBRATION_DATA_SCHEMA,
                        "split": molecule_split[molecule], "molecule_id": molecule,
                        "record_id": str(item["row"].sample_id), "rollout_step": step,
                        "bond_index": bond_index,
                        "confidence_logit": float(output["bond_confidence_logit"][global_index]),
                        "unattenuated_residual": float(predicted[global_index]),
                        "target_residual": residual, "optimal_scale": float(scales[global_index]),
                        "weight": float(weights[global_index]),
                        "active_target": abs(residual) > float(calibration["active_threshold"]),
                        "outlier": length < float(lower) or length > float(upper),
                        "severe_outlier": abs(residual) > float(calibration["severe_threshold"]),
                        "ring": bool(ring_mask[global_index]), "zero_target": abs(residual) <= 1.0e-4,
                        "source": str(item["row"].generator_name),
                        "severity": str(item["row"].source_severity),
                        "training_only": True, "test_records_read": 0,
                    })
            current = current + float(inference["step_size"]) * output["v_final"]
    return pd.DataFrame(rows)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    pd.read_parquet(temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_e0_confidence_calibration.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-molecules", type=int, default=0)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["output_dir"])
    smoke = bool(args.limit_molecules or args.limit_records)
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_training_items(
        config["data"]["train_sources"], config["data"]["train_targets"], validity,
        limit_molecules=args.limit_molecules, limit_records=args.limit_records,
    )
    molecules = sorted({str(item["row"].molecule_id) for item in items})
    manifest = build_calibration_manifest(
        molecules, checkpoint_sha256=config["checkpoint"]["sha256"],
        frozen_identities=config["frozen_identities"], seed=int(config["seed"]),
        fit_fraction=float(config["data"]["calibration_fit_fraction"]), smoke=smoke,
    )
    split = {value: "fit" for value in manifest["fit_molecule_ids"]}
    split.update({value: "internal_check" for value in manifest["internal_check_molecule_ids"]})
    model, payload = strict_load_frozen_model(
        config["checkpoint"]["path"], expected_sha256=config["checkpoint"]["sha256"],
        device=torch.device(args.device),
    )
    if payload["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("Stage E0 frozen data identities differ from checkpoint")
    frame = collect_calibration_rows(
        model, items, validity, device=torch.device(args.device), config=config,
        molecule_split=split,
    )
    validate_calibration_frame(frame, manifest)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(frame, output / "calibration_dataset.parquet")
    manifest.update({
        "records": len(items), "bond_step_rows": len(frame),
        "dataset_file": "calibration_dataset.parquet",
    })
    atomic_json_save(manifest, output / "calibration_manifest.json")
    print(json.dumps({
        "status": "SMOKE_COMPLETE" if smoke else "COMPLETE",
        "records": len(items), "molecules": len(molecules), "bond_step_rows": len(frame),
        "training_only": True, "validation_records_read": 0, "test_records_read": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
