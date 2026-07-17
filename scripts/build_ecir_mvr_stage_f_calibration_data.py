#!/usr/bin/env python
"""Build train-only Stage F feature-conditioned calibration data."""

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
from etflow.ecir.audit import torsion_change_metrics
from etflow.ecir.bond_explicit import bond_length_residual
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.confidence_calibration import optimal_scale_targets, strict_load_frozen_model
from etflow.ecir.feature_conditioned_confidence import (
    DATA_SCHEMA, build_stage_f_manifest, inference_feature_batch,
    validate_stage_f_frame,
)
from etflow.ecir.mvr_dataset import deterministic_error_features
from scripts.build_ecir_mvr_stage_e0_calibration_data import load_training_items


@torch.inference_mode()
def collect_stage_f_rows(
    model, items, validity, *, device: torch.device, config: Mapping[str, Any],
    molecule_split: Mapping[str, str], batch_size: int = 16,
) -> pd.DataFrame:
    rows = []
    inference, training = config["inference"], config["training"]
    schedule = torch.linspace(0.0, 1.0, int(inference["teacher_steps"])).tolist()
    molecule_codes = {
        molecule: index for index, molecule in enumerate(sorted(molecule_split))
    }
    for start in range(0, len(items), int(batch_size)):
        selected = items[start:start + int(batch_size)]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = torch.cat([item["input"] for item in selected]).to(device)
        target = torch.cat([item["minimal_target"] for item in selected]).to(device)
        ptr = batch.ptr.detach().cpu().tolist()
        for rollout_step, time_value in enumerate(schedule, start=1):
            current_cpu = current.detach().cpu()
            deterministic, remaining = [], []
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                values = validity.evaluate(
                    current_cpu[left:right], item["record"], baseline_coordinates=item["input"]
                )
                deterministic.append(deterministic_error_features(
                    values, item["record"], str(item["row"].source_severity)
                ))
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                remaining.append(max(0.0, limit - float(changed)))
            output = model(
                batch, current, current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(deterministic).to(device),
                torsion_trust_remaining=current.new_tensor(remaining),
            )
            if float(output["torsion_gate"].abs().max()) != 0.0 or float(output["v_torsion_contribution"].abs().max()) != 0.0:
                raise RuntimeError("Stage F observed a nonzero torsion branch")
            features, metadata = inference_feature_batch(
                current=current, output=output, batch=batch, items=selected, ptr=ptr,
                validity=validity, time_value=float(time_value),
            )
            target_residual = bond_length_residual(current, target, output["bond_indices"])
            predicted = output["bond_unattenuated_residual"]
            optimal = optimal_scale_targets(
                predicted, target_residual, epsilon=float(config["calibrator"]["epsilon"])
            )
            active = target_residual.abs() > float(training["active_target_threshold"])
            severe = target_residual.abs() > float(training["severe_target_threshold"])
            scale_weight = torch.ones_like(target_residual)
            scale_weight = torch.where(active, scale_weight * float(training["active_scale_weight"]), scale_weight)
            scale_weight = torch.where(severe, scale_weight * float(training["severe_scale_weight"]), scale_weight)
            wrong_sign = (
                target_residual.abs() > float(config["calibrator"]["epsilon"])
            ) & (torch.sign(predicted) != torch.sign(target_residual))
            zero_target = target_residual.abs() <= 1.0e-4
            already_valid = features["distance_to_valid_interval"] <= 0.0
            unsafe = ~features["sign_safe_mask"].to(torch.bool)
            already_valid_unsafe = already_valid & unsafe
            beneficial = (optimal > 0.0) & features["sign_safe_mask"].to(torch.bool)
            for index, meta in enumerate(metadata):
                molecule = str(meta["molecule_id"])
                row = {
                    "schema_version": DATA_SCHEMA, "split": molecule_split[molecule],
                    "molecule_id": molecule, "molecule_code": molecule_codes[molecule],
                    "record_id": meta["record_id"], "rollout_step": rollout_step,
                    "bond_index": meta["bond_index"],
                    **{
                        name: float(value[index]) if value[index].dtype.is_floating_point else int(value[index])
                        for name, value in features.items()
                    },
                    "target_residual": float(target_residual[index]),
                    "optimal_scale": float(optimal[index]), "scale_weight": float(scale_weight[index]),
                    "wrong_sign": bool(wrong_sign[index]), "zero_target": bool(zero_target[index]),
                    "already_valid_unsafe": bool(already_valid_unsafe[index]),
                    "beneficial": bool(beneficial[index]), "training_only": True,
                    "validation_records_read": 0, "test_records_read": 0,
                }
                rows.append(row)
            current = current + float(inference["step_size"]) * output["v_final"]
    return pd.DataFrame(rows)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    pd.read_parquet(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_f_feature_confidence.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["output_dir"])
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_training_items(
        config["data"]["train_sources"], config["data"]["train_targets"], validity,
        limit_records=args.limit_records,
    )
    molecules = sorted({str(item["row"].molecule_id) for item in items})
    manifest = build_stage_f_manifest(
        molecules, checkpoint_sha256=config["checkpoint"]["sha256"],
        frozen_identities=config["frozen_identities"], seed=int(config["seed"]),
        fit_fraction=float(config["data"]["calibration_fit_fraction"]),
        smoke=bool(args.limit_records),
    )
    split = {value: "fit" for value in manifest["fit_molecule_ids"]}
    split.update({value: "internal_check" for value in manifest["internal_check_molecule_ids"]})
    model, checkpoint = strict_load_frozen_model(
        config["checkpoint"]["path"], expected_sha256=config["checkpoint"]["sha256"],
        device=torch.device(args.device),
    )
    if checkpoint["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("Stage F frozen identities changed")
    frame = collect_stage_f_rows(
        model, items, validity, device=torch.device(args.device), config=config,
        molecule_split=split,
    )
    validate_stage_f_frame(frame, manifest)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(frame, output / "calibration_dataset.parquet")
    manifest.update({
        "records": len(items), "bond_step_rows": len(frame),
        "dataset_file": "calibration_dataset.parquet",
    })
    atomic_json_save(manifest, output / "calibration_manifest.json")
    print(json.dumps({
        "status": "SMOKE_COMPLETE" if args.limit_records else "COMPLETE",
        "records": len(items), "molecules": len(molecules), "bond_step_rows": len(frame),
        "training_only": True, "validation_records_read": 0, "test_records_read": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
