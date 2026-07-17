#!/usr/bin/env python
"""Build train-only Stage G calibration data with configurable GPU batching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from etflow.ecir.bounded_residual_confidence import (
    DATA_SCHEMA,
    build_stage_g_manifest,
    validate_stage_g_frame,
    verify_stage_f_identity,
)
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.confidence_calibration import optimal_scale_targets, strict_load_frozen_model
from etflow.ecir.feature_conditioned_confidence import inference_feature_batch
from etflow.ecir.mvr_dataset import deterministic_error_features
from scripts.build_ecir_mvr_stage_e0_calibration_data import load_training_items


def iter_builder_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if int(batch_size) < 1:
        raise ValueError("builder batch size must be positive")
    for start in range(0, len(items), int(batch_size)):
        yield items[start : start + int(batch_size)]


@torch.inference_mode()
def collect_stage_g_rows(
    model,
    items,
    validity,
    *,
    device: torch.device,
    config: Mapping[str, Any],
    molecule_split: Mapping[str, str],
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inference, training = config["inference"], config["training"]
    schedule = torch.linspace(0.0, 1.0, int(inference["teacher_steps"])).tolist()
    molecule_codes = {molecule: index for index, molecule in enumerate(sorted(molecule_split))}
    for selected in iter_builder_batches(items, batch_size):
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
                deterministic.append(
                    deterministic_error_features(
                        values, item["record"], str(item["row"].source_severity)
                    )
                )
                changed = torsion_change_metrics(
                    item["input"], current_cpu[left:right], item["record"]
                )["max_rotatable_torsion_change"]
                limit = 0.35 if item["rotatable"] >= 6 else 0.70
                remaining.append(max(0.0, limit - float(changed)))
            output = model(
                batch,
                current,
                current.new_full((len(selected),), float(time_value)),
                deterministic_features=torch.stack(deterministic).to(device),
                torsion_trust_remaining=current.new_tensor(remaining),
            )
            if (
                float(output["torsion_gate"].abs().max()) != 0.0
                or float(output["v_torsion_contribution"].abs().max()) != 0.0
            ):
                raise RuntimeError("Stage G observed a nonzero torsion branch")
            features, metadata = inference_feature_batch(
                current=current,
                output=output,
                batch=batch,
                items=selected,
                ptr=ptr,
                validity=validity,
                time_value=float(time_value),
            )
            features["original_confidence"] = output["bond_confidence"]
            target_residual = bond_length_residual(current, target, output["bond_indices"])
            predicted = output["bond_unattenuated_residual"]
            optimal = optimal_scale_targets(
                predicted, target_residual, epsilon=float(config["calibrator"]["epsilon"])
            )
            active = target_residual.abs() > float(training["active_target_threshold"])
            severe = target_residual.abs() > float(training["severe_target_threshold"])
            scale_weight = torch.ones_like(target_residual)
            scale_weight = torch.where(
                active, scale_weight * float(training["active_scale_weight"]), scale_weight
            )
            scale_weight = torch.where(
                severe, scale_weight * float(training["severe_scale_weight"]), scale_weight
            )
            wrong_sign = (
                target_residual.abs() > float(config["calibrator"]["epsilon"])
            ) & (torch.sign(predicted) != torch.sign(target_residual))
            zero_target = target_residual.abs() <= 1.0e-4
            already_valid = features["distance_to_valid_interval"] <= 0.0
            unsafe = ~features["sign_safe_mask"].to(torch.bool)
            already_valid_unsafe = already_valid & unsafe
            beneficial = (optimal > 0.0) & features["sign_safe_mask"].to(torch.bool)
            cpu_values = {
                **{name: value.detach().cpu() for name, value in features.items()},
                "target_residual": target_residual.detach().cpu(),
                "optimal_scale": optimal.detach().cpu(),
                "scale_weight": scale_weight.detach().cpu(),
                "wrong_sign": wrong_sign.detach().cpu(),
                "zero_target": zero_target.detach().cpu(),
                "already_valid_unsafe": already_valid_unsafe.detach().cpu(),
                "beneficial": beneficial.detach().cpu(),
            }
            for index, meta in enumerate(metadata):
                molecule = str(meta["molecule_id"])
                row = {
                    "schema_version": DATA_SCHEMA,
                    "split": molecule_split[molecule],
                    "molecule_id": molecule,
                    "molecule_code": molecule_codes[molecule],
                    "record_id": meta["record_id"],
                    "rollout_step": rollout_step,
                    "bond_index": meta["bond_index"],
                    **{
                        name: float(value[index]) if value[index].dtype.is_floating_point else int(value[index])
                        for name, value in features.items()
                        for value in (cpu_values[name],)
                    },
                    "target_residual": float(cpu_values["target_residual"][index]),
                    "optimal_scale": float(cpu_values["optimal_scale"][index]),
                    "scale_weight": float(cpu_values["scale_weight"][index]),
                    "wrong_sign": bool(cpu_values["wrong_sign"][index]),
                    "zero_target": bool(cpu_values["zero_target"][index]),
                    "already_valid_unsafe": bool(cpu_values["already_valid_unsafe"][index]),
                    "beneficial": bool(cpu_values["beneficial"][index]),
                    "training_only": True,
                    "validation_records_read": 0,
                    "test_records_read": 0,
                }
                rows.append(row)
            current = current + float(inference["step_size"]) * output["v_final"]
        del batch, current, target, output, features, cpu_values
    return pd.DataFrame(rows)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    pd.read_parquet(temporary)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ecir_mvr_stage_g_bounded_residual.yaml")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--builder-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verify_stage_f_identity(config)
    output = args.output_dir or Path(config["output_dir"])
    builder_batch_size = int(args.builder_batch_size or config["builder"]["batch_size"])
    if builder_batch_size < 1:
        raise ValueError("builder batch size must be positive")
    seed = int(args.seed if args.seed is not None else config["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage G requested CUDA but CUDA is unavailable")
    validity = ChemicalValidity(config["data"]["validity_statistics"])
    items = load_training_items(
        config["data"]["train_sources"],
        config["data"]["train_targets"],
        validity,
        limit_records=args.limit_records,
    )
    molecules = sorted({str(item["row"].molecule_id) for item in items})
    manifest = build_stage_g_manifest(
        molecules,
        checkpoint_sha256=config["checkpoint"]["sha256"],
        frozen_identities=config["frozen_identities"],
        builder_batch_size=builder_batch_size,
        seed=seed,
        fit_fraction=float(config["data"]["calibration_fit_fraction"]),
        smoke=bool(args.limit_records),
    )
    split = {value: "fit" for value in manifest["fit_molecule_ids"]}
    split.update({value: "internal_check" for value in manifest["internal_check_molecule_ids"]})
    model, checkpoint = strict_load_frozen_model(
        config["checkpoint"]["path"],
        expected_sha256=config["checkpoint"]["sha256"],
        device=device,
    )
    if checkpoint["frozen_identities"] != config["frozen_identities"]:
        raise RuntimeError("Stage G frozen identities changed")
    try:
        frame = collect_stage_g_rows(
            model,
            items,
            validity,
            device=device,
            config=config,
            molecule_split=split,
            batch_size=builder_batch_size,
        )
    except torch.cuda.OutOfMemoryError:
        print(json.dumps({"status": "CUDA_OOM", "builder_batch_size": builder_batch_size}))
        raise
    validate_stage_g_frame(frame, manifest)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(frame, output / "calibration_dataset.parquet")
    manifest.update(
        {
            "records": len(items),
            "bond_step_rows": len(frame),
            "dataset_file": "calibration_dataset.parquet",
            "actual_builder_batch_size": builder_batch_size,
        }
    )
    atomic_json_save(manifest, output / "calibration_manifest.json")
    print(
        json.dumps(
            {
                "status": "SMOKE_COMPLETE" if args.limit_records else "COMPLETE",
                "records": len(items),
                "molecules": len(molecules),
                "bond_step_rows": len(frame),
                "builder_batch_size": builder_batch_size,
                "training_only": True,
                "validation_records_read": 0,
                "test_records_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
