#!/usr/bin/env python
"""Resumable per-record Cartesian Adapter sampling for formal-large selection."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch
import yaml

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    atomic_torch_save,
    checkpoint_inference_identity,
    file_sha256,
    resolve_device,
)
from etflow.commons.provenance import collect_run_provenance
from etflow.data.flexbond_eval_manifest import (
    build_manifest_aware_sample_payload,
    load_eval_manifest,
    manifest_content_sha256,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule
from scripts.sample_flexbond_optimizer import _bond_stability, resolve_correction_scale


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refinement_steps", type=int, default=10)
    parser.add_argument("--update_scale", "--alpha", type=float, required=True)
    parser.add_argument("--max_displacement", type=float, default=0.1)
    parser.add_argument("--max_coordinate_norm", type=float, default=1000.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.is_file() and args.output.stat().st_size:
        raise FileExistsError(f"Refusing to overwrite complete output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.parent / "partial_samples.pt"
    state_path = args.output.parent / "sampling_state.json"
    device = resolve_device(args.device)
    checkpoint_identity = checkpoint_inference_identity(args.checkpoint)

    manifest = load_eval_manifest(args.manifest)
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
    inference = validate_dataset_against_manifest(dataset, manifest)
    run_identity = {
        "checkpoint_inference_sha256": checkpoint_identity["inference_sha256"],
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": manifest_content_sha256(manifest),
        "split": args.split,
        "alpha": args.update_scale,
        "refinement_steps": args.refinement_steps,
        "max_displacement": args.max_displacement,
        "max_coordinate_norm": args.max_coordinate_norm,
    }
    records = []
    if partial_path.is_file():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if partial.get("partial") is not True or partial.get("run_identity") != run_identity:
            raise ValueError("Cartesian partial payload belongs to another command")
        records = list(partial.get("records", []))
    expected_prefix = [str(row["sample_id"]) for row in manifest["records"][: len(records)]]
    if [str(row["sample_id"]) for row in records] != expected_prefix:
        raise ValueError("Cartesian partial payload is not an ordered manifest prefix")

    model = FlexBondOptimizerLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).to(device).eval()
    if model.optimizer_mode != "cartesian_optimizer":
        raise ValueError(f"Expected cartesian_optimizer, got {model.optimizer_mode}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    scale_metadata = resolve_correction_scale(
        model,
        float(config.get("model", {}).get("correction_scale", model.hparams.correction_scale)),
        None,
    )
    started = time.perf_counter()
    total = len(manifest["records"])
    for index in range(len(records), total):
        row = manifest["records"][index]
        sample_id = str(row["sample_id"])
        data = inference[sample_id].to(device)
        sample_started = time.perf_counter()
        refined, stability = model.refine(
            data,
            refinement_steps=args.refinement_steps,
            update_scale=args.update_scale,
            max_displacement=args.max_displacement,
            max_coordinate_norm=args.max_coordinate_norm,
        )
        bond = _bond_stability(data, refined)
        stable = bool(stability["stable"] and bond["bond_stable"])
        records.append({
            "mol_id": data.mol_id,
            "sample_id": data.sample_id,
            "source_mol_id": data.source_mol_id,
            "smiles": data.smiles,
            "atomic_numbers": data.atomic_numbers.cpu(),
            "x_init": data.x_init.cpu(),
            "x_init_hash": str(row["x_init_hash"]),
            "x_refined": refined.cpu() if stable else None,
            "num_rotatable_bonds": int(data.num_rotatable_bonds.item()),
            "method_name": "cartesian_adapter",
            "optimizer_mode": model.optimizer_mode,
            "status": "success" if stable else "failed",
            "checkpoint_path": checkpoint_identity["path"],
            "checkpoint_inference_sha256": checkpoint_identity["inference_sha256"],
            "config_path": str(args.config.resolve()),
            "refinement_steps": args.refinement_steps,
            "update_scale": args.update_scale,
            "alpha": args.update_scale,
            "max_displacement": args.max_displacement,
            **scale_metadata,
            **stability,
            **bond,
        })
        selected = {**manifest, "records": manifest["records"][: len(records)]}
        partial_payload = build_manifest_aware_sample_payload(
            records=records,
            manifest=manifest,
            manifest_path=args.manifest,
            selected_manifest=selected,
            split=args.split,
            inference_cache_path=args.cache_dir,
            inference_by_id=inference,
            extra={"partial": True, "run_identity": run_identity},
        )
        atomic_torch_save(partial_payload, partial_path)
        elapsed = time.perf_counter() - started
        average = elapsed / len(records)
        atomic_json_save({
            "status": "partial" if len(records) < total else "finalizing",
            "updated_at": _now(),
            "method": "cartesian",
            "checkpoint_path": checkpoint_identity["path"],
            "checkpoint_file_sha256": checkpoint_identity["file_sha256"],
            "config_sha256": run_identity["config_sha256"],
            "manifest_sha256": run_identity["manifest_sha256"],
            "split": args.split,
            "alpha": args.update_scale,
            "refinement_steps": args.refinement_steps,
            "completed_count": len(records),
            "total_count": total,
            "current_record": sample_id,
            "last_record_seconds": time.perf_counter() - sample_started,
            "average_seconds_per_record": average,
            "eta_seconds": average * (total - len(records)),
            "partial_samples_path": str(partial_path.resolve()),
        }, state_path)
        print(
            f"[{index + 1}/{total}] {sample_id}; "
            f"ETA={average * (total - index - 1):.1f}s",
            flush=True,
        )

    failures = sum(row["status"] != "success" for row in records)
    provenance = collect_run_provenance(
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        cache_path=str(args.cache_dir),
    )
    final = build_manifest_aware_sample_payload(
        records=records,
        manifest=manifest,
        manifest_path=args.manifest,
        selected_manifest=manifest,
        split=args.split,
        inference_cache_path=args.cache_dir,
        inference_by_id=inference,
        extra={
            "provenance": provenance,
            "failure_count": failures,
            "failure_rate": failures / len(records) if records else 0.0,
        },
    )
    atomic_torch_save(final, args.output)
    partial_path.unlink(missing_ok=True)
    atomic_json_save({
        "status": "completed",
        "updated_at": _now(),
        "method": "cartesian",
        "completed_count": total,
        "total_count": total,
        "eta_seconds": 0.0,
        "total_seconds": time.perf_counter() - started,
        "output": str(args.output.resolve()),
    }, state_path)


if __name__ == "__main__":
    main()
