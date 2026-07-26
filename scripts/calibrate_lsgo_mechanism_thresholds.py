#!/usr/bin/env python3
"""Freeze TRAIN-Reference BA thresholds before mechanism xTB access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
from etflow.ecir.learned_geometry import LearnedGeometryObjective, distribution_parameters, prepare_graph
from etflow.ecir.lsgo_io import atomic_json, file_sha256
from etflow.ecir.lsgo_mechanism import primitive_z

OUT = ROOT / "reports/ecir_mvr/lsgo_mechanism"
CONFIG = ROOT / "configs/ecir_mvr_lsgo_mechanism.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data_path = Path(config["dataset"]["training_compact"])
    if file_sha256(data_path) != config["dataset"]["training_compact_sha256"]:
        raise RuntimeError("TRAIN compact SHA mismatch")
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    if payload["formal_test_records_read"] or payload["frozen_holdout_records_read"]:
        raise RuntimeError("protected split access")
    calibration = json.loads(Path(config["dataset"]["drcsr_calibration"]).read_text(encoding="utf-8"))
    manifest_path = Path(config["ba_anchor"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(config["representative_ba_seed"])
    row = next(value for value in manifest["checkpoints"] if value["variant"] == "B" and int(value["seed"]) == seed)
    checkpoint_path = Path(row["path"])
    if file_sha256(checkpoint_path) != row["sha256"]:
        raise RuntimeError("representative BA checkpoint SHA mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LearnedGeometryObjective(hidden_dim=128, layers=3, learned_sigma=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    bond_abs, angle_abs, ba_scores = [], [], []
    train_items = [item for item in payload["items"] if item["partition"] == "train"]
    for index, item in enumerate(train_items):
        graph = prepare_graph(item["record"], calibration).to(device)
        with torch.no_grad(): parameters = distribution_parameters(graph, model=model, variant="B")
        for reference_cpu in item["references"]:
            reference = torch.as_tensor(reference_cpu, dtype=torch.float64, device=device)
            bond_z, angle_z = primitive_z(reference, graph, parameters)
            bond_abs.append(bond_z.abs().cpu())
            angle_abs.append(angle_z.abs().cpu())
            b = bond_z.square().mean() if bond_z.numel() else reference.new_zeros(())
            a = angle_z.square().mean() if angle_z.numel() else reference.new_zeros(())
            ba_scores.append(float(torch.stack((b, a)).mean()))
        if (index + 1) % 250 == 0:
            print(f"THRESHOLD TRAIN {index + 1}/{len(train_items)}", flush=True)
    bond = torch.cat(bond_abs).numpy(); angle = torch.cat(angle_abs).numpy()
    result = {
        "schema_version": "mcvr-lsgo-mechanism-threshold-v1", "status": "FROZEN_BEFORE_XTB",
        "role": "diagnostic threshold calibration only; no coordinate or checkpoint selection",
        "partition": "train", "molecules": len(train_items), "reference_conformers": len(ba_scores),
        "representative_ba_seed": seed, "representative_ba_checkpoint_sha256": row["sha256"],
        "ba_abnormality_reference_median": float(np.median(ba_scores)),
        "bond_abs_z_reference_p95": float(np.quantile(bond, .95)),
        "angle_abs_z_reference_p95": float(np.quantile(angle, .95)),
        "bond_primitive_count": int(bond.size), "angle_primitive_count": int(angle.size),
        "training_compact_sha256": file_sha256(data_path), "config_sha256": file_sha256(CONFIG),
        "ba_manifest_sha256": file_sha256(manifest_path),
        "xtb_energy_records_read": 0, "xtb_force_records_read": 0,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
    }
    atomic_json(OUT / "manifests/THRESHOLD_FREEZE.json", result)
    print("LSGO_MECHANISM_THRESHOLDS_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
