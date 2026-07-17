#!/usr/bin/env python
"""Fit the two-parameter Stage E0 calibrator on training molecules only."""

from __future__ import annotations

import argparse
import json
import os
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

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.ecir.confidence_calibration import (
    calibrator_payload, fit_monotonic_calibrator, load_calibrator,
    validate_calibration_frame, validate_calibration_manifest,
)


def calibration_curve(frame: pd.DataFrame, calibrator) -> pd.DataFrame:
    values = frame.copy()
    logits = torch.as_tensor(
        values.confidence_logit.to_numpy(dtype=np.float64, copy=True), dtype=torch.float64
    )
    values["original_confidence"] = torch.sigmoid(logits).numpy()
    values["calibrated_confidence"] = calibrator(logits).detach().numpy()
    masks = {
        "all": np.ones(len(values), dtype=bool),
        "active_target": values.active_target.to_numpy(bool),
        "outlier": values.outlier.to_numpy(bool),
        "severe_outlier": values.severe_outlier.to_numpy(bool),
        "ring": values.ring.to_numpy(bool),
        "nonring": ~values.ring.to_numpy(bool),
        "zero_target": values.zero_target.to_numpy(bool),
    }
    rows = []
    for split in ("fit", "internal_check"):
        split_mask = values.split.eq(split).to_numpy()
        for group, mask in masks.items():
            subset = values[split_mask & mask]
            if subset.empty:
                rows.append({
                    "split": split, "group": group, "bonds": 0,
                    "mean_optimal_scale": np.nan,
                    "mean_original_confidence": np.nan,
                    "mean_calibrated_confidence": np.nan,
                    "original_mae": np.nan, "calibrated_mae": np.nan,
                    "original_brier": np.nan, "calibrated_brier": np.nan,
                })
                continue
            target = subset.optimal_scale.to_numpy(float)
            original = subset.original_confidence.to_numpy(float)
            calibrated = subset.calibrated_confidence.to_numpy(float)
            rows.append({
                "split": split, "group": group, "bonds": len(subset),
                "mean_optimal_scale": float(target.mean()),
                "mean_original_confidence": float(original.mean()),
                "mean_calibrated_confidence": float(calibrated.mean()),
                "original_mae": float(np.abs(original - target).mean()),
                "calibrated_mae": float(np.abs(calibrated - target).mean()),
                "original_brier": float(np.square(original - target).mean()),
                "calibrated_brier": float(np.square(calibrated - target).mean()),
            })
    return pd.DataFrame(rows)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_e0_confidence_calibration.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.input_dir or Path(config["output_dir"])
    output = args.output_dir or source
    manifest = json.loads((source / "calibration_manifest.json").read_text(encoding="utf-8"))
    validate_calibration_manifest(manifest)
    if manifest["checkpoint_sha256"] != config["checkpoint"]["sha256"]:
        raise RuntimeError("calibration manifest checkpoint identity changed")
    frame = pd.read_parquet(source / "calibration_dataset.parquet")
    validate_calibration_frame(frame, manifest)
    calibrator, fit_metrics = fit_monotonic_calibrator(
        frame, manifest, epsilon=float(config["calibrator"]["epsilon"]),
        max_iter=int(config["calibrator"]["max_iter"]),
    )
    payload = calibrator_payload(
        calibrator, checkpoint_sha256=config["checkpoint"]["sha256"],
        training_molecule_identity_sha256=manifest["training_molecule_identity_sha256"],
        manifest_identity_sha256=manifest["manifest_identity_sha256"],
        fit_metrics=fit_metrics, smoke=bool(manifest.get("smoke")),
    )
    load_calibrator(payload)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_save(payload, output / "calibrator.json")
    _atomic_csv(calibration_curve(frame, calibrator), output / "calibration_curve.csv")
    print(json.dumps({
        "status": "SMOKE_COMPLETE" if manifest.get("smoke") else "COMPLETE",
        "learned_parameters": ["raw_a", "b"], "raw_a": payload["raw_a"],
        "a": payload["a"], "b": payload["b"],
        "validation_used_for_fit": False, "test_records_read": 0,
        "fit_metrics": fit_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
