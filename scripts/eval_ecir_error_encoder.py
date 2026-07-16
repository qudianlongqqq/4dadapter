#!/usr/bin/env python
"""Evaluate ECIR error-mode calibration on controlled validation corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch

from etflow.ecir.dataset import ECIRMixedDataset
from etflow.ecir.geometry import ERROR_MODES, geometry_error_vector
from etflow.ecir.model import ECIRFlowSystem
from etflow.ecir.structured_corruption import corrupt_conformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--target_cache_dir", type=Path, required=True)
    parser.add_argument("--atlas_path", type=Path)
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/ecir/error_encoder"))
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ECIRFlowSystem(**dict(payload["config"].get("model") or {})).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    dataset = ECIRMixedDataset(
        args.cache_dir,
        "val",
        atlas_path=args.atlas_path,
        target_cache_dir=args.target_cache_dir,
        real_error_ratio=1.0,
        synthetic_error_ratio=0.0,
        clean_identity_ratio=0.0,
        max_records=args.max_records,
    )
    rows = []
    modes = ("torsion", "bond_angle", "bond_length", "clash", "ring", "zero")
    with torch.inference_mode():
        for index, path in enumerate(dataset.files):
            record = torch.load(path, map_location="cpu", weights_only=False)
            base = torch.as_tensor(record["x_ref_aligned"], dtype=torch.float32)
            for mode_index, mode in enumerate(modes):
                if mode == "ring" and not bool(torch.as_tensor(record["bond_is_in_ring"]).any()):
                    continue
                if mode in {"torsion", "bond_angle"} and torch.as_tensor(record["rotatable_bond_index"]).size(1) == 0:
                    continue
                corrupted, metadata = corrupt_conformer(
                    record,
                    mode=mode,
                    coordinates=base,
                    generator=torch.Generator().manual_seed(10_000 * index + mode_index),
                )
                data = dataset.get(index)
                data.x_init = corrupted
                encoded = model.error_encoder(
                    data.to(device),
                    corrupted.to(device),
                    torch.tensor([0.5], device=device),
                    upstream_metadata=None,
                    apply_metadata_dropout=False,
                )
                truth = geometry_error_vector(corrupted, base, record)
                prediction = encoded["error_mean"][0].cpu()
                row = {
                    "sample_id": str(record.get("sample_id")),
                    "corruption_mode": mode,
                    "effective": bool(metadata["effective"]),
                    "true_dominant_mode": ERROR_MODES[int(truth.argmax())] if float(truth.max()) > 0 else "clean",
                    "predicted_dominant_mode": ERROR_MODES[int(prediction.argmax())],
                }
                for error_index, error_mode in enumerate(ERROR_MODES):
                    row[f"true_{error_mode}"] = float(truth[error_index])
                    row[f"predicted_{error_mode}"] = float(prediction[error_index])
                rows.append(row)
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_corruption.csv", index=False)
    summary = {}
    for mode, group in frame.groupby("corruption_mode"):
        nonclean = group[group["true_dominant_mode"] != "clean"]
        summary[mode] = {
            "records": len(group),
            "effective_fraction": float(group["effective"].mean()),
            "dominant_mode_accuracy": (
                float((nonclean["true_dominant_mode"] == nonclean["predicted_dominant_mode"]).mean())
                if len(nonclean) else None
            ),
            "mean_absolute_error": float(
                sum(
                    (group[f"predicted_{name}"] - group[f"true_{name}"]).abs().mean()
                    for name in ERROR_MODES
                )
                / len(ERROR_MODES)
            ),
            "predicted_error_sum": float(
                sum(group[f"predicted_{name}"].mean() for name in ERROR_MODES)
            ),
        }
    nonclean = frame[frame["true_dominant_mode"] != "clean"]
    result = {
        "records": len(frame),
        "metadata_supplied": False,
        "overall_dominant_mode_accuracy": float(
            (nonclean["true_dominant_mode"] == nonclean["predicted_dominant_mode"]).mean()
        ),
        "by_corruption": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
