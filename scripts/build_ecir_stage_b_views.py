#!/usr/bin/env python
"""Build label-free Stage B validation views without touching frozen Stage A data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch
import yaml

from etflow.serial_global4d.cache import (
    load_frozen_cartesian_teacher,
    rollout_frozen_cartesian,
    tensor_sha256,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_atlas", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    atlas = pd.read_parquet(args.val_atlas)
    if set(atlas.split.unique()) != {"val"}:
        raise ValueError("Stage B views require validation atlas only")
    config = yaml.safe_load(args.teacher_config.read_text(encoding="utf-8"))
    sampling = dict(config["sampling"])
    teacher = load_frozen_cartesian_teacher(args.teacher_checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir = args.output_dir / "coordinates"
    coordinate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in atlas.sort_values(["source_type", "molecule_id", "sample_id"]).itertuples(index=False):
        base = {
            "molecule_id": row.molecule_id, "sample_id": row.sample_id,
            "source_path": row.source_path, "target_cache_path": row.target_cache_path,
            "coordinate_key": row.coordinate_key, "source_type": row.source_type,
            "rotatable_bond_count": int(row.rotatable_bond_count),
            "view_mixed": True, "view_etflow_normal": row.source_type == "upstream_etflow_formal",
            "view_cartesian_severity": row.source_type == "cartesian_teacher_100k",
        }
        if row.source_type == "upstream_etflow_formal":
            rows.append({**base, "source_severity": "normal", "rollout_steps": 0,
                         "time_schedule_mode": "source_persisted", "actual_time_schedule": "[]",
                         "update_scale": 0.0, "generated_coordinate_path": None})
            continue
        rows.append({**base, "source_severity": "extrapolated_extreme", "rollout_steps": 10,
                     "time_schedule_mode": "legacy_full", "actual_time_schedule": json.dumps([index / 9 for index in range(10)]),
                     "update_scale": 0.5, "generated_coordinate_path": None})
        record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
        for steps, severity in ((1, "mild"), (2, "medium"), (4, "severe")):
            coordinates, diagnostics = rollout_frozen_cartesian(
                teacher, record, refinement_steps=steps,
                update_scale=float(sampling["update_scale"]),
                max_displacement=float(sampling["max_displacement"]),
                max_coordinate_norm=float(sampling["max_coordinate_norm"]),
                device=args.device, time_schedule_mode="train_range",
                strict_training_range=True,
            )
            output = coordinate_dir / f"{len(rows):04d}_{steps}step.pt"
            payload = {
                "sample_id": str(row.sample_id), "molecule_id": str(row.molecule_id),
                "coordinates": coordinates, "coordinates_sha256": tensor_sha256(coordinates),
                "source_x_init_hash": str(record["x_init_hash"]),
                "teacher_checkpoint_sha256": _sha(args.teacher_checkpoint),
                "teacher_config_sha256": _sha(args.teacher_config),
                "rollout_steps": steps, "source_severity": severity,
                "time_schedule_mode": "train_range",
                "actual_time_schedule": diagnostics["inference_time_schedule"],
                "training_time_range": diagnostics["training_time_range"],
                "update_scale": float(sampling["update_scale"]),
                "max_displacement": float(sampling["max_displacement"]),
                "diagnostics": diagnostics,
            }
            torch.save(payload, output)
            rows.append({
                **base, "view_mixed": False, "view_etflow_normal": False,
                "source_severity": severity, "rollout_steps": steps,
                "time_schedule_mode": "train_range",
                "actual_time_schedule": json.dumps(diagnostics["inference_time_schedule"]),
                "update_scale": float(sampling["update_scale"]),
                "generated_coordinate_path": str(output.resolve()),
            })
    frame = pd.DataFrame(rows)
    manifest = args.output_dir / "manifest.parquet"
    frame.to_parquet(manifest, index=False)
    metadata = {
        "schema_version": "ecir-stage-b-views-v1", "records": len(frame),
        "mixed_records": int(frame.view_mixed.sum()),
        "etflow_normal_records": int(frame.view_etflow_normal.sum()),
        "cartesian_severity_records": int(frame.view_cartesian_severity.sum()),
        "severity_counts": frame.source_severity.value_counts().to_dict(),
        "validation_atlas_sha256": _sha(args.val_atlas),
        "teacher_checkpoint_sha256": _sha(args.teacher_checkpoint),
        "teacher_config_sha256": _sha(args.teacher_config),
        "test_used": False,
    }
    metadata["manifest_sha256"] = _sha(manifest)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
