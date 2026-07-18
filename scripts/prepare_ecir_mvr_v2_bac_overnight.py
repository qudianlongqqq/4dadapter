#!/usr/bin/env python3
"""Freeze validation-only BAC pilot identities without opening data payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


SCHEMA_VERSION = "mcvr-v2-bac-overnight-plan-v1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _metric_definitions() -> dict[str, Any]:
    public = {
        "aligned_RMSD": "Kabsch-aligned Cartesian RMSD",
        "COV_P": "precision-side conformer coverage at the registered threshold",
        "COV_R": "recall-side conformer coverage at the registered threshold",
        "MAT_P": "precision-side mean nearest-reference RMSD",
        "MAT_R": "recall-side mean nearest-generated RMSD",
        "identity_preservation": "atom and bond identity equality",
        "chirality_preservation": "stereocenter sign preservation",
    }
    custom = {
        "bond_outlier_rate": "train-envelope bond violation fraction",
        "bond_outlier_magnitude": "standardized bond envelope excess",
        "angle_outlier_rate": "train-envelope angle violation fraction",
        "ring_bond_outlier_rate": "ring-bond envelope violation fraction",
        "total_thresholded_validity_score": "registered internal weighted score",
        "accepted_fraction": "fraction passing BAC safety acceptance",
        "molecule_rms_displacement": "aligned RMS coordinate displacement",
    }
    return {
        "schema_version": "mcvr-v2-bac-metric-definitions-v1",
        "public_metrics": {
            name: {"definition": value, "official_metric": True}
            for name, value in public.items()
        },
        "custom_metrics": {
            name: {"definition": value, "custom_diagnostic": True}
            for name, value in custom.items()
        },
        "GenBench3D": {
            "available": False,
            "official_metric": False,
            "reason": "package is not installed in the frozen Windows environment",
        },
        "PoseBusters": {
            "available": False,
            "official_metric": False,
            "reason": "package is not installed in the frozen Windows environment",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_overnight"),
    )
    parser.add_argument("--split-seed", type=int, default=42017)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.formal_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    val_manifest = root / "real_sources" / "val.parquet"
    train_manifest = root / "real_sources" / "train.parquet"
    source_metadata_path = root / "real_sources" / "metadata.json"
    target_metadata_path = root / "minimal_targets" / "metadata.json"
    validation_path = root / "statistics" / "validation.json"
    for path in (
        val_manifest,
        train_manifest,
        source_metadata_path,
        target_metadata_path,
        validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    target_metadata = json.loads(target_metadata_path.read_text(encoding="utf-8"))
    asset_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if asset_validation.get("decision") != "D1B_FORMAL_TARGETS_READY":
        raise RuntimeError("formal train/validation assets are not ready")
    if any(
        int(payload.get("test_records_read", -1)) != 0
        for payload in (source_metadata, target_metadata, asset_validation)
    ):
        raise RuntimeError("an input identity does not certify zero test reads")
    frame = pd.read_parquet(val_manifest, columns=["sample_id", "molecule_id", "split"])
    if set(map(str, frame.split.unique())) != {"val"}:
        raise RuntimeError("validation manifest contains a non-val split")
    molecules = sorted(set(map(str, frame.molecule_id)))
    ranked = sorted(
        molecules,
        key=lambda molecule_id: hashlib.sha256(
            (
                f"{args.split_seed}|{source_metadata['formal_source_identity_sha256']}|"
                f"{molecule_id}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    tune_count = int(round(0.8 * len(ranked)))
    tune = set(ranked[:tune_count])
    holdout = set(ranked[tune_count:])
    if tune & holdout or tune | holdout != set(molecules):
        raise RuntimeError("invalid validation molecule partition")

    def cohort(name: str, values: set[str]) -> dict[str, Any]:
        selected = frame[frame.molecule_id.astype(str).isin(values)]
        ids = sorted(map(str, selected.sample_id))
        molecule_ids = sorted(values)
        return {
            "name": name,
            "records": len(ids),
            "molecules": len(molecule_ids),
            "ordered_sample_ids_sha256": _canonical_sha256(ids),
            "ordered_molecule_ids_sha256": _canonical_sha256(molecule_ids),
            "molecule_ids": molecule_ids,
        }

    cohorts = {
        "schema_version": "mcvr-v2-bac-validation-cohorts-v1",
        "split_seed": int(args.split_seed),
        "formal_source_identity_sha256": source_metadata[
            "formal_source_identity_sha256"
        ],
        "validation_manifest_identity_sha256": source_metadata["splits"]["val"][
            "manifest_sha256"
        ],
        "validation_tune": cohort("validation_tune", tune),
        "validation_holdout": cohort("validation_holdout", holdout),
        "holdout_policy": {
            "maximum_candidates": 2,
            "evaluations_per_candidate": 1,
            "tuning_after_holdout": False,
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    cohorts["identity_sha256"] = _canonical_sha256(cohorts)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "validation_cohorts.json", cohorts)
    _write_json(
        output / "code_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(_git("status", "--short")),
            "formal_source_identity_sha256": source_metadata[
                "formal_source_identity_sha256"
            ],
            "formal_target_identity_sha256": target_metadata[
                "formal_target_identity_sha256"
            ],
            "asset_validation_identity_sha256": asset_validation[
                "validation_identity_sha256"
            ],
            "validity_statistics_identity_sha256": target_metadata[
                "validity_statistics_identity_sha256"
            ],
            "baseline_parameters": 384678,
            "baseline_model": "MCVRModel LightEGNN D1-B",
            "test_records_read": 0,
            "test_assets_opened": False,
            "validation_only": True,
        },
    )
    _write_json(
        output / "environment.json",
        {
            "schema_version": "mcvr-v2-bac-environment-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pid": os.getpid(),
            "test_records_read": 0,
            "test_assets_opened": False,
            "validation_only": True,
        },
    )
    _write_json(output / "metric_definitions.json", _metric_definitions())
    (output / "decision_log.jsonl").touch(exist_ok=True)
    with (output / "experiment_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerow(
            [
                "experiment_id",
                "mode",
                "status",
                "optimizer_steps",
                "config_sha256",
                "checkpoint_sha256",
            ]
        )
    for name, columns in {
        "constraint_statistics.csv": [
            "experiment_id",
            "split",
            "bond_count",
            "angle_count",
            "clash_count",
        ],
        "candidate_comparison.csv": [
            "experiment_id",
            "qualified",
            "bond_delta",
            "angle_delta",
            "clash_delta",
            "rmsd_delta",
            "accepted_fraction",
        ],
    }.items():
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)
    not_run = {
        "status": "NOT_RUN",
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    for name in (
        "validation_tune_summary.json",
        "validation_holdout_summary.json",
        "failure_analysis.json",
        "recommended_next_step.json",
    ):
        _write_json(output / name, not_run)
    print(
        json.dumps(
            {**not_run, "status": "MCVR_V2_BAC_PLAN_FROZEN"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
