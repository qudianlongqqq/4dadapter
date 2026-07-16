#!/usr/bin/env python
"""Build medium MCVR real sources from a pre-registered molecule split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch
import yaml

from etflow.ecir.audit import file_sha256
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.serial_global4d.cache import load_frozen_cartesian_teacher
from scripts.build_ecir_mvr_real_sources import _build_split, _grade_cartesian


SCHEMA_VERSION = "ecir-mvr-medium-real-sources-v1"


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_split(path: Path, expected: str) -> tuple[list[dict[str, Any]], str]:
    if "test" in {part.lower() for part in path.resolve().parts}:
        raise ValueError("test split paths are forbidden")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["split"] != expected or payload["seed"] != 42:
        raise ValueError(f"unexpected frozen split payload: {path}")
    records = list(payload["records"])
    expected_count = 5000 if expected == "train" else 500
    if len(records) != expected_count or len({item["molecule_id"] for item in records}) != expected_count:
        raise ValueError(f"{expected} split does not contain {expected_count} unique molecules")
    for item in records:
        source = Path(item["source_path"])
        if "test" in {part.lower() for part in source.resolve().parts}:
            raise ValueError("test source path is forbidden")
        if not source.is_file():
            raise FileNotFoundError(source)
    return records, file_sha256(path)


def _pairs(records: list[dict[str, Any]], role: str) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for item in records:
        if item["source_role"] != role:
            continue
        path = Path(item["source_path"])
        record = torch.load(path, map_location="cpu", weights_only=False)
        molecule = str(record.get("source_mol_id", record.get("mol_id")))
        if molecule != item["molecule_id"]:
            raise ValueError(f"molecule identity mismatch: {path}")
        result.append((path, record))
    return result


def _strata_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        key: dict(Counter(item[key] for item in records))
        for key in ("source_role", "rotatable_group", "ring_group", "size_group")
    }


def _source_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "records": int(len(frame)),
        "molecules": int(frame.molecule_id.nunique()),
        "source_counts": {str(k): int(v) for k, v in frame.generator_name.value_counts().items()},
        "severity_counts": {str(k): int(v) for k, v in frame.source_severity.value_counts().items()},
        "update_scales": sorted(float(value) for value in frame.update_scale.unique()),
        "max_t": float(frame.t_max.max()),
        "out_of_domain_extreme_records": int((frame.source_severity == "out_of_domain_extreme").sum()),
    }


def _select_validation_sources(
    candidate: pd.DataFrame, records: list[dict[str, Any]], *, cartesian_molecules: int = 200
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cart = candidate[candidate.generator_name == "Cartesian_teacher_100k"].copy()
    upstream = candidate[candidate.generator_name == "ETFlow_formal_upstream"].copy()
    severity_order = {"normal": 0, "mild": 1, "medium": 2, "severe": 3}
    max_severity = {}
    for molecule, group in cart.groupby("molecule_id"):
        max_severity[str(molecule)] = max(
            (str(value) for value in group.source_severity), key=lambda value: severity_order[value]
        )
    available = Counter(max_severity.values())
    floors = {"severe": 20, "medium": 50, "mild": 50}
    selected: set[str] = set()
    for severity in ("severe", "medium", "mild"):
        choices = sorted(
            (molecule for molecule, value in max_severity.items() if value == severity),
            key=lambda molecule: _canonical_sha([42, "validation_source", severity, molecule]),
        )
        selected.update(choices[: min(floors[severity], len(choices))])
    remaining = sorted(
        (molecule for molecule in max_severity if molecule not in selected),
        key=lambda molecule: _canonical_sha([42, "validation_source", "fill", molecule]),
    )
    selected.update(remaining[: cartesian_molecules - len(selected)])
    if len(selected) != cartesian_molecules:
        raise ValueError("unable to assign 200 validation Cartesian molecules")
    output = pd.concat([
        cart[cart.molecule_id.astype(str).isin(selected)],
        upstream[~upstream.molecule_id.astype(str).isin(selected)],
    ], ignore_index=True)
    final_molecule_severity = Counter(max_severity[molecule] for molecule in selected)
    metadata = {
        "rule": "frozen severity floors followed by deterministic hash fill",
        "floors": floors,
        "candidate_cartesian_molecules": len(max_severity),
        "available_candidate_molecule_severity": dict(available),
        "selected_cartesian_molecules": len(selected),
        "selected_cartesian_molecule_severity": dict(final_molecule_severity),
        "selected_cartesian_molecule_ids_sha256": _canonical_sha(sorted(selected)),
        "model_results_used": False,
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_train", type=Path, required=True)
    parser.add_argument("--split_val", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--validity_stats", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/ecir_mvr/medium/real_sources"))
    parser.add_argument("--train_update_scale", type=float, default=0.50)
    parser.add_argument("--val_update_scale", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.train_update_scale != 0.50 or args.val_update_scale != 0.35:
        raise ValueError("medium update scales are frozen to train=0.50 and val=0.35")
    validity = ChemicalValidity(args.validity_stats)
    if validity.statistics["identity_sha256"] != "66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3":
        raise ValueError("frozen validity statistics identity changed")
    teacher_config = yaml.safe_load(args.teacher_config.read_text(encoding="utf-8"))
    if float(teacher_config["time_sampling"]["t_max"]) != 0.25:
        raise ValueError("Cartesian teacher training t_max must be 0.25")
    train_records, train_split_sha = _load_split(args.split_train, "train")
    val_records, val_split_sha = _load_split(args.split_val, "val")
    overlap = {item["molecule_id"] for item in train_records} & {item["molecule_id"] for item in val_records}
    if overlap:
        raise ValueError("train/validation molecule overlap")

    teacher = load_frozen_cartesian_teacher(args.teacher_checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _build_split(
        "train", _pairs(train_records, "etflow"), _pairs(train_records, "cartesian"),
        teacher=teacher, teacher_checkpoint=args.teacher_checkpoint,
        teacher_config=args.teacher_config, validity=validity, output_dir=args.output_dir,
        update_scale=args.train_update_scale, sampling=dict(teacher_config["sampling"]),
        device=args.device,
    )
    val_candidates = _pairs(val_records, "candidate")
    val_rows = _build_split(
        "val", val_candidates, val_candidates,
        teacher=teacher, teacher_checkpoint=args.teacher_checkpoint,
        teacher_config=args.teacher_config, validity=validity, output_dir=args.output_dir,
        update_scale=args.val_update_scale, sampling=dict(teacher_config["sampling"]),
        device=args.device,
    )
    combined = pd.concat([pd.DataFrame(train_rows), pd.DataFrame(val_rows)], ignore_index=True)
    thresholds = _grade_cartesian(combined[combined.split == "train"], combined)
    lookup = {
        item["molecule_id"]: item
        for item in train_records + val_records
    }
    for column in ("rotatable_group", "ring_group", "size_group"):
        combined[column] = combined.molecule_id.map(lambda molecule: lookup[str(molecule)][column])
    train = combined[combined.split == "train"].reset_index(drop=True)
    val, val_selection = _select_validation_sources(
        combined[combined.split == "val"].reset_index(drop=True), val_records
    )
    if train.molecule_id.nunique() != 5000 or val.molecule_id.nunique() != 500:
        raise AssertionError("medium molecule counts changed during source build")
    for frame in (train, val):
        if float(frame.t_max.max()) > 0.25 + 1e-12:
            raise ValueError("Cartesian schedule exceeded frozen training range")
        if int((frame.source_severity == "out_of_domain_extreme").sum()) != 0:
            raise ValueError("out-of-domain extreme source detected")
    required_severity = {"normal", "mild", "medium", "severe"}
    if not required_severity.issubset(set(val.source_severity)):
        raise ValueError(f"validation lacks required severity strata: {required_severity - set(val.source_severity)}")
    if not {"ring", "non_ring"}.issubset(set(val.ring_group)):
        raise ValueError("validation lacks ring or non-ring molecules")
    if int((val.rotatable_group == "rotatable_ge_6").sum()) < 20:
        raise ValueError("validation lacks sufficient high-flex records")
    for split, frame in (("train", train), ("val", val)):
        frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": 42,
        "test_paths_opened": 0,
        "test_used": False,
        "train_val_overlap": 0,
        "split_manifests": {
            "train_sha256": train_split_sha,
            "val_sha256": val_split_sha,
        },
        "validity_statistics_identity_sha256": validity.statistics["identity_sha256"],
        "teacher": {
            "checkpoint": str(args.teacher_checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.teacher_checkpoint),
            "config": str(args.teacher_config.resolve()),
            "config_sha256": file_sha256(args.teacher_config),
            "training_time_range": [0.0, 0.25],
        },
        "protocol": {
            "rollout_steps": [1, 2],
            "train_update_scale": 0.50,
            "validation_unseen_update_scale": 0.35,
            "update_scale_applied_once": True,
            "historical_ten_step_extreme_used": False,
            "severity_thresholds_from_train_only": thresholds,
            "validation_source_selection": val_selection,
        },
        "splits": {
            "train": {**_source_summary(train), "strata": _strata_counts(train_records), "parquet_sha256": file_sha256(args.output_dir / "train.parquet")},
            "val": {**_source_summary(val), "strata": _strata_counts(val_records), "parquet_sha256": file_sha256(args.output_dir / "val.parquet")},
        },
    }
    metadata["medium_real_source_identity_sha256"] = _canonical_sha(metadata)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
