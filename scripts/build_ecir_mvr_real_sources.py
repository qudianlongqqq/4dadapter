#!/usr/bin/env python
"""Build leakage-free Stage C real-error sources without reading test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import pandas as pd
import torch
import yaml

from etflow.ecir.audit import displacement_metrics, file_sha256, torsion_change_metrics
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.serial_global4d.cache import load_frozen_cartesian_teacher, rollout_frozen_cartesian


SCHEMA_VERSION = "ecir-mvr-real-sources-v1"
VALIDITY_COLUMNS = (
    "bond_outlier_rate",
    "bond_outlier_magnitude",
    "angle_outlier_rate",
    "angle_outlier_magnitude",
    "ring_bond_outlier_rate",
    "ring_planarity_outlier_rate",
    "clash_penetration",
    "severe_clash_rate",
    "chirality_preserved",
    "stereocenter_degenerate_rate",
    "torsion_prior_outlier_score",
    "total_thresholded_validity_score",
)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    array = torch.as_tensor(value, dtype=torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"source cache record must be a mapping: {path}")
    return value


def _molecule_id(record: dict[str, Any]) -> str:
    return str(record.get("source_mol_id", record.get("mol_id")))


def _unique_records(
    directory: Path,
    count: int,
    *,
    excluded: set[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    seen = set(excluded or ())
    for path in sorted(directory.glob("*.pt")):
        record = _record(path)
        molecule = _molecule_id(record)
        if molecule in seen:
            continue
        seen.add(molecule)
        result.append((path, record))
        if len(result) == count:
            return result
    raise ValueError(f"{directory} has only {len(result)} eligible unique molecules; need {count}")


def _source_checkpoint_identity(record: dict[str, Any]) -> tuple[str, str]:
    checkpoint = str(record.get("generator_checkpoint", "formal_upstream_checkpoint_unavailable"))
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return file_sha256(path), "file"
    return _canonical_sha({"unavailable_checkpoint_identity": checkpoint}), "identifier_only"


def _severity_score(base, candidate, displacement, torsion) -> float:
    """A frozen joint score; train quantiles turn it into severity strata."""

    components = [
        displacement["aligned_rms_displacement"] / 0.03,
        displacement["max_atom_displacement"] / 0.08,
        max(0.0, candidate["bond_outlier_rate"] - base["bond_outlier_rate"]) / 0.05,
        max(0.0, candidate["angle_outlier_rate"] - base["angle_outlier_rate"]) / 0.05,
        max(0.0, candidate["ring_bond_outlier_rate"] - base["ring_bond_outlier_rate"]) / 0.05,
        max(0.0, candidate["ring_planarity_outlier_rate"] - base["ring_planarity_outlier_rate"]) / 0.05,
        max(0.0, candidate["clash_penetration"] - base["clash_penetration"]) / 0.01,
        max(0.0, candidate["severe_clash_rate"] - base["severe_clash_rate"]) * 4.0,
        torsion["max_rotatable_torsion_change"] / 0.25,
    ]
    return float(np.mean(components))


def _flatten_validity(values: dict[str, float]) -> dict[str, float]:
    return {f"source_{name}": float(values[name]) for name in VALIDITY_COLUMNS}


def _base_row(
    *,
    split: str,
    generator_name: str,
    checkpoint_sha256: str,
    config_sha256: str,
    seed: int,
    nfe: int,
    solver: str,
    rollout_steps: int,
    time_schedule: list[float],
    update_scale: float,
    molecule_id: str,
    sample_id: str,
    coordinate_sha256: str,
    source_path: Path,
    coordinate_path: Path | None,
    coordinate_key: str,
    reference_availability: bool,
    source_validity: dict[str, float],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "generator_name": generator_name,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "seed": int(seed),
        "NFE": int(nfe),
        "solver": solver,
        "rollout_steps": int(rollout_steps),
        "time_schedule": json.dumps([float(value) for value in time_schedule]),
        "t_min": float(min(time_schedule)) if time_schedule else 0.0,
        "t_max": float(max(time_schedule)) if time_schedule else 0.0,
        "update_scale": float(update_scale),
        "source_severity": "pending",
        "source_validity": json.dumps(source_validity, sort_keys=True),
        "molecule_id": molecule_id,
        "sample_id": sample_id,
        "coordinate_sha256": coordinate_sha256,
        "source_path": str(source_path.resolve()),
        "coordinate_path": str(coordinate_path.resolve()) if coordinate_path else None,
        "coordinate_key": coordinate_key,
        "reference_availability": bool(reference_availability),
        "provenance": json.dumps(provenance, sort_keys=True, default=str),
        **_flatten_validity(source_validity),
    }


def _write_coordinate(
    output: Path,
    coordinates: torch.Tensor,
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "coordinates": torch.as_tensor(coordinates, dtype=torch.float32).cpu(),
        "coordinate_sha256": _tensor_sha(coordinates),
        "source_sample_id": str(record.get("sample_id")),
        "source_molecule_id": _molecule_id(record),
        **metadata,
    }, output)


def _build_split(
    split: str,
    upstream_records: Iterable[tuple[Path, dict[str, Any]]],
    cartesian_records: Iterable[tuple[Path, dict[str, Any]]],
    *,
    teacher,
    teacher_checkpoint: Path,
    teacher_config: Path,
    validity: ChemicalValidity,
    output_dir: Path,
    update_scale: float,
    sampling: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    rows = []
    for path, record in upstream_records:
        coordinates = torch.as_tensor(record["x_init"], dtype=torch.float32)
        metrics = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
        checkpoint_sha, checkpoint_mode = _source_checkpoint_identity(record)
        config_identity = {
            "generator_name": record.get("generator_name", "ETFlow"),
            "generator_checkpoint": record.get("generator_checkpoint"),
            "sample_seed": record.get("sample_seed", 42),
            "coordinate_key": "x_init",
        }
        rows.append(_base_row(
            split=split,
            generator_name="ETFlow_formal_upstream",
            checkpoint_sha256=checkpoint_sha,
            config_sha256=_canonical_sha(config_identity),
            seed=int(record.get("sample_seed", 42)),
            nfe=10,
            solver="ETFlow",
            rollout_steps=0,
            time_schedule=[],
            update_scale=0.0,
            molecule_id=_molecule_id(record),
            sample_id=str(record.get("sample_id")),
            coordinate_sha256=_tensor_sha(coordinates),
            source_path=path,
            coordinate_path=None,
            coordinate_key="x_init",
            reference_availability=record.get("x_ref_aligned") is not None,
            source_validity=metrics,
            provenance={
                "source_kind": "formal_upstream",
                "source_path": str(path.resolve()),
                "checkpoint_identity_mode": checkpoint_mode,
                "test_read": False,
            },
        ))
        rows[-1]["source_severity"] = "normal"
        rows[-1]["source_severity_score"] = 0.0
        rows[-1].update({
            "molecule_rms_displacement": 0.0,
            "mean_atom_displacement": 0.0,
            "max_atom_displacement": 0.0,
            "torsion_change": 0.0,
            "max_rotatable_torsion_change": 0.0,
        })

    checkpoint_sha = file_sha256(teacher_checkpoint)
    config_sha = file_sha256(teacher_config)
    coordinate_dir = output_dir / "coordinates" / split
    for source_index, (path, record) in enumerate(cartesian_records):
        x_input = torch.as_tensor(record["x_init"], dtype=torch.float32)
        base_validity = validity.evaluate(x_input, record, baseline_coordinates=x_input)
        for steps in (1, 2):
            coordinates, diagnostics = rollout_frozen_cartesian(
                teacher,
                record,
                refinement_steps=steps,
                update_scale=float(update_scale),
                max_displacement=float(sampling["max_displacement"]),
                max_coordinate_norm=float(sampling["max_coordinate_norm"]),
                device=device,
                time_schedule_mode="train_range",
                strict_training_range=True,
            )
            schedule = [float(value) for value in diagnostics["inference_time_schedule"]]
            if schedule and max(schedule) > 0.25 + 1.0e-12:
                raise ValueError("Cartesian source schedule exceeds frozen teacher t_max=0.25")
            source_validity = validity.evaluate(coordinates, record, baseline_coordinates=x_input)
            displacement = displacement_metrics(x_input, coordinates)
            torsion = torsion_change_metrics(x_input, coordinates, record)
            score = _severity_score(base_validity, source_validity, displacement, torsion)
            sample = f"{record.get('sample_id')}::cartesian_s{steps}_u{update_scale:.3f}"
            coordinate_path = coordinate_dir / f"{source_index:04d}_s{steps}_u{update_scale:.3f}.pt"
            _write_coordinate(coordinate_path, coordinates, record, {
                "rollout_steps": steps,
                "time_schedule": schedule,
                "update_scale": float(update_scale),
                "teacher_checkpoint_sha256": checkpoint_sha,
                "teacher_config_sha256": config_sha,
                "diagnostics": diagnostics,
            })
            row = _base_row(
                split=split,
                generator_name="Cartesian_teacher_100k",
                checkpoint_sha256=checkpoint_sha,
                config_sha256=config_sha,
                seed=42,
                nfe=steps,
                solver="Cartesian_train_range",
                rollout_steps=steps,
                time_schedule=schedule,
                update_scale=update_scale,
                molecule_id=_molecule_id(record),
                sample_id=sample,
                coordinate_sha256=_tensor_sha(coordinates),
                source_path=path,
                coordinate_path=coordinate_path,
                coordinate_key="coordinates",
                reference_availability=record.get("x_ref_aligned") is not None,
                source_validity=source_validity,
                provenance={
                    "source_kind": "in_domain_cartesian_rollout",
                    "source_sample_id": str(record.get("sample_id")),
                    "source_x_init_hash": str(record.get("x_init_hash", "")),
                    "training_time_range": diagnostics["training_time_range"],
                    "update_scale_applied_once": True,
                    "test_read": False,
                },
            )
            row.update({
                "source_severity_score": score,
                "molecule_rms_displacement": displacement["aligned_rms_displacement"],
                "mean_atom_displacement": displacement["mean_atom_displacement"],
                "max_atom_displacement": displacement["max_atom_displacement"],
                "torsion_change": torsion["torsion_circular_change"],
                "max_rotatable_torsion_change": torsion["max_rotatable_torsion_change"],
            })
            rows.append(row)
    return rows


def _grade_cartesian(train: pd.DataFrame, frame: pd.DataFrame) -> dict[str, float]:
    calibration = train[train.generator_name == "Cartesian_teacher_100k"]["source_severity_score"]
    if len(calibration) < 20:
        raise ValueError("too few train Cartesian records to freeze severity thresholds")
    thresholds = {
        "normal_upper": float(calibration.quantile(0.10)),
        "mild_upper": float(calibration.quantile(0.50)),
        "medium_upper": float(calibration.quantile(0.85)),
    }
    for index, row in frame[frame.generator_name == "Cartesian_teacher_100k"].iterrows():
        schedule = json.loads(row.time_schedule)
        if row.t_max > 0.25 + 1.0e-12 or any(value < 0.0 for value in schedule):
            severity = "out_of_domain_extreme"
        elif row.source_severity_score <= thresholds["normal_upper"]:
            severity = "normal"
        elif row.source_severity_score <= thresholds["mild_upper"]:
            severity = "mild"
        elif row.source_severity_score <= thresholds["medium_upper"]:
            severity = "medium"
        else:
            severity = "severe"
        frame.at[index, "source_severity"] = severity
    return thresholds


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream_cache", type=Path, required=True)
    parser.add_argument("--cartesian_train_cache", type=Path, required=True)
    parser.add_argument("--cartesian_val_cache", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--teacher_config", type=Path, required=True)
    parser.add_argument("--validity_stats", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_molecules", type=int, default=500)
    parser.add_argument("--val_molecules", type=int, default=100)
    parser.add_argument("--train_cartesian_molecules", type=int, default=250)
    parser.add_argument("--val_cartesian_molecules", type=int, default=30)
    parser.add_argument("--train_update_scale", type=float, default=0.50)
    parser.add_argument("--val_update_scale", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not 0.0 < args.train_update_scale <= 1.0 or not 0.0 < args.val_update_scale <= 1.0:
        raise ValueError("Cartesian update scales must be in (0, 1]")
    if args.train_update_scale == args.val_update_scale:
        raise ValueError("validation update scale must be a pre-frozen unseen condition")
    for path in (args.upstream_cache, args.cartesian_train_cache, args.cartesian_val_cache):
        if "test" in {part.lower() for part in path.resolve().parts}:
            raise ValueError(f"test path is forbidden for Stage C sources: {path}")

    upstream_train_count = args.train_molecules - args.train_cartesian_molecules
    upstream_val_count = args.val_molecules - args.val_cartesian_molecules
    if min(upstream_train_count, upstream_val_count) < 1:
        raise ValueError("each split requires both ETFlow and Cartesian molecules")

    validity = ChemicalValidity(args.validity_stats)
    config = yaml.safe_load(args.teacher_config.read_text(encoding="utf-8"))
    if float(config["time_sampling"]["t_max"]) != 0.25:
        raise ValueError("Cartesian teacher config must declare t_max=0.25")
    sampling = dict(config["sampling"])
    teacher = load_frozen_cartesian_teacher(args.teacher_checkpoint, device=args.device)

    cart_train_dir = args.cartesian_train_cache / "train" if (args.cartesian_train_cache / "train").is_dir() else args.cartesian_train_cache
    cart_val_dir = args.cartesian_val_cache / "val" if (args.cartesian_val_cache / "val").is_dir() else args.cartesian_val_cache
    cart_train = _unique_records(cart_train_dir, args.train_cartesian_molecules)
    cart_val = _unique_records(cart_val_dir, args.val_cartesian_molecules)
    upstream_train = _unique_records(
        args.upstream_cache / "train", upstream_train_count,
        excluded={_molecule_id(record) for _, record in cart_train},
    )
    upstream_val = _unique_records(
        args.upstream_cache / "val", upstream_val_count,
        excluded={_molecule_id(record) for _, record in cart_val},
    )

    train_molecules = {_molecule_id(record) for _, record in cart_train + upstream_train}
    val_molecules = {_molecule_id(record) for _, record in cart_val + upstream_val}
    leakage = sorted(train_molecules & val_molecules)
    if leakage:
        raise ValueError(f"train/val molecule leakage: {leakage[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _build_split(
        "train", upstream_train, cart_train, teacher=teacher,
        teacher_checkpoint=args.teacher_checkpoint, teacher_config=args.teacher_config,
        validity=validity, output_dir=args.output_dir, update_scale=args.train_update_scale,
        sampling=sampling, device=args.device,
    )
    val_rows = _build_split(
        "val", upstream_val, cart_val, teacher=teacher,
        teacher_checkpoint=args.teacher_checkpoint, teacher_config=args.teacher_config,
        validity=validity, output_dir=args.output_dir, update_scale=args.val_update_scale,
        sampling=sampling, device=args.device,
    )
    train = pd.DataFrame(train_rows)
    val = pd.DataFrame(val_rows)
    combined = pd.concat([train, val], ignore_index=True)
    severity_thresholds = _grade_cartesian(train, combined)
    train = combined[combined.split == "train"].reset_index(drop=True)
    val = combined[combined.split == "val"].reset_index(drop=True)
    for split, frame in (("train", train), ("val", val)):
        if frame.molecule_id.nunique() != (args.train_molecules if split == "train" else args.val_molecules):
            raise AssertionError(f"{split} molecule count changed during source construction")
        frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "test_used": False,
        "test_paths_read": [],
        "train_val_molecule_leakage": False,
        "same_molecule_cross_split": False,
        "validity_statistics": {
            "path": str(args.validity_stats.resolve()),
            "file_sha256": file_sha256(args.validity_stats),
            "identity_sha256": validity.statistics["identity_sha256"],
            "source": validity.statistics["source"],
            "fallback_hierarchy": {
                "bond": ["detailed", "coarse", "basic", "global"],
                "angle": ["detailed", "coarse", "basic", "global"],
                "ring_planarity": ["detailed", "global"],
                "torsion_prior": ["detailed", "coarse", "global"],
            },
            "minimum_sample_count": validity.config["minimum_sample_count"],
            "threshold": "zero inside frozen [lower, upper]; excess outside",
            "units": validity.statistics["units"],
        },
        "teacher": {
            "checkpoint": str(args.teacher_checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.teacher_checkpoint),
            "config": str(args.teacher_config.resolve()),
            "config_sha256": file_sha256(args.teacher_config),
            "training_time_range": [0.0, 0.25],
        },
        "severity_definition": {
            "joint_inputs": [
                "molecule_rms_displacement", "max_atom_displacement", "bond_outlier",
                "angle_outlier", "ring_outlier", "clash", "torsion_change",
            ],
            "train_only_quantiles": [0.10, 0.50, 0.85],
            "thresholds": severity_thresholds,
            "out_of_domain_rule": "time outside [0,0.25] only",
        },
        "splits": {
            "train": {
                "records": len(train), "molecules": int(train.molecule_id.nunique()),
                "source_counts": _counts(train, "generator_name"),
                "severity_counts": _counts(train, "source_severity"),
                "update_scales": sorted(float(value) for value in train.update_scale.unique()),
                "parquet_sha256": file_sha256(args.output_dir / "train.parquet"),
            },
            "val": {
                "records": len(val), "molecules": int(val.molecule_id.nunique()),
                "source_counts": _counts(val, "generator_name"),
                "severity_counts": _counts(val, "source_severity"),
                "update_scales": sorted(float(value) for value in val.update_scale.unique()),
                "parquet_sha256": file_sha256(args.output_dir / "val.parquet"),
            },
        },
        "source_controls": {
            "out_of_domain_extreme_default_ratio": 0.0,
            "out_of_domain_extreme_max_batch_ratio": 0.05,
            "single_source_max_batch_ratio": 0.30,
        },
        "unseen_condition": {
            "name": f"cartesian_update_scale_{args.val_update_scale:.2f}",
            "kind": "update_scale",
            "train_values": [args.train_update_scale],
            "validation_holdout_values": [args.val_update_scale],
            "frozen_before_training": True,
            "not_present_in_train": True,
        },
    }
    metadata["identity_sha256"] = _canonical_sha(metadata)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
