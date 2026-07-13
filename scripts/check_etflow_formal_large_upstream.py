#!/usr/bin/env python
"""Check ETFlow formal-large train/val shards and the frozen test output."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

try:
    from generate_etflow_formal_large_upstream import (
        atomic_json_save,
        load_and_validate_generated_file,
        validate_generation_manifest,
    )
except ModuleNotFoundError:
    from scripts.generate_etflow_formal_large_upstream import (
        atomic_json_save,
        load_and_validate_generated_file,
        validate_generation_manifest,
    )


FORMAL_EXPECTED = {
    "train": {"molecules": 50_000, "conformers": 150_000},
    "val": {"molecules": 5_000, "conformers": 10_000},
    "test": {"molecules": 100},
}


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _test_records(path: Path) -> Iterator[Any]:
    if path.is_dir():
        for item in sorted(path.rglob("*.pt")):
            yield torch.load(item, map_location="cpu", weights_only=False)
        return
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(payload.get("molecules"), list):
        payload = payload["molecules"]
    if isinstance(payload, (list, tuple)):
        yield from payload
    else:
        yield payload


def inspect_generated_split(root: Path, split: str) -> tuple[dict[str, Any], set[str]]:
    manifest_path = root / "generation_manifest.json"
    molecules_dir = root / "molecules"
    errors = []
    missing_files = []
    duplicate_source_ids = []
    duplicate_dataset_indices = []
    source_ids: set[str] = set()
    dataset_indices: set[int] = set()
    conformers = 0
    completed = 0
    manifest_hash = None
    target = 0
    checkpoint_hash = None
    config_hash = None
    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")
        return (
            {
                "split": split,
                "actual_molecules": 0,
                "actual_generated_conformers": 0,
                "target_molecules": 0,
                "completed_fraction": 0.0,
                "manifest_sha256": None,
                "missing_files": [str(manifest_path)],
                "duplicate_source_mol_ids": [],
                "duplicate_dataset_indices": [],
                "unexpected_files": [],
                "error_count": 1,
                "errors": errors,
            },
            set(),
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_generation_manifest(manifest)
        if str(manifest["split"]) != split:
            raise ValueError(f"Manifest split is {manifest['split']!r}, expected {split!r}")
        target = int(manifest["target_molecules"])
        manifest_hash = str(manifest["manifest_sha256"])
        checkpoint_hash = str(manifest["checkpoint_sha256"])
        config_hash = str(manifest["config_sha256"])
    except Exception as exc:
        errors.append(f"manifest validation: {exc}")
        manifest = {"records": []}
    expected_files = {
        str((molecules_dir / str(row["output_file"])).resolve())
        for row in manifest.get("records", [])
    }
    unexpected_files = [
        str(path)
        for path in sorted(molecules_dir.rglob("*.pt"))
        if str(path.resolve()) not in expected_files
    ] if molecules_dir.is_dir() else []
    for row in manifest.get("records", []):
        path = molecules_dir / str(row["output_file"])
        if not path.is_file():
            missing_files.append(str(path))
            continue
        try:
            checked = load_and_validate_generated_file(
                path, manifest=manifest, manifest_row=row
            )
            completed += 1
            conformers += int(checked["generated_conformers"])
            source_id = str(checked["source_mol_id"])
            index = int(checked["dataset_index"])
            if source_id in source_ids:
                duplicate_source_ids.append(source_id)
            source_ids.add(source_id)
            if index in dataset_indices:
                duplicate_dataset_indices.append(index)
            dataset_indices.add(index)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if duplicate_source_ids:
        errors.append("duplicate source_mol_id values")
    if duplicate_dataset_indices:
        errors.append("duplicate dataset_index values")
    if missing_files:
        errors.append(f"missing {len(missing_files)} molecule files")
    if unexpected_files:
        errors.append(f"found {len(unexpected_files)} unmanifested molecule files")
    return (
        {
            "split": split,
            "actual_molecules": completed,
            "actual_generated_conformers": conformers,
            "target_molecules": target,
            "completed_fraction": completed / target if target else 0.0,
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": config_hash,
            "missing_files": missing_files[:100],
            "missing_file_count": len(missing_files),
            "unexpected_files": unexpected_files[:100],
            "unexpected_file_count": len(unexpected_files),
            "duplicate_source_mol_ids": sorted(set(duplicate_source_ids))[:100],
            "duplicate_dataset_indices": sorted(set(duplicate_dataset_indices))[:100],
            "error_count": len(errors),
            "errors": errors[:200],
        },
        source_ids,
    )


def inspect_test_output(path: Path) -> tuple[dict[str, Any], set[str]]:
    source_ids = set()
    duplicate_ids = []
    errors = []
    count = 0
    if not path.exists():
        return (
            {
                "actual_molecules": 0,
                "duplicate_source_mol_ids": [],
                "error_count": 1,
                "errors": [f"missing test output: {path}"],
            },
            set(),
        )
    try:
        for index, record in enumerate(_test_records(path)):
            source_id = str(
                _field(record, "source_mol_id", _field(record, "mol_id", ""))
            )
            if not source_id:
                errors.append(f"test record {index} has no stable source_mol_id")
                continue
            if source_id in source_ids:
                duplicate_ids.append(source_id)
            source_ids.add(source_id)
            count += 1
            pos_gen = _field(record, "pos_gen")
            pos_ref = _field(record, "pos_ref")
            if pos_gen is None or pos_ref is None:
                errors.append(f"test record {index} lacks pos_gen/pos_ref")
                continue
            generated = torch.as_tensor(pos_gen)
            reference = torch.as_tensor(pos_ref)
            if generated.ndim != 3 or generated.size(-1) != 3:
                errors.append(f"test record {index} has invalid pos_gen shape")
            if reference.ndim not in (2, 3) or reference.size(-1) != 3:
                errors.append(f"test record {index} has invalid pos_ref shape")
            if not bool(torch.isfinite(generated).all()) or not bool(
                torch.isfinite(reference).all()
            ):
                errors.append(f"test record {index} has non-finite coordinates")
    except Exception as exc:
        errors.append(f"test output load failed: {exc}")
    if duplicate_ids:
        errors.append("duplicate test source_mol_id values")
    return (
        {
            "actual_molecules": count,
            "duplicate_source_mol_ids": sorted(set(duplicate_ids))[:100],
            "error_count": len(errors),
            "errors": errors[:200],
        },
        source_ids,
    )


def build_integrity_report(train: Path, val: Path, test: Path) -> dict[str, Any]:
    train_report, train_ids = inspect_generated_split(train, "train")
    val_report, val_ids = inspect_generated_split(val, "val")
    test_report, test_ids = inspect_test_output(test)
    overlap = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }
    errors = (
        int(train_report["error_count"])
        + int(val_report["error_count"])
        + int(test_report["error_count"])
        + sum(bool(values) for values in overlap.values())
    )
    ready = (
        train_report["actual_molecules"] == FORMAL_EXPECTED["train"]["molecules"]
        and train_report["actual_generated_conformers"]
        == FORMAL_EXPECTED["train"]["conformers"]
        and val_report["actual_molecules"] == FORMAL_EXPECTED["val"]["molecules"]
        and val_report["actual_generated_conformers"]
        == FORMAL_EXPECTED["val"]["conformers"]
        and test_report["actual_molecules"] == FORMAL_EXPECTED["test"]["molecules"]
        and errors == 0
    )
    return {
        "train": train_report,
        "val": val_report,
        "test": test_report,
        "overlap": {key: values[:100] for key, values in overlap.items()},
        "overlap_counts": {key: len(values) for key, values in overlap.items()},
        "error_count": errors,
        "can_enter_build_formal_large_data": ready,
        "status": "FORMAL_LARGE_UPSTREAM_READY" if ready else "INCOMPLETE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", type=Path, default=Path("data/upstream_formal_large/train")
    )
    parser.add_argument(
        "--val_dir", type=Path, default=Path("data/upstream_formal_large/val")
    )
    parser.add_argument(
        "--test_output",
        type=Path,
        default=Path("data/upstream_formal_small/test/generated_files.pkl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/formal_large_upstream_integrity.json"),
    )
    args = parser.parse_args()
    report = build_integrity_report(args.train_dir, args.val_dir, args.test_output)
    atomic_json_save(report, args.report)
    print(json.dumps(report, indent=2))
    if report["can_enter_build_formal_large_data"]:
        print("FORMAL_LARGE_UPSTREAM_READY")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
