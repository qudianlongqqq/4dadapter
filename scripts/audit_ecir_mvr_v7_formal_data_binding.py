#!/usr/bin/env python3
"""Audit formal-validation pairing and compare it with the frozen 10K cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.acceptance import displacement_metrics  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v7_formal import file_sha256  # noqa: E402
from etflow.ecir.run_a_evaluation import nearest_rmsd  # noqa: E402
from scripts.run_ecir_mvr_v7_10k_validation import _build_items  # noqa: E402
from scripts.run_ecir_mvr_v7_formal_validation import (  # noqa: E402
    SOURCE_MANIFEST_SHA256,
    TARGET_MANIFEST_SHA256,
    _validate_cohort_frames,
)


ISOLATION = {
    "test_records_read": 0,
    "test_assets_opened": False,
    "frozen_holdout_records_opened": 0,
    "formal_test_run": False,
    "training_performed": False,
    "target_rematerialization": False,
}


def _stats(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _identity(frame: pd.DataFrame) -> str:
    payload = frame[["sample_id", "molecule_id"]].sort_values("sample_id")
    encoded = json.dumps(
        payload.astype(str).values.tolist(), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution(
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    source_rows = metrics.loc[metrics.method.isin(["Source", "upstream"])].copy()
    source_rows = source_rows.drop_duplicates("sample_id").set_index("sample_id")
    selected = sources.set_index("sample_id").loc[source_rows.index]
    target = targets.set_index("sample_id").loc[source_rows.index]
    return {
        "records": len(source_rows),
        "molecules": int(source_rows.molecule_id.nunique()),
        "atom_count": _stats(selected.num_atoms),
        "rotatable_bond_count": _stats(selected.num_rotatable_bonds),
        "source_bond_outlier_rate": _stats(source_rows.bond_outlier_rate),
        "source_angle_outlier_rate": _stats(source_rows.angle_outlier_rate),
        "active_angle_fraction": float((source_rows.angle_outlier_rate > 0.0).mean()),
        "source_ring_bond_outlier_rate": _stats(source_rows.ring_bond_outlier_rate),
        "source_clash_penetration": _stats(source_rows.clash_penetration),
        "source_target_rms": _stats(target.initial_to_target_rmsd),
        "target_max_atom_displacement": _stats(target.max_atom_displacement),
        "target_identity_fraction": float((target.initial_to_target_rmsd == 0.0).mean()),
    }


def _sample_audit(
    cohort: str,
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    source_cache_root: Path,
    target_cache_root: Path,
    validity: ChemicalValidity,
    count: int,
) -> list[dict[str, Any]]:
    ordered = sources.assign(
        audit_order=sources.sample_id.map(
            lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        )
    ).sort_values("audit_order")
    selected = ordered.head(count).drop(columns="audit_order")
    selected_targets = targets.set_index("sample_id").loc[selected.sample_id].reset_index()
    items = _build_items(
        selected,
        selected_targets,
        validity,
        source_cache_root=source_cache_root,
        target_cache_root=target_cache_root,
    )
    target_by_sample = selected_targets.set_index("sample_id")
    rows = []
    for item in items:
        sample_id = str(item["row"].sample_id)
        target_row = target_by_sample.loc[sample_id]
        target_path = target_cache_root / str(target_row.split) / Path(
            str(target_row.target_cache_path)
        ).name
        payload = torch.load(target_path, map_location="cpu", weights_only=False)
        source = item["input"]
        target = item["minimal_target"]
        before = item["input_validity"]
        after = validity.evaluate(target, item["record"], baseline_coordinates=source)
        source_atomic = torch.as_tensor(item["record"]["atomic_numbers"]).cpu()
        payload_atomic = torch.as_tensor(payload["source_atomic_numbers"]).cpu()
        payload_input = torch.as_tensor(payload["x_input"], dtype=torch.float32)
        displacement = displacement_metrics(source, target)
        rows.append(
            {
                "cohort": cohort,
                "sample_id": sample_id,
                "molecule_id": str(item["row"].molecule_id),
                "source_target_rms": displacement["aligned_rms_displacement"],
                "source_target_max": displacement["max_atom_displacement"],
                "source_reference_rms": nearest_rmsd(source, item["references"]),
                "target_reference_rms": nearest_rmsd(target, item["references"]),
                "bond_difference": float(
                    after["bond_outlier_rate"] - before["bond_outlier_rate"]
                ),
                "angle_difference": float(
                    after["angle_outlier_rate"] - before["angle_outlier_rate"]
                ),
                "ring_difference": float(
                    after["ring_bond_outlier_rate"]
                    - before["ring_bond_outlier_rate"]
                ),
                "payload_input_matches_source": bool(
                    torch.equal(payload_input, source)
                ),
                "atomic_order_matches": bool(torch.equal(source_atomic, payload_atomic)),
                "coordinate_shape_matches": bool(source.shape == target.shape),
                "source_file_sha_matches": bool(
                    str(payload["source_file_sha256"])
                    == str(target_row.source_file_sha256)
                    == str(item["row"].source_file_sha256)
                ),
                "source_coordinate_sha_matches": bool(
                    str(payload["source_coordinate_sha256"])
                    == str(target_row.source_coordinate_sha256)
                    == str(item["row"].coordinate_sha256)
                ),
                "target_test_records_read": int(payload["test_records_read"]),
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-sources", type=Path, required=True)
    parser.add_argument("--formal-targets", type=Path, required=True)
    parser.add_argument("--development-sources", type=Path, required=True)
    parser.add_argument("--development-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--formal-record-metrics", type=Path, required=True)
    parser.add_argument("--development-record-metrics", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.output_dir.mkdir(parents=True)
    formal_sources = pd.read_parquet(args.formal_sources)
    formal_targets = pd.read_parquet(args.formal_targets)
    development_sources = pd.read_parquet(args.development_sources)
    development_targets = pd.read_parquet(args.development_targets)
    _validate_cohort_frames(formal_sources, formal_targets)
    if file_sha256(args.formal_sources) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("formal source manifest SHA mismatch")
    if file_sha256(args.formal_targets) != TARGET_MANIFEST_SHA256:
        raise RuntimeError("formal target manifest SHA mismatch")
    for name, sources, targets in (
        ("formal", formal_sources, formal_targets),
        ("development", development_sources, development_targets),
    ):
        if sources.sample_id.duplicated().any() or targets.sample_id.duplicated().any():
            raise RuntimeError(f"{name} duplicate sample identity")
        source_pairs = sources.set_index("sample_id").molecule_id.astype(str).sort_index()
        target_pairs = targets.set_index("sample_id").molecule_id.astype(str).sort_index()
        if not source_pairs.equals(target_pairs):
            raise RuntimeError(f"{name} source/target pairing mismatch")
        if bool(sources.test_record.astype(bool).any()) or bool(
            targets.test_records_read.astype(bool).any()
        ):
            raise RuntimeError(f"{name} includes test records")
        if not (
            sources.set_index("sample_id").coordinate_sha256.astype(str).sort_index()
            == targets.set_index("sample_id").source_coordinate_sha256.astype(str).sort_index()
        ).all():
            raise RuntimeError(f"{name} coordinate identity mismatch")
        if not (
            sources.set_index("sample_id").source_file_sha256.astype(str).sort_index()
            == targets.set_index("sample_id").source_file_sha256.astype(str).sort_index()
        ).all():
            raise RuntimeError(f"{name} source file identity mismatch")
    if set(formal_sources.split.astype(str)) != {"val"}:
        raise RuntimeError("formal audit loaded a non-validation split")
    validity = ChemicalValidity(args.validity_statistics)
    formal_metrics = pd.read_csv(args.formal_record_metrics)
    development_metrics = pd.read_csv(args.development_record_metrics)
    sample_rows = [
        *_sample_audit(
            "formal",
            formal_sources,
            formal_targets,
            source_cache_root=args.source_cache_root,
            target_cache_root=args.target_cache_root,
            validity=validity,
            count=args.sample_count,
        ),
        *_sample_audit(
            "development",
            development_sources,
            development_targets,
            source_cache_root=args.source_cache_root,
            target_cache_root=args.target_cache_root,
            validity=validity,
            count=args.sample_count,
        ),
    ]
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(args.output_dir / "sample_binding_audit.csv", index=False)
    boolean_columns = [
        "payload_input_matches_source",
        "atomic_order_matches",
        "coordinate_shape_matches",
        "source_file_sha_matches",
        "source_coordinate_sha_matches",
    ]
    summary = {
        "schema_version": "mcvr-v7-formal-data-binding-audit-v1",
        "formal": {
            "manifest_identity_sha256": _identity(formal_sources),
            "distribution": _distribution(
                formal_sources, formal_targets, formal_metrics
            ),
        },
        "development": {
            "manifest_identity_sha256": _identity(development_sources),
            "distribution": _distribution(
                development_sources, development_targets, development_metrics
            ),
        },
        "builder_code_same": bool(
            set(formal_targets.builder_code_sha256)
            == set(development_targets.builder_code_sha256)
        ),
        "builder_config_same": bool(
            set(formal_targets.builder_config_sha256)
            == set(development_targets.builder_config_sha256)
        ),
        "sample_checks": {
            name: bool(samples[name].all()) for name in boolean_columns
        },
        "sample_test_records_read": int(samples.target_test_records_read.sum()),
        **ISOLATION,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = ("sample_binding_audit.csv", "summary.json")
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{file_sha256(args.output_dir / name)}  {name}\n" for name in files),
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
