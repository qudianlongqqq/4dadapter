#!/usr/bin/env python
"""Build and audit medium minimal-validity targets with the frozen Stage C algorithm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch

from etflow.ecir.audit import displacement_metrics, field, file_sha256, torsion_change_metrics
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.minimal_validity_target import MinimalValidityTargetBuilder
from scripts.build_minimal_validity_targets import run_full, run_pilot


SCHEMA_VERSION = "ecir-mvr-medium-minimal-targets-v1"
VALIDITY = (
    "bond_outlier_rate", "bond_outlier_magnitude", "angle_outlier_rate",
    "angle_outlier_magnitude", "ring_bond_outlier_rate",
    "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
    "chirality_preserved", "stereocenter_degenerate_rate",
    "torsion_prior_outlier_score", "total_thresholded_validity_score",
)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_source(row) -> tuple[dict[str, Any], torch.Tensor]:
    record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
    if row.coordinate_path is not None and not pd.isna(row.coordinate_path):
        payload = torch.load(Path(row.coordinate_path), map_location="cpu", weights_only=False)
        coordinates = torch.as_tensor(payload[row.coordinate_key], dtype=torch.float32)
    else:
        coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
    return record, coordinates


def _audit_rows(sources: pd.DataFrame, targets: pd.DataFrame, validity: ChemicalValidity) -> pd.DataFrame:
    targets = targets.set_index("sample_id")
    rows = []
    for row in sources.sort_values(["molecule_id", "sample_id"]).itertuples(index=False):
        target_row = targets.loc[row.sample_id]
        payload = torch.load(Path(target_row.target_cache_path), map_location="cpu", weights_only=False)
        record, x_input = _load_source(row)
        x_target = torch.as_tensor(payload["x_target"], dtype=torch.float32)
        initial = validity.evaluate(x_input, record, baseline_coordinates=x_input)
        target = validity.evaluate(x_target, record, baseline_coordinates=x_input)
        displacement = displacement_metrics(x_input, x_target)
        torsion = torsion_change_metrics(x_input, x_target, record)
        rotatable = int(field(record, "num_rotatable_bonds", 0))
        ring = bool(torch.as_tensor(field(record, "bond_is_in_ring", [])).any())
        metadata = payload["target_metadata"]
        output = {
            "split": str(row.split), "sample_id": str(row.sample_id),
            "molecule_id": str(row.molecule_id), "source": str(row.generator_name),
            "severity": str(row.source_severity), "rotatable_bonds": rotatable,
            "flexibility": "rotatable_le_2" if rotatable <= 2 else ("rotatable_3_5" if rotatable <= 5 else "rotatable_ge_6"),
            "ring_group": "ring" if ring else "non_ring",
            "target_status": str(metadata["target_status"]),
            "stop_reason": str(metadata["stop_reason"]),
            "reference_fallback_used": bool(metadata.get("reference_fallback_used", False)),
            "coordinate_identity_exact": bool(torch.equal(x_input, x_target)),
            "coordinate_max_abs_change": float((x_input - x_target).abs().max()),
            "aligned_rms_displacement": displacement["aligned_rms_displacement"],
            "mean_atom_displacement": displacement["mean_atom_displacement"],
            "max_atom_displacement": displacement["max_atom_displacement"],
            "torsion_change": torsion["torsion_circular_change"],
            "max_rotatable_torsion_change": torsion["max_rotatable_torsion_change"],
        }
        for metric in VALIDITY:
            output[f"initial_{metric}"] = float(initial[metric])
            output[f"target_{metric}"] = float(target[metric])
            output[f"delta_{metric}"] = float(target[metric] - initial[metric])
        output["validity_gain"] = -output["delta_total_thresholded_validity_score"]
        output["validity_gain_per_displacement"] = output["validity_gain"] / max(output["aligned_rms_displacement"], 1e-8)
        rows.append(output)
    return pd.DataFrame(rows)


def _group_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"records": 0, "available": False}
    return {
        "records": int(len(frame)), "molecules": int(frame.molecule_id.nunique()), "available": True,
        "status_counts": {str(k): int(v) for k, v in frame.target_status.value_counts().items()},
        "mean_displacement": float(frame.aligned_rms_displacement.mean()),
        "p95_displacement": float(frame.aligned_rms_displacement.quantile(0.95)),
        "max_atom_displacement": float(frame.max_atom_displacement.max()),
        "mean_max_torsion_change": float(frame.max_rotatable_torsion_change.mean()),
        "validity_gain": float(frame.validity_gain.mean()),
        "validity_gain_per_displacement": float(frame.validity_gain_per_displacement.mean()),
        "severe_clash_delta": float(frame.delta_severe_clash_rate.mean()),
        "chirality_preserved_delta": float(frame.delta_chirality_preserved.mean()),
        "ring_bond_delta": float(frame.delta_ring_bond_outlier_rate.mean()),
        "ring_planarity_delta": float(frame.delta_ring_planarity_outlier_rate.mean()),
    }


def _full_audit(frame: pd.DataFrame, builder: MinimalValidityTargetBuilder, pilot: dict[str, Any]) -> dict[str, Any]:
    local = ("bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate", "ring_planarity_outlier_rate", "clash_penetration")
    true = local + ("severe_clash_rate", "stereocenter_degenerate_rate")
    deltas = {metric: float(frame[f"delta_{metric}"].mean()) for metric in true}
    initial = {metric: float(frame[f"initial_{metric}"].mean()) for metric in local}
    relative = {metric: (-deltas[metric] / max(abs(initial[metric]), 1e-12)) for metric in local}
    clean = frame[frame.target_status == "identity_clean"]
    fallback = frame[frame.target_status == "identity_fallback"]
    high_flex = frame[frame.flexibility == "rotatable_ge_6"]
    ring = frame[frame.ring_group == "ring"]
    cfg = builder.config.__dict__
    criteria = {
        "stage_c_pilot_gate_pass": pilot["decision"] == "PASS",
        "at_least_two_true_validity_metrics_improve": sum(value < -1e-12 for value in deltas.values()) >= 2,
        "one_local_metric_improves_at_least_10_percent": any(value >= 0.10 for value in relative.values()),
        "trust_limits_respected": bool(
            (frame.aligned_rms_displacement <= float(cfg["max_molecule_rms_displacement"]) + 1e-6).all()
            and (frame.max_atom_displacement <= float(cfg["max_atom_displacement"]) + 1e-6).all()
        ),
        "high_flex_torsion_trust_respected": bool(
            (high_flex.max_rotatable_torsion_change <= float(cfg["max_high_flex_torsion_change_rad"]) + 1e-6).all()
        ),
        "severe_clash_not_increased": float(frame.delta_severe_clash_rate.mean()) <= 1e-12,
        "chirality_not_worse": float(frame.delta_chirality_preserved.mean()) >= -1e-12,
        "ring_safety_not_worse": bool(
            float(ring.delta_ring_bond_outlier_rate.mean()) <= 1e-12
            and float(ring.delta_ring_planarity_outlier_rate.mean()) <= 1e-12
        ),
        "clean_identity_preserved": clean.empty or float(clean.coordinate_identity_exact.mean()) >= 0.90,
        "failure_returns_identity": fallback.empty or bool(fallback.coordinate_identity_exact.all()),
        "no_reference_fallback": not bool(frame.reference_fallback_used.any()),
        "positive_validity_gain_per_displacement": float(frame.validity_gain_per_displacement.mean()) > 0.0,
    }
    groups = {"all": _group_summary(frame)}
    for column in ("source", "severity", "flexibility", "ring_group"):
        for value, group in frame.groupby(column):
            groups[f"{column}:{value}"] = _group_summary(group)
    return {
        "schema_version": "ecir-mvr-medium-target-audit-v1",
        "decision": "PASS" if all(criteria.values()) else "NO_GO_MEDIUM_TARGET",
        "test_paths_opened": 0,
        "records": int(len(frame)), "molecules": int(frame.molecule_id.nunique()),
        "success_records": int((frame.target_status == "minimal_validity_success").sum()),
        "fallback_records": int((frame.target_status == "identity_fallback").sum()),
        "clean_identity_records": int((frame.target_status == "identity_clean").sum()),
        "target_status_counts": {str(k): int(v) for k, v in frame.target_status.value_counts().items()},
        "mean_displacement": float(frame.aligned_rms_displacement.mean()),
        "p95_displacement": float(frame.aligned_rms_displacement.quantile(0.95)),
        "max_atom_displacement": float(frame.max_atom_displacement.max()),
        "high_flex_mean_max_torsion_change": float(high_flex.max_rotatable_torsion_change.mean()),
        "validity_gain": float(frame.validity_gain.mean()),
        "validity_gain_per_displacement": float(frame.validity_gain_per_displacement.mean()),
        "metric_deltas": deltas, "relative_improvements": relative,
        "criteria": criteria, "groups": groups,
        "target_builder_config": cfg,
        "stage_c_equivalent_pilot_identity_sha256": pilot["identity_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sources", type=Path, required=True)
    parser.add_argument("--val_sources", type=Path, required=True)
    parser.add_argument("--source_metadata", type=Path, required=True)
    parser.add_argument("--validity_stats", type=Path, required=True)
    parser.add_argument("--target_output_dir", type=Path, default=Path("data/ecir_mvr/medium/minimal_targets"))
    parser.add_argument("--audit_dir", type=Path, default=Path("diagnostics/ecir_mvr/medium/run_a_seed42_20k"))
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    for path in (args.train_sources, args.val_sources, args.source_metadata, args.validity_stats):
        if "test" in {part.lower() for part in path.resolve().parts}:
            raise ValueError("test paths are forbidden")
    train, val = pd.read_parquet(args.train_sources), pd.read_parquet(args.val_sources)
    if train.molecule_id.nunique() != 5000 or val.molecule_id.nunique() != 500:
        raise ValueError("medium source manifests must contain 5000/500 molecules")
    if set(train.molecule_id) & set(val.molecule_id):
        raise ValueError("train/validation molecule overlap")
    validity = ChemicalValidity(args.validity_stats)
    if validity.statistics["identity_sha256"] != "66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3":
        raise ValueError("frozen validity statistics identity changed")
    builder = MinimalValidityTargetBuilder(validity, {"learning_rate": 0.001, "max_steps": 40})
    args.target_output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(
        target_output_dir=args.target_output_dir,
        pilot_csv=args.audit_dir / "medium_target_pilot.csv",
        summary_json=args.audit_dir / "medium_target_pilot_summary.json",
    )
    if args.audit_only:
        if not ns.summary_json.is_file() or not all(
            (args.target_output_dir / f"{split}.parquet").is_file() for split in ("train", "val")
        ):
            raise FileNotFoundError("--audit_only requires completed pilot and full target manifests")
        pilot = json.loads(ns.summary_json.read_text(encoding="utf-8"))
    else:
        pilot = run_pilot(ns, train, validity, builder)
        if pilot["decision"] != "PASS":
            metadata = {
                "schema_version": SCHEMA_VERSION, "decision": "NO_GO_MEDIUM_TARGET",
                "test_paths_opened": 0, "pilot": pilot,
            }
            metadata["medium_target_identity_sha256"] = _canonical_sha(metadata)
            (args.target_output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise SystemExit("NO_GO_MEDIUM_TARGET: Stage C-equivalent pilot gate failed")
        run_full(ns, {"train": train, "val": val}, validity, builder)
    audit_frames = []
    for split, sources in (("train", train), ("val", val)):
        targets = pd.read_parquet(args.target_output_dir / f"{split}.parquet")
        audit_frames.append(_audit_rows(sources, targets, validity))
    audit_frame = pd.concat(audit_frames, ignore_index=True)
    audit = _full_audit(audit_frame, builder, pilot)
    audit_frame.to_csv(args.audit_dir / "target_audit.csv", index=False)
    (args.audit_dir / "target_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    source_metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "decision": audit["decision"], "test_paths_opened": 0,
        "algorithm": "Stage C MinimalValidityTargetBuilder, unchanged parameters and safety rules",
        "validity_statistics_identity_sha256": validity.statistics["identity_sha256"],
        "medium_real_source_identity_sha256": source_metadata["medium_real_source_identity_sha256"],
        "target_builder_config": builder.config.__dict__,
        "pilot_identity_sha256": pilot["identity_sha256"],
        "audit_sha256": file_sha256(args.audit_dir / "target_audit.json"),
        "splits": {
            split: {
                "records": int(len(pd.read_parquet(args.target_output_dir / f"{split}.parquet"))),
                "molecules": int(pd.read_parquet(args.target_output_dir / f"{split}.parquet").molecule_id.nunique()),
                "parquet_sha256": file_sha256(args.target_output_dir / f"{split}.parquet"),
            } for split in ("train", "val")
        },
    }
    metadata["medium_target_identity_sha256"] = _canonical_sha(metadata)
    (args.target_output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if audit["decision"] != "PASS":
        raise SystemExit("NO_GO_MEDIUM_TARGET: full target audit failed")


if __name__ == "__main__":
    main()
