#!/usr/bin/env python
"""Fail-closed data and identity audit for the frozen Stage 2b Run A."""

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

from etflow.ecir.mvr_dataset import balanced_sample_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(values) -> dict[str, int]:
    return {str(key): int(value) for key, value in pd.Series(values).value_counts().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    frozen = config["frozen_identities"]
    paths = {name: Path(data[name]) for name in (
        "train_sources", "val_sources", "train_targets", "val_targets",
        "validity_statistics",
    )}
    if any("test" in {part.lower() for part in path.resolve().parts} for path in paths.values()):
        raise ValueError("Run A configuration may not name a test path")

    stage_c = _load_json(Path("diagnostics/ecir_mvr/stage_c/decision.json"))
    source_metadata = _load_json(paths["train_sources"].parent / "metadata.json")
    validity = _load_json(paths["validity_statistics"])
    expected = {
        "stage_c_decision": stage_c["decision"],
        "target_gate_identity_sha256": stage_c["target_gate_identity_sha256"],
        "real_source_identity_sha256": source_metadata["identity_sha256"],
        "validity_statistics_identity_sha256": validity["identity_sha256"],
    }
    if expected != frozen:
        raise ValueError(f"frozen identity mismatch: expected={expected}, config={frozen}")
    if stage_c["target_gate"] != "PASS" or stage_c["test_used_for_selection"]:
        raise ValueError("Stage C gate is not a test-free PASS")
    if stage_c["20k_permitted"] or stage_c["100k_permitted"]:
        raise ValueError("long-training permission unexpectedly enabled")

    train = pd.read_parquet(paths["train_sources"])
    val = pd.read_parquet(paths["val_sources"])
    train_targets = pd.read_parquet(paths["train_targets"])
    val_targets = pd.read_parquet(paths["val_targets"])
    if set(train.split) != {"train"} or set(val.split) != {"val"}:
        raise ValueError("source split labels are invalid")
    overlap = sorted(set(train.molecule_id) & set(val.molecule_id))
    if overlap:
        raise ValueError(f"train/val molecule leakage: {overlap[:5]}")
    if train.molecule_id.nunique() != int(data["train_molecules"]):
        raise ValueError("train molecule count mismatch")
    if val.molecule_id.nunique() != int(data["val_molecules"]):
        raise ValueError("validation molecule count mismatch")
    if set(train.sample_id) != set(train_targets.sample_id):
        raise ValueError("train source/target sample identities differ")
    if set(val.sample_id) != set(val_targets.sample_id):
        raise ValueError("validation source/target sample identities differ")

    train_scales = sorted(float(value) for value in train[train.update_scale > 0].update_scale.unique())
    val_scales = sorted(float(value) for value in val[val.update_scale > 0].update_scale.unique())
    if train_scales != [0.5] or val_scales != [0.35]:
        raise ValueError(f"unseen scale identity failed: train={train_scales}, val={val_scales}")
    if (train.source_severity == "out_of_domain_extreme").any() or (
        val.source_severity == "out_of_domain_extreme"
    ).any():
        raise ValueError("out_of_domain_extreme entered Run A sources")

    plan = balanced_sample_plan(
        train, int(data["train_epoch_size"]), ratios=data["mixture"],
        synthetic_ratios=data["synthetic_mixture"], seed=int(config["seed"]),
        out_of_domain_extreme_ratio=float(data["out_of_domain_extreme_fraction"]),
    )
    type_counts = _counts(item["sample_type"] for item in plan)
    source_counts = _counts(item["source"] for item in plan if item["sample_type"] == "real_error")
    severity_counts = _counts(item["severity"] for item in plan if item["sample_type"] == "real_error")
    max_source_fraction = max(source_counts.values()) / len(plan)
    if max_source_fraction > float(data["max_single_source_fraction"]) + 1e-12:
        raise ValueError("source-balanced plan exceeds the single-source limit")

    molecule_flags = {}
    for frame in (train, val):
        for row in frame.sort_values("sample_id").itertuples(index=False):
            if row.molecule_id in molecule_flags:
                continue
            record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
            molecule_flags[str(row.molecule_id)] = {
                "split": str(row.split),
                "high_flex": int(record.get("num_rotatable_bonds", 0)) >= 6,
                "ring": bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any()),
            }
    train_flags = [value for value in molecule_flags.values() if value["split"] == "train"]
    val_flags = [value for value in molecule_flags.values() if value["split"] == "val"]

    audit = {
        "schema_version": "ecir-mvr-stage2b-run-a-data-audit-v1",
        "status": "PASS",
        "config": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "identities": expected,
        "source_metadata_file_sha256": _sha(paths["train_sources"].parent / "metadata.json"),
        "validity_statistics_file_sha256": _sha(paths["validity_statistics"]),
        "train": {
            "molecules": int(train.molecule_id.nunique()), "records": len(train),
            "source_records": _counts(train.generator_name),
            "severity_records": _counts(train.source_severity),
            "target_status": _counts(train_targets.target_status),
            "high_flex_molecule_fraction": sum(v["high_flex"] for v in train_flags) / len(train_flags),
            "ring_molecule_fraction": sum(v["ring"] for v in train_flags) / len(train_flags),
            "cartesian_update_scales": train_scales,
        },
        "val": {
            "molecules": int(val.molecule_id.nunique()), "records": len(val),
            "source_records": _counts(val.generator_name),
            "severity_records": _counts(val.source_severity),
            "target_status": _counts(val_targets.target_status),
            "high_flex_molecule_fraction": sum(v["high_flex"] for v in val_flags) / len(val_flags),
            "ring_molecule_fraction": sum(v["ring"] for v in val_flags) / len(val_flags),
            "cartesian_update_scales": val_scales,
        },
        "training_plan": {
            "records": len(plan), "sample_type_counts": type_counts,
            "sample_type_fraction": {key: value / len(plan) for key, value in type_counts.items()},
            "real_source_counts": source_counts,
            "real_source_fraction_of_batch": {key: value / len(plan) for key, value in source_counts.items()},
            "severity_counts": severity_counts,
            "max_single_source_fraction": max_source_fraction,
            "out_of_domain_extreme_records": sum(item["severity"] == "out_of_domain_extreme" for item in plan),
        },
        "test_records_read": 0,
        "test_paths_read": [],
        "train_val_molecule_intersection": overlap,
        "unseen_update_scale_0_35_validation_only": True,
        "protected_file_tracked": False,
    }
    audit["identity_sha256"] = _canonical(audit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# MCVR Stage 2b Run A data audit", "", "Status: `PASS`.", "",
        f"- train: {audit['train']['molecules']} molecules / {audit['train']['records']} records",
        f"- val: {audit['val']['molecules']} molecules / {audit['val']['records']} records",
        f"- mixture: {audit['training_plan']['sample_type_fraction']}",
        f"- real sources as total batch fraction: {audit['training_plan']['real_source_fraction_of_batch']}",
        f"- severity counts: {audit['training_plan']['severity_counts']}",
        f"- target status train: {audit['train']['target_status']}",
        f"- target status val: {audit['val']['target_status']}",
        f"- train/val molecule intersection: {audit['train_val_molecule_intersection']}",
        "- unseen update scale: train 0.50; validation-only 0.35",
        "- test records read: 0",
        f"- audit identity: `{audit['identity_sha256']}`",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
