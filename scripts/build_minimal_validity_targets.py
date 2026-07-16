#!/usr/bin/env python
"""Build and gate Stage C minimal-validity targets before any MCVR training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import torch

from etflow.ecir.audit import displacement_metrics, field, torsion_change_metrics
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.minimal_validity_target import MinimalValidityTargetBuilder
from etflow.ecir.target_building import build_real_error_target


TRUE_METRICS = (
    "bond_outlier_rate",
    "angle_outlier_rate",
    "ring_bond_outlier_rate",
    "ring_planarity_outlier_rate",
    "clash_penetration",
    "severe_clash_rate",
    "stereocenter_degenerate_rate",
    "total_thresholded_validity_score",
)
LOCAL_METRICS = (
    "bond_outlier_rate",
    "angle_outlier_rate",
    "ring_bond_outlier_rate",
    "ring_planarity_outlier_rate",
    "clash_penetration",
)


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_input(row) -> tuple[dict[str, Any], torch.Tensor]:
    record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
    if row.coordinate_path is not None and not pd.isna(row.coordinate_path):
        payload = torch.load(Path(row.coordinate_path), map_location="cpu", weights_only=False)
        coordinates = torch.as_tensor(payload[row.coordinate_key], dtype=torch.float32)
    else:
        coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
    digest = hashlib.sha256(coordinates.contiguous().numpy().tobytes()).hexdigest()
    if digest != row.coordinate_sha256:
        raise ValueError(f"coordinate identity mismatch: {row.sample_id}")
    return record, coordinates


def _flags(record: dict[str, Any]) -> tuple[bool, bool]:
    high_flex = int(field(record, "num_rotatable_bonds", 0)) >= 6
    ring = bool(torch.as_tensor(field(record, "bond_is_in_ring", [])).any())
    return high_flex, ring


def _pilot_items(frame: pd.DataFrame, validity: ChemicalValidity) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_samples = set()

    def add_rows(rows, category, limit):
        added = 0
        for row in rows.itertuples(index=False):
            if row.sample_id in used_samples:
                continue
            record, coordinates = _load_input(row)
            high_flex, ring = _flags(record)
            selected.append({
                "row": row, "record": record, "coordinates": coordinates,
                "pilot_category": category, "high_flex": high_flex, "ring": ring,
                "clean_input": False,
            })
            used_samples.add(row.sample_id)
            added += 1
            if added == limit:
                return
        raise ValueError(f"pilot has only {added}/{limit} records for {category}")

    ordered = frame.sort_values(["molecule_id", "sample_id"])
    add_rows(ordered[ordered.generator_name == "ETFlow_formal_upstream"], "ETFlow_normal", 50)
    add_rows(ordered[ordered.source_severity == "mild"], "Cartesian_mild", 50)
    add_rows(ordered[ordered.source_severity == "medium"], "Cartesian_medium", 50)

    for flag, category in (("high_flex", "high_flex_supplement"), ("ring", "ring_supplement")):
        existing = sum(bool(item[flag]) for item in selected)
        needed = max(0, 20 - existing)
        if needed:
            candidates = []
            for row in ordered.itertuples(index=False):
                if row.sample_id in used_samples:
                    continue
                record, coordinates = _load_input(row)
                high_flex, ring = _flags(record)
                if (high_flex if flag == "high_flex" else ring):
                    candidates.append((row, record, coordinates, high_flex, ring))
                if len(candidates) == needed:
                    break
            if len(candidates) != needed:
                raise ValueError(f"pilot cannot satisfy 20 {flag} records")
            for row, record, coordinates, high_flex, ring in candidates:
                selected.append({
                    "row": row, "record": record, "coordinates": coordinates,
                    "pilot_category": category, "high_flex": high_flex, "ring": ring,
                    "clean_input": False,
                })
                used_samples.add(row.sample_id)

    # Clean controls are frozen train reference conformers. They are not real-error
    # labels, and are used only to verify the required identity behavior.
    clean_count = 0
    seen_clean_molecules = set()
    for row in ordered.itertuples(index=False):
        if clean_count == 20:
            break
        record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
        molecule = str(row.molecule_id)
        if molecule in seen_clean_molecules:
            continue
        references = torch.as_tensor(
            record.get("x_ref_candidates", record.get("x_ref_aligned")), dtype=torch.float32
        )
        if references.ndim == 2:
            references = references.unsqueeze(0)
        for reference_index, coordinates in enumerate(references):
            values = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
            clean = all(values[name] <= 0.0 for name in (
                "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
                "ring_planarity_outlier_rate", "clash_penetration", "severe_clash_rate",
            )) and values["stereocenter_degenerate_rate"] <= 0.0
            if clean:
                high_flex, ring = _flags(record)
                selected.append({
                    "row": row, "record": record, "coordinates": coordinates.clone(),
                    "pilot_category": "clean_reference", "high_flex": high_flex,
                    "ring": ring, "clean_input": True,
                    "sample_id_override": f"{row.sample_id}::clean_ref{reference_index}",
                })
                seen_clean_molecules.add(molecule)
                clean_count += 1
                break
    if clean_count < 20:
        raise ValueError(f"pilot has only {clean_count}/20 clean validity controls")
    return selected


def _target_metrics(prefix, x_input, target, record, validity):
    values = validity.evaluate(target, record, baseline_coordinates=x_input)
    displacement = displacement_metrics(x_input, target)
    torsion = torsion_change_metrics(x_input, target, record)
    result = {f"{prefix}_{name}": float(value) for name, value in values.items()}
    result.update({
        f"{prefix}_aligned_displacement": displacement["aligned_rms_displacement"],
        f"{prefix}_mean_atom_displacement": displacement["mean_atom_displacement"],
        f"{prefix}_max_atom_displacement": displacement["max_atom_displacement"],
        f"{prefix}_torsion_change": torsion["torsion_circular_change"],
        f"{prefix}_max_rotatable_torsion_change": torsion["max_rotatable_torsion_change"],
    })
    return result


def _mean(frame, name):
    return float(frame[name].mean())


def _pilot_summary(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for metric in TRUE_METRICS:
        before = _mean(frame, f"input_{metric}")
        minimal = _mean(frame, f"minimal_{metric}")
        old = _mean(frame, f"old_{metric}")
        comparisons[metric] = {
            "input": before,
            "old_target": old,
            "minimal_target": minimal,
            "minimal_delta_vs_input": minimal - before,
            "minimal_relative_improvement_vs_input": (
                (before - minimal) / max(abs(before), 1.0e-12)
            ),
        }
    nonclean = frame[~frame.clean_input]
    clean = frame[frame.clean_input]
    high_flex = frame[frame.high_flex]
    minimal_gain = (
        nonclean.input_total_thresholded_validity_score
        - nonclean.minimal_total_thresholded_validity_score
    )
    old_gain = (
        nonclean.input_total_thresholded_validity_score
        - nonclean.old_total_thresholded_validity_score
    )
    minimal_efficiency = float((minimal_gain / nonclean.minimal_aligned_displacement.clip(lower=1e-8)).mean())
    old_efficiency = float((old_gain / nonclean.old_aligned_displacement.clip(lower=1e-8)).mean())
    improved_metrics = [
        metric for metric in TRUE_METRICS[:-1]
        if comparisons[metric]["minimal_delta_vs_input"] < -1.0e-12
    ]
    ten_percent = [
        metric for metric in LOCAL_METRICS
        if comparisons[metric]["minimal_relative_improvement_vs_input"] >= 0.10
    ]
    trust_rms = float(config["max_molecule_rms_displacement"])
    trust_atom = float(config["max_atom_displacement"])
    criteria = {
        "at_least_two_true_validity_metrics_improve": len(improved_metrics) >= 2,
        "one_local_metric_improves_at_least_10_percent": bool(ten_percent),
        "mean_displacement_below_old_target": _mean(nonclean, "minimal_aligned_displacement") < _mean(nonclean, "old_aligned_displacement"),
        "p95_displacement_below_old_target": float(nonclean.minimal_aligned_displacement.quantile(0.95)) < float(nonclean.old_aligned_displacement.quantile(0.95)),
        "high_flex_torsion_below_old_target": _mean(high_flex, "minimal_max_rotatable_torsion_change") < _mean(high_flex, "old_max_rotatable_torsion_change"),
        "severe_clash_not_increased": _mean(frame, "minimal_severe_clash_rate") <= _mean(frame, "input_severe_clash_rate") + 1.0e-12,
        "chirality_not_worse": _mean(frame, "minimal_chirality_preserved") >= _mean(frame, "input_chirality_preserved") - 1.0e-12,
        "clean_identity_at_least_90_percent": float((clean.minimal_aligned_displacement <= 1.0e-3).mean()) >= 0.90,
        "no_target_exceeds_trust_limit": bool(
            (frame.minimal_aligned_displacement <= trust_rms + 1.0e-6).all()
            and (frame.minimal_max_atom_displacement <= trust_atom + 1.0e-6).all()
        ),
        "no_four_angstrom_fallback": bool((frame.minimal_max_atom_displacement < 4.0).all()),
        "fallback_reasons_complete": bool(frame.minimal_stop_reason.notna().all()),
        "validity_gain_per_displacement_better_than_old": minimal_efficiency > old_efficiency,
    }
    return {
        "schema_version": "ecir-mvr-minimal-target-pilot-v1",
        "decision": "PASS" if all(criteria.values()) else "NO_GO_MINIMAL_TARGET",
        "test_used": False,
        "selection_split": "train",
        "pilot_records": len(frame),
        "category_counts": {str(k): int(v) for k, v in frame.pilot_category.value_counts().items()},
        "high_flex_records": int(frame.high_flex.sum()),
        "ring_records": int(frame.ring.sum()),
        "clean_records": int(frame.clean_input.sum()),
        "target_status_counts": {str(k): int(v) for k, v in frame.minimal_target_status.value_counts().items()},
        "old_target_status_counts": {str(k): int(v) for k, v in frame.old_target_status.value_counts().items()},
        "comparisons": comparisons,
        "displacement": {
            "minimal_mean": _mean(nonclean, "minimal_aligned_displacement"),
            "old_mean": _mean(nonclean, "old_aligned_displacement"),
            "minimal_p95": float(nonclean.minimal_aligned_displacement.quantile(0.95)),
            "old_p95": float(nonclean.old_aligned_displacement.quantile(0.95)),
            "minimal_max_atom": float(frame.minimal_max_atom_displacement.max()),
        },
        "torsion": {
            "high_flex_minimal_mean_max_change": _mean(high_flex, "minimal_max_rotatable_torsion_change"),
            "high_flex_old_mean_max_change": _mean(high_flex, "old_max_rotatable_torsion_change"),
        },
        "identity_preservation": float((clean.minimal_aligned_displacement <= 1.0e-3).mean()),
        "target_failure_rate": float((frame.minimal_target_status == "identity_fallback").mean()),
        "validity_gain_per_displacement": {
            "minimal": minimal_efficiency,
            "old": old_efficiency,
        },
        "improved_metrics": improved_metrics,
        "ten_percent_metrics": ten_percent,
        "criteria": criteria,
        "config": config,
    }


def run_pilot(args, frame, validity, builder):
    items = _pilot_items(frame, validity)
    output_rows = []
    target_dir = args.target_output_dir / "pilot"
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items):
        row = item["row"]
        record = item["record"]
        x_input = item["coordinates"]
        sample_id = item.get("sample_id_override", str(row.sample_id))
        old = build_real_error_target(record, coordinates=x_input)
        old_target = torch.as_tensor(old["x_target"], dtype=torch.float32)
        minimal = builder.build(x_input, record)
        minimal_target = torch.as_tensor(minimal["x_target"], dtype=torch.float32)
        metadata = minimal["target_metadata"]
        payload = {
            "sample_id": sample_id,
            "molecule_id": str(row.molecule_id),
            "x_input": x_input,
            "x_target": minimal_target,
            "old_target": old_target,
            "target_metadata": metadata,
            "old_target_metadata": {key: value for key, value in old.items() if key != "x_target"},
        }
        cache_path = target_dir / f"{index:04d}.pt"
        torch.save(payload, cache_path)
        output = {
            "sample_id": sample_id,
            "molecule_id": str(row.molecule_id),
            "pilot_category": item["pilot_category"],
            "source": str(row.generator_name),
            "severity": str(row.source_severity),
            "high_flex": item["high_flex"],
            "ring": item["ring"],
            "clean_input": item["clean_input"],
            "minimal_target_status": metadata["target_status"],
            "minimal_selected_step": metadata["selected_step"],
            "minimal_stop_reason": metadata["stop_reason"],
            "minimal_target_sha256": metadata["target_sha256"],
            "minimal_reference_fallback_used": metadata["reference_fallback_used"],
            "old_target_status": old["target_source"],
            "target_cache_path": str(cache_path.resolve()),
            **_target_metrics("input", x_input, x_input, record, validity),
            **_target_metrics("old", x_input, old_target, record, validity),
            **_target_metrics("minimal", x_input, minimal_target, record, validity),
        }
        output_rows.append(output)
    result = pd.DataFrame(output_rows)
    args.pilot_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.pilot_csv, index=False)
    summary = _pilot_summary(result, builder.config.__dict__)
    summary["pilot_manifest_sha256"] = hashlib.sha256(args.pilot_csv.read_bytes()).hexdigest()
    summary["identity_sha256"] = _sha(summary)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def run_full(args, frames, validity, builder):
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    if summary["decision"] != "PASS":
        raise RuntimeError("full target build is blocked until the pilot decision is PASS")
    for split, frame in frames.items():
        target_dir = args.target_output_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, row in enumerate(frame.sort_values(["molecule_id", "sample_id"]).itertuples(index=False)):
            record, x_input = _load_input(row)
            result = builder.build(x_input, record)
            cache_path = target_dir / f"{index:05d}.pt"
            torch.save({
                "sample_id": str(row.sample_id),
                "molecule_id": str(row.molecule_id),
                "x_input": x_input,
                "x_target": result["x_target"],
                "target_metadata": result["target_metadata"],
            }, cache_path)
            metadata = result["target_metadata"]
            rows.append({
                "split": split,
                "sample_id": str(row.sample_id),
                "molecule_id": str(row.molecule_id),
                "source": str(row.generator_name),
                "severity": str(row.source_severity),
                "target_cache_path": str(cache_path.resolve()),
                "target_status": metadata["target_status"],
                "validity_gain": metadata["validity_gain"],
                "initial_to_target_rmsd": metadata["initial_to_target_rmsd"],
                "max_atom_displacement": metadata["max_atom_displacement"],
                "torsion_change": metadata["torsion_change"],
                "max_rotatable_torsion_change": metadata["max_rotatable_torsion_change"],
                "selected_step": metadata["selected_step"],
                "stop_reason": metadata["stop_reason"],
                "target_sha256": metadata["target_sha256"],
            })
        manifest = pd.DataFrame(rows)
        manifest.to_parquet(args.target_output_dir / f"{split}.parquet", index=False)
        print(json.dumps({
            "split": split, "records": len(manifest),
            "molecules": int(manifest.molecule_id.nunique()),
            "status_counts": manifest.target_status.value_counts().to_dict(),
        }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--train_sources", type=Path, required=True)
    parser.add_argument("--val_sources", type=Path)
    parser.add_argument("--validity_stats", type=Path, required=True)
    parser.add_argument("--target_output_dir", type=Path, required=True)
    parser.add_argument("--pilot_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_steps", type=int, default=40)
    args = parser.parse_args()
    if "test" in {part.lower() for part in args.train_sources.resolve().parts}:
        raise ValueError("test data is forbidden")
    train = pd.read_parquet(args.train_sources)
    if set(train.split.unique()) != {"train"}:
        raise ValueError("pilot sources must be train-only")
    validity = ChemicalValidity(args.validity_stats)
    builder = MinimalValidityTargetBuilder(validity, {
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
    })
    if args.mode == "pilot":
        run_pilot(args, train, validity, builder)
    else:
        if args.val_sources is None:
            raise ValueError("--val_sources is required for full mode")
        val = pd.read_parquet(args.val_sources)
        if set(val.split.unique()) != {"val"}:
            raise ValueError("validation source manifest must contain val only")
        if set(train.molecule_id) & set(val.molecule_id):
            raise ValueError("train/val molecule leakage")
        run_full(args, {"train": train, "val": val}, validity, builder)


if __name__ == "__main__":
    main()
