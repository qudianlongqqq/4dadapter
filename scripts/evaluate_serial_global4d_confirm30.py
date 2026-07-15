#!/usr/bin/env python
"""Validation-only Confirm30 evaluation for Serial Global4D refinement."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.commons.kabsch_utils import kabsch_rmsd
from etflow.formal_large import canonical_sha256
from etflow.serial_global4d.cache import SerialGlobal4DResidualDataset
from etflow.serial_global4d.model import SerialGlobal4DResidualRefiner
from etflow.serial_global4d.safety import safe_serial_update, trust_region_clip

VARIANTS = (
    "etflow_raw",
    "cartesian",
    "serial_1step_ungated",
    "serial_1step_gate",
    "serial_1step_gate_trust",
    "serial_1step_full_safety",
    "serial_2step_full_safety",
    "stretch_only",
    "angular_only",
    "torsion_only",
    "full4d",
)


def _rmsd(coordinates: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.sqrt((coordinates - reference).square().sum(-1).mean()))


def _apply(
    model,
    batch,
    current,
    *,
    gate_override,
    joint_mode="full_4d",
    trust=False,
    full_safety=False,
):
    t = current.new_zeros(1)
    output = model(
        batch,
        current,
        t,
        gate_override=gate_override,
        joint_mode=joint_mode,
    )
    delta = output["delta"]
    diagnostics = {
        "gate": float(output["gate"].detach().mean()),
        "accepted": True,
        "clipped": False,
        "backtracking_count": 0,
        "reject_reason": None,
        "correction_norm": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
    }
    if full_safety:
        result = safe_serial_update(
            current,
            delta,
            batch.edge_index,
            output["atom_batch"],
            gate=output["gate"],
            gate_accept_threshold=0.0,
            max_atom_displacement=0.1,
            max_graph_rms_displacement=0.05,
            max_internal_velocity_norm=10.0,
            max_backtracks=4,
            backtrack_factor=0.5,
            min_bond_ratio=0.5,
            max_bond_ratio=1.5,
            min_nonbond_distance=0.5,
        )
        diagnostics.update(
            {
                "accepted": result.accepted,
                "clipped": result.atom_clipped
                or result.graph_rms_clipped
                or result.internal_norm_clipped,
                "backtracking_count": result.backtracking_count,
                "reject_reason": result.reject_reason,
                "correction_norm": float(
                    torch.linalg.vector_norm(result.accepted_delta, dim=-1).mean()
                ),
            }
        )
        return result.coordinates, diagnostics
    if trust:
        delta, clip = trust_region_clip(
            delta,
            output["atom_batch"],
            max_atom_displacement=0.1,
            max_graph_rms_displacement=0.05,
            max_internal_velocity_norm=10.0,
        )
        diagnostics["clipped"] = any(clip.values())
        diagnostics["correction_norm"] = float(
            torch.linalg.vector_norm(delta.detach(), dim=-1).mean()
        )
    return current + delta, diagnostics


def _coverage(
    by_molecule: dict[str, list[dict]],
    source_by_sample: dict[str, dict],
    variant: str,
    threshold: float,
) -> dict[str, float]:
    rows = []
    for records in by_molecule.values():
        generated = [record["coordinates"][variant] for record in records]
        references = torch.as_tensor(
            source_by_sample[records[0]["sample_id"]]["x_ref_candidates"],
            dtype=torch.float32,
        )
        distances = torch.stack(
            [
                torch.stack(
                    [kabsch_rmsd(candidate, reference) for candidate in generated]
                )
                for reference in references
            ]
        )
        best_reference = distances.min(dim=1).values
        best_generated = distances.min(dim=0).values
        rows.append(
            {
                "MAT-R": float(best_reference.mean()),
                "MAT-P": float(best_generated.mean()),
                "COV-R": float((best_reference < threshold).float().mean()),
                "COV-P": float((best_generated < threshold).float().mean()),
            }
        )
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage2_cache", required=True, type=Path)
    parser.add_argument("--source_cache", required=True, type=Path)
    parser.add_argument("--validation_manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(args.validation_manifest.read_text(encoding="utf-8"))
    expected_manifest = (
        "5a7da0b3fdbdf88aafe565c45728d65ff112151dd75162cb3b4b0022924162c2"
    )
    if canonical_sha256(manifest) != expected_manifest:
        raise ValueError("Confirm30 canonical identity mismatch")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("phase") != "b":
        raise ValueError("Confirm30 Serial evaluation requires a Phase B checkpoint")
    model = SerialGlobal4DResidualRefiner(**payload["config"]["model"]).to(args.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    dataset = SerialGlobal4DResidualDataset(
        args.stage2_cache, "val", require_completed=True
    )
    source_by_sample = {}
    manifest_by_sample = {str(row["sample_id"]): row for row in manifest["records"]}
    for path in sorted((args.source_cache / "val").glob("*.pt")):
        if len(source_by_sample) == len(manifest_by_sample):
            break
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        sample_id = str(record.get("sample_id", record.get("mol_id", "")))
        row = manifest_by_sample.get(sample_id)
        if row is None:
            continue
        if str(record.get("x_init_hash", "")) != str(row["x_init_hash"]):
            raise ValueError("Source validation x_init_hash mismatch")
        source_by_sample[sample_id] = record
    missing = sorted(set(manifest_by_sample).difference(source_by_sample))
    if missing:
        raise ValueError(f"Formal validation cache is missing Confirm30: {missing[:5]}")
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = dataset[index]
            sample_id = str(batch.sample_id)
            batch.batch = torch.zeros(
                batch.num_nodes, dtype=torch.long, device=batch.x_cart.device
            )
            batch = batch.to(args.device)
            coordinates = {
                "etflow_raw": batch.x_source_init.detach().cpu(),
                "cartesian": batch.x_cart.detach().cpu(),
            }
            diagnostics = {}
            variants = {
                "serial_1step_ungated": dict(gate_override=1.0),
                "serial_1step_gate": dict(gate_override=None),
                "serial_1step_gate_trust": dict(gate_override=None, trust=True),
                "serial_1step_full_safety": dict(gate_override=None, full_safety=True),
                "stretch_only": dict(gate_override=1.0, joint_mode="stretch_only"),
                "angular_only": dict(gate_override=1.0, joint_mode="angular_only"),
                "torsion_only": dict(gate_override=1.0, joint_mode="torsion_only"),
                "full4d": dict(gate_override=1.0, joint_mode="full_4d"),
            }
            for name, kwargs in variants.items():
                value, detail = _apply(model, batch, batch.x_cart, **kwargs)
                coordinates[name] = value.detach().cpu()
                diagnostics[name] = detail
            current = batch.x_cart
            step_details = []
            for _ in range(2):
                current, detail = _apply(
                    model,
                    batch,
                    current,
                    gate_override=None,
                    full_safety=True,
                )
                step_details.append(detail)
            coordinates["serial_2step_full_safety"] = current.detach().cpu()
            diagnostics["serial_2step_full_safety"] = {
                "steps": step_details,
                "gate": statistics.mean(row["gate"] for row in step_details),
                "accepted": all(row["accepted"] for row in step_details),
                "clipped": any(row["clipped"] for row in step_details),
                "backtracking_count": sum(
                    row["backtracking_count"] for row in step_details
                ),
                "reject_reason": next(
                    (
                        row["reject_reason"]
                        for row in step_details
                        if row["reject_reason"]
                    ),
                    None,
                ),
                "correction_norm": sum(row["correction_norm"] for row in step_details),
            }
            reference = batch.x_ref_aligned.detach().cpu()
            records.append(
                {
                    "sample_id": sample_id,
                    "mol_id": str(batch.mol_id),
                    "cohort": str(batch.flexibility_cohort),
                    "coordinates": coordinates,
                    "rmsd": {
                        name: _rmsd(value, reference)
                        for name, value in coordinates.items()
                    },
                    "diagnostics": diagnostics,
                }
            )
    by_molecule = defaultdict(list)
    for record in records:
        by_molecule[record["mol_id"]].append(record)
    summary = {}
    baseline_molecules = {
        mol_id: statistics.mean(row["rmsd"]["cartesian"] for row in group)
        for mol_id, group in by_molecule.items()
    }
    for variant in VARIANTS:
        values = [record["rmsd"][variant] for record in records]
        molecule_values = {
            mol_id: statistics.mean(row["rmsd"][variant] for row in group)
            for mol_id, group in by_molecule.items()
        }
        deltas = [
            molecule_values[mol_id] - baseline_molecules[mol_id]
            for mol_id in molecule_values
        ]
        row = {
            "rmsd_mean": statistics.mean(values),
            "rmsd_median": statistics.median(values),
            "molecule_improved_fraction": sum(value < -1.0e-6 for value in deltas)
            / len(deltas),
            "molecule_degraded_fraction": sum(value > 1.0e-6 for value in deltas)
            / len(deltas),
            "molecule_unchanged_fraction": sum(abs(value) <= 1.0e-6 for value in deltas)
            / len(deltas),
            "cohort_rmsd": {
                cohort: statistics.mean(
                    record["rmsd"][variant]
                    for record in records
                    if record["cohort"] == cohort
                )
                for cohort in ("low", "medium", "high")
            },
            **_coverage(by_molecule, source_by_sample, variant, args.threshold),
        }
        if variant.startswith("serial_"):
            details = [record["diagnostics"][variant] for record in records]
            row.update(
                {
                    "gate_mean": statistics.mean(value["gate"] for value in details),
                    "acceptance_fraction": sum(value["accepted"] for value in details)
                    / len(details),
                    "clipping_fraction": sum(value["clipped"] for value in details)
                    / len(details),
                    "backtracking_fraction": sum(
                        value["backtracking_count"] > 0 for value in details
                    )
                    / len(details),
                    "reject_fraction": sum(not value["accepted"] for value in details)
                    / len(details),
                    "correction_norm_mean": statistics.mean(
                        value["correction_norm"] for value in details
                    ),
                }
            )
        summary[variant] = row
    serial = summary["serial_1step_full_safety"]
    two_step = summary["serial_2step_full_safety"]
    checks = {
        "one_or_two_step_beats_cartesian": min(
            serial["rmsd_mean"], two_step["rmsd_mean"]
        )
        < summary["cartesian"]["rmsd_mean"],
        "high_flex_not_materially_worse": min(
            serial["cohort_rmsd"]["high"], two_step["cohort_rmsd"]["high"]
        )
        <= 1.01 * summary["cartesian"]["cohort_rmsd"]["high"],
        "improved_exceeds_degraded": max(
            serial["molecule_improved_fraction"],
            two_step["molecule_improved_fraction"],
        )
        > min(
            serial["molecule_degraded_fraction"],
            two_step["molecule_degraded_fraction"],
        ),
        "failure_rate_zero": serial["reject_fraction"] == 0
        and two_step["reject_fraction"] == 0,
        "corrections_not_all_rejected": serial["acceptance_fraction"] > 0
        and two_step["acceptance_fraction"] > 0,
    }
    output = {
        "status": "COMPLETED",
        "selection_split": "validation",
        "test_used": False,
        "manifest_canonical_sha256": expected_manifest,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": payload["step"],
        "record_count": len(records),
        "molecule_count": len(by_molecule),
        "coverage_threshold": args.threshold,
        "inference_seconds": time.perf_counter() - started,
        "summary": summary,
        "checks": checks,
        "pass": all(checks.values()),
        "records": [
            {key: value for key, value in record.items() if key != "coordinates"}
            for record in records
        ],
    }
    atomic_json_save(output, args.output)
    print(
        json.dumps(
            {key: output[key] for key in ("summary", "checks", "pass")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
