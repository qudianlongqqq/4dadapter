#!/usr/bin/env python
"""Evaluate cached V8 coordinates without executing model inference."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import torch

from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.run_a_evaluation import rmsd_matrix
from etflow.ecir.v8_validation_cache import (
    ISOLATION,
    ValidationStatus,
    atomic_json,
    file_sha256,
    iter_prediction_records,
    tensor_sha256,
)


def _memberships(item) -> dict[str, bool]:
    active = torch.as_tensor(item.active_mode_mask).reshape(-1)
    movement = torch.linalg.vector_norm(item.x_target - item.x_input, dim=-1).mean()
    return {
        "natural": True,
        "active_angle": bool(active.numel() > 1 and active[1] > 0),
        "active_clash": bool(active.numel() > 3 and active[3] > 0),
        "ring_risk": bool(active.numel() > 2 and active[2] > 0),
        "high_flexibility": int(torch.as_tensor(item.num_rotatable_bonds).max()) >= 6,
        "low_error_minimal_movement": float(movement) <= 0.0025,
    }


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in rows[0]}


def _bootstrap(values: list[float], *, draws: int = 10_000, seed: int = 43) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"draws": draws, "mean": None, "ci95_low": None, "ci95_high": None}
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = generator.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    return {
        "draws": draws,
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--training-step", type=int, default=0)
    parser.add_argument("--mode", choices=("FAST", "FULL"), required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    prediction_manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_cache_manifest.read_text(encoding="utf-8"))
    if prediction_manifest.get("status") != "COMPLETED" or source_manifest.get("status") != "COMPLETED":
        raise RuntimeError("evaluation requires completed source and prediction caches")
    for payload in (prediction_manifest, source_manifest):
        for key, value in ISOLATION.items():
            if payload.get(key) != value:
                raise RuntimeError(f"cache isolation changed: {key}")
    output = args.output.resolve()
    status = ValidationStatus.start(
        args.status_file.resolve()
        if args.status_file
        else output.with_name(output.stem + ".status.json"),
        phase="FAST_VALIDATING" if args.mode == "FAST" else "FULL_EVALUATING",
        training_step=args.training_step,
        validation_mode=args.mode,
    )
    started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        predictions = list(iter_prediction_records(args.prediction_manifest))
        wanted = {int(row["record_index"]) for row in predictions}
        sources = {
            int(row["record_index"]): row
            for row in iter_prediction_records(args.source_cache_manifest)
            if int(row["record_index"]) in wanted
        }
        cache_load_seconds = time.perf_counter() - load_started
        if len(sources) != len(predictions):
            raise RuntimeError("source/prediction cache join is incomplete")
        validity = ChemicalValidity(args.validity_statistics.resolve())
        reasons: Counter[str] = Counter()
        metric_rows: list[dict[str, float]] = []
        per_record_rows: list[dict] = []
        cohort_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
        molecule_coordinates: dict[str, list[torch.Tensor]] = defaultdict(list)
        molecule_references: dict[str, torch.Tensor] = {}
        metrics_started = time.perf_counter()
        for offset, prediction in enumerate(predictions, start=1):
            source_row = sources[int(prediction["record_index"])]
            if prediction["sample_id"] != source_row["sample_id"]:
                raise RuntimeError("paired cache sample identity changed")
            item = source_row["item"]
            record = source_row["record"]
            source = item.x_input.detach().cpu()
            if tensor_sha256(source) != prediction["source_coordinate_sha256"]:
                raise RuntimeError("paired cache source coordinate identity changed")
            safe = torch.as_tensor(prediction["safe_coordinates"], dtype=source.dtype)
            before = validity.evaluate(source, record, baseline_coordinates=source)
            after = validity.evaluate(safe, record, baseline_coordinates=source)
            displacement = torch.linalg.vector_norm(safe - source, dim=-1)
            references = torch.as_tensor(
                record.get("x_ref_candidates", record.get("x_ref_aligned")),
                dtype=torch.float32,
            )
            if references.ndim == 2:
                references = references.unsqueeze(0)
            nearest = float(rmsd_matrix([safe], references).min())
            row = {
                "accepted": float(bool(prediction["accepted"])),
                "bond_delta": float(after["bond_outlier_rate"] - before["bond_outlier_rate"]),
                "angle_delta": float(after["angle_outlier_rate"] - before["angle_outlier_rate"]),
                "active_angle_delta": float(after["angle_outlier_rate"] - before["angle_outlier_rate"]),
                "clash_delta": float(after["clash_penetration"] - before["clash_penetration"]),
                "ring_delta": float(after["ring_bond_outlier_rate"] - before["ring_bond_outlier_rate"]),
                "weighted_bac_delta": float(
                    after["total_thresholded_validity_score"]
                    - before["total_thresholded_validity_score"]
                ),
                "mean_displacement": float(displacement.mean()),
                "max_atom_displacement": float(displacement.max()),
                "target_loss": float(torch.nn.functional.smooth_l1_loss(safe, item.x_target)),
                "chirality_preserved": float(after["chirality_preserved"]),
                "solver_failure_count": float(
                    prediction["solver_diagnostics"]["failure_count"]
                ),
                "confidence_mean": float(
                    torch.as_tensor(
                        prediction.get("method_diagnostics", {}).get("confidence", [0.0])
                    ).mean()
                ),
                "solver_bond_contribution": float(
                    prediction["solver_diagnostics"]["bond_contribution"]
                ),
                "solver_angle_contribution": float(
                    prediction["solver_diagnostics"]["angle_contribution"]
                ),
                "rmsd": nearest,
            }
            metric_rows.append(row)
            per_record_rows.append(
                {
                    "record_index": int(prediction["record_index"]),
                    "sample_id": str(prediction["sample_id"]),
                    "molecule_id": str(prediction["molecule_id"]),
                    **row,
                }
            )
            memberships = _memberships(item)
            for name, selected in memberships.items():
                if selected:
                    cohort_rows[name].append(row)
            for reason in prediction["backtracking_decision"].get("reasons", ()):
                reasons[str(reason)] += 1
            molecule = str(prediction["molecule_id"])
            molecule_coordinates[molecule].append(safe)
            molecule_references.setdefault(molecule, references)
            if offset % 250 == 0 or offset == len(predictions):
                elapsed = time.perf_counter() - started
                status.update(
                    current_validation_record=offset,
                    evaluation_chunks_completed=(offset + 249) // 250,
                    records_per_second=offset / max(elapsed, 1.0e-9),
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=(len(predictions) - offset) * elapsed / offset,
                )
        metric_seconds = time.perf_counter() - metrics_started
        set_metrics = None
        bootstrap = None
        set_started = time.perf_counter()
        if args.mode == "FULL":
            molecule_rows = []
            for molecule, coordinates in molecule_coordinates.items():
                matrix = rmsd_matrix(coordinates, molecule_references[molecule])
                molecule_rows.append(
                    {
                        "MAT_P": float(matrix.min(1).values.mean()),
                        "MAT_R": float(matrix.min(0).values.mean()),
                        "COV_P": float((matrix.min(1).values < 1.25).float().mean()),
                        "COV_R": float((matrix.min(0).values < 1.25).float().mean()),
                    }
                )
            set_metrics = _mean(molecule_rows)
            bootstrap = {
                "weighted_bac_delta": _bootstrap(
                    [row["weighted_bac_delta"] for row in metric_rows],
                    draws=args.bootstrap_draws,
                )
            }
        set_bootstrap_seconds = time.perf_counter() - set_started
        result = {
            "schema_version": "mcvr-v8-cached-validation-report-v1",
            "status": "COMPLETED",
            "mode": args.mode,
            "records": len(metric_rows),
            "metrics": _mean(metric_rows),
            "cohort_metrics": {name: _mean(rows) for name, rows in cohort_rows.items()},
            "cohort_counts": {name: len(rows) for name, rows in cohort_rows.items()},
            "set_metrics": set_metrics,
            "paired_bootstrap": bootstrap,
            "per_record_metrics": per_record_rows if args.mode == "FULL" else None,
            "rejection_reasons": dict(reasons),
            "prediction_manifest_sha256": file_sha256(args.prediction_manifest),
            "source_cache_manifest_sha256": file_sha256(args.source_cache_manifest),
            "evaluator_semantics": "frozen_v7_bac_safety_weighted_thresholded_validity",
            "timing": {
                "cache_load_seconds": cache_load_seconds,
                "metric_seconds": metric_seconds,
                "set_and_bootstrap_seconds": set_bootstrap_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            **ISOLATION,
        }
        atomic_json(output, result)
        elapsed = time.perf_counter() - started
        status.update(
            status="COMPLETED",
            phase="COMPLETED",
            current_validation_record=len(metric_rows),
            evaluation_chunks_completed=(len(metric_rows) + 249) // 250,
            records_per_second=len(metric_rows) / max(elapsed, 1.0e-9),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=0.0,
        )
    except BaseException as error:
        status.fail(error, elapsed_seconds=time.perf_counter() - started)
        raise


if __name__ == "__main__":
    main()
