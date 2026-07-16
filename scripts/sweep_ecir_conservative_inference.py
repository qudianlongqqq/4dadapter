#!/usr/bin/env python
"""Two-stage conservative inference sweep for the frozen ECIR 5k checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
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
from torch_geometric.data import Batch

from etflow.ecir.acceptance import evaluate_candidate
from etflow.ecir.audit import displacement_metrics, internal_metrics
from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.dataset import ECIRMixedDataset
from etflow.ecir.model import ECIRFlowSystem
from etflow.ecir.stage_b_decision import compare_train_range_to_legacy


CHEMICAL = (
    "bond_outlier_rate", "bond_outlier_magnitude", "angle_outlier_rate",
    "angle_outlier_magnitude", "severe_clash_rate", "clash_penetration",
    "ring_bond_outlier_rate", "ring_planarity_outlier_rate",
    "stereocenter_degenerate_rate", "torsion_prior_outlier_score",
    "total_thresholded_validity_score",
)
TARGET_RELATIVE = (
    "bond_target_mae", "angle_target_mae_rad", "torsion_reference_error",
    "ring_bond_target_mae", "clash_score", "chirality_error",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _schedule_specs():
    yield {"time_schedule_mode": "legacy_full", "fixed_t": None}
    yield {"time_schedule_mode": "train_range", "fixed_t": None}
    for value in (0.0, 0.05, 0.10, 0.20, 0.25):
        yield {"time_schedule_mode": "fixed", "fixed_t": value}


def _coarse_configs():
    for schedule, steps, scale, trust, gate in itertools.product(
        list(_schedule_specs()), (1, 2, 4), (0.05, 0.10, 0.20, 0.50, 1.00),
        (0.50, 1.00), (0.00, 0.10, 0.30),
    ):
        config = {
            **schedule, "teacher_steps": steps, "update_scale": scale,
            "trust_radius_scale": trust, "gate_threshold": gate,
            "phase": "coarse", "acceptance_mode": "final_step",
        }
        config["config_id"] = _config_id(config)
        yield config


def _historical_config():
    config = {
        "time_schedule_mode": "explicit", "fixed_t": None,
        "explicit_time_schedule": [0.0, 0.25, 0.5, 0.75],
        "teacher_steps": 4, "update_scale": 1.0, "trust_radius_scale": 1.0,
        "gate_threshold": 0.0, "phase": "historical_stage_a",
        "acceptance_mode": "none",
    }
    config["config_id"] = _config_id(config)
    return config


def _fine_configs(pareto: pd.DataFrame, existing: set[str]) -> list[dict[str, Any]]:
    scales = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
    trusts = (0.25, 0.50, 0.75, 1.00)
    gates = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50)
    steps = (1, 2, 3, 4)
    results = []
    candidates = pareto.sort_values(
        ["delta_total_thresholded_validity_score", "delta_aligned_RMSD", "aligned_rms_displacement"]
    ).head(20)
    for offset, (_, row) in enumerate(candidates.iterrows()):
        base = {
            "time_schedule_mode": row.time_schedule_mode,
            "fixed_t": None if pd.isna(row.fixed_t) else float(row.fixed_t),
            "teacher_steps": int(row.teacher_steps), "update_scale": float(row.update_scale),
            "trust_radius_scale": float(row.trust_radius_scale),
            "gate_threshold": float(row.gate_threshold), "phase": "fine",
            "acceptance_mode": "final_step",
        }
        variants = []
        for value in scales: variants.append({**base, "update_scale": value})
        for value in trusts: variants.append({**base, "trust_radius_scale": value})
        for value in gates: variants.append({**base, "gate_threshold": value})
        for value in steps: variants.append({**base, "teacher_steps": value})
        for variant in variants[offset % len(variants):] + variants[:offset % len(variants)]:
            variant["config_id"] = _config_id(variant)
            if variant["config_id"] not in existing:
                existing.add(variant["config_id"]); results.append(variant); break
        if len(results) >= 20:
            break
    return results


def _pareto(frame: pd.DataFrame) -> pd.DataFrame:
    objectives = [
        "delta_total_thresholded_validity_score", "delta_aligned_RMSD",
        "aligned_rms_displacement", "validity_worsened_fraction",
    ]
    values = frame[objectives].to_numpy(float)
    keep = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        if not keep[index]: continue
        dominates = np.all(values <= values[index] + 1e-12, axis=1) & np.any(values < values[index] - 1e-12, axis=1)
        if dominates.any(): keep[index] = False
    return frame.loc[keep].copy()


def _rmsd_matrix(generated: list[torch.Tensor], references: torch.Tensor) -> torch.Tensor:
    mobile = torch.stack([torch.as_tensor(value, dtype=torch.float64) for value in generated])
    target = torch.as_tensor(references, dtype=torch.float64)
    if target.ndim == 2: target = target.unsqueeze(0)
    x = mobile - mobile.mean(1, keepdim=True)
    y = target - target.mean(1, keepdim=True)
    covariance = torch.einsum("gni,rnj->grij", x, y)
    u, singular, vh = torch.linalg.svd(covariance)
    determinant = torch.linalg.det(vh.transpose(-2, -1) @ u.transpose(-2, -1))
    trace = singular.sum(-1) - torch.where(determinant < 0, 2.0 * singular[..., -1], 0.0)
    squared = (
        x.square().sum((1, 2))[:, None]
        + y.square().sum((1, 2))[None, :]
        - 2.0 * trace
    ) / x.size(1)
    return squared.clamp_min(0.0).sqrt().to(torch.float32)


def _nearest_rmsd_fast(coordinates: torch.Tensor, references: torch.Tensor) -> float:
    return float(_rmsd_matrix([coordinates], references).min())


def _cov_mat(records, coordinate_key: str, threshold: float = 1.25) -> dict[str, float]:
    per_molecule = []
    grouped = defaultdict(list)
    for row in records: grouped[row["molecule_id"]].append(row)
    for rows in grouped.values():
        generated = [row[coordinate_key] for row in rows]
        references = rows[0]["references"]
        matrix = _rmsd_matrix(generated, references)
        diversity = (
            _rmsd_matrix(generated, torch.stack(generated))[torch.triu(torch.ones(len(generated), len(generated), dtype=torch.bool), diagonal=1)].mean()
            if len(generated) > 1 else matrix.new_zeros(())
        )
        per_molecule.append({
            "COV_P": float((matrix.min(1).values < threshold).float().mean()),
            "COV_R": float((matrix.min(0).values < threshold).float().mean()),
            "MAT_P": float(matrix.min(1).values.mean()),
            "MAT_R": float(matrix.min(0).values.mean()), "diversity": float(diversity),
        })
    return {key: float(np.mean([row[key] for row in per_molecule])) for key in per_molecule[0]}


def _summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    candidate_keys = {key for row in records for key in row}
    numeric_keys = sorted(
        key for key in candidate_keys
        if all(
            key not in row
            or row[key] is None
            or isinstance(row[key], (int, float, bool, np.number))
            for row in records
        )
    )
    molecule_rows = []
    grouped = defaultdict(list)
    for row in records: grouped[row["molecule_id"]].append(row)
    for molecule, rows in grouped.items():
        molecule_rows.append({"molecule_id": molecule, **{
            key: float(np.mean([float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]))
            for key in numeric_keys if any(key in row and math.isfinite(float(row[key])) for row in rows)
        }})
    frame = pd.DataFrame(molecule_rows)
    summary = {key: float(frame[key].mean()) for key in frame.select_dtypes(include=[np.number]).columns}
    summary.update({f"input_{key}": value for key, value in _cov_mat(records, "input_coordinates").items()})
    summary.update(_cov_mat(records, "accepted_coordinates"))
    for key in ("COV_P", "COV_R", "MAT_P", "MAT_R", "diversity"):
        summary[f"delta_{key}"] = summary[key] - summary[f"input_{key}"]
    summary["molecules"] = int(len(grouped)); summary["records"] = int(len(records))
    summary["accepted_fraction"] = float(frame.accepted.mean())
    summary["rejected_fraction"] = 1.0 - summary["accepted_fraction"]
    summary["unchanged_fraction"] = float((frame.aligned_rms_displacement <= 1.0e-6).mean())
    summary["validity_improved_fraction"] = float((frame.delta_total_thresholded_validity_score < -1.0e-6).mean())
    summary["validity_worsened_fraction"] = float((frame.delta_total_thresholded_validity_score > 1.0e-6).mean())
    summary["RMSD_improved_fraction"] = float((frame.delta_aligned_RMSD < -1.0e-6).mean())
    summary["RMSD_worsened_fraction"] = float((frame.delta_aligned_RMSD > 1.0e-6).mean())
    summary["raw_RMSD_worsened_fraction"] = float((frame.raw_delta_aligned_RMSD > 1.0e-6).mean())
    return summary


def _bootstrap(records, metric: str, draws: int = 1000):
    grouped = defaultdict(list)
    for row in records: grouped[row["molecule_id"]].append(float(row[f"delta_{metric}"]))
    delta = np.asarray([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(42)
    means = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(draws)])
    return {"mean": float(delta.mean()), "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--atlas_path", type=Path, required=True)
    parser.add_argument("--target_cache_dir", type=Path, required=True)
    parser.add_argument("--views_manifest", type=Path, required=True)
    parser.add_argument("--validity_stats", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph_batch_size", type=int, default=50)
    parser.add_argument("--max_coarse_configs", type=int)
    args = parser.parse_args()
    if _sha(args.checkpoint) != "232e47865d01a71543cf2cd16ede577764fd3d94ac843d78dcdcf8c9789fa98d":
        raise ValueError("unexpected ECIR checkpoint identity")
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = dict(payload["config"].get("model") or {})
    model = ECIRFlowSystem(**model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
    validity = ChemicalValidity(args.validity_stats)
    dataset = ECIRMixedDataset(
        args.cache_dir, "val", atlas_path=args.atlas_path,
        target_cache_dir=args.target_cache_dir, real_error_ratio=1.0,
        synthetic_error_ratio=0.0, clean_identity_ratio=0.0,
    )
    data_by_sample = {}
    for index, path in enumerate(dataset.files):
        record = torch.load(path, map_location="cpu", weights_only=False)
        data_by_sample[str(record["sample_id"])] = dataset.get(index)
    manifest = pd.read_parquet(args.views_manifest)
    items = []
    record_cache = {}
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        record = record_cache.setdefault(sample_id, torch.load(Path(row.source_path), map_location="cpu", weights_only=False))
        if isinstance(row.generated_coordinate_path, str) and row.generated_coordinate_path:
            coordinates = torch.load(Path(row.generated_coordinate_path), map_location="cpu", weights_only=False)["coordinates"]
        else:
            coordinates = torch.as_tensor(record[row.coordinate_key], dtype=torch.float32)
        target = data_by_sample[sample_id].x_target
        references = torch.as_tensor(record.get("x_ref_candidates", record["x_ref_aligned"]), dtype=torch.float32)
        if references.ndim == 2: references = references.unsqueeze(0)
        input_validity = validity.evaluate(coordinates, record, baseline_coordinates=coordinates)
        target_values = internal_metrics(coordinates, target, record)
        metadata = row._asdict()
        metadata["has_ring"] = bool(torch.as_tensor(record.get("bond_is_in_ring", [])).any())
        items.append({
            "meta": metadata, "record": record, "data": data_by_sample[sample_id],
            "input": coordinates, "target": target, "references": references,
            "input_validity": input_validity,
            "input_target": {
                "bond_target_mae": target_values["bond_violation"],
                "angle_target_mae_rad": target_values["angle_violation"],
                "torsion_reference_error": target_values["torsion_circular_error"],
                "ring_bond_target_mae": target_values["ring_invalidity"],
                "clash_score": target_values["clash_score"], "chirality_error": target_values["chirality_error"],
            },
            "input_rmsd": _nearest_rmsd_fast(coordinates, references),
        })
    chunks = []
    for start in range(0, len(items), args.graph_batch_size):
        selected = items[start:start + args.graph_batch_size]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        chunks.append((selected, batch, torch.cat([item["input"] for item in selected]).to(device)))

    def evaluate(config):
        started = time.perf_counter(); records = []
        for selected, batch, coordinates in chunks:
            refined, diagnostics = model.refine(
                batch, coordinates=coordinates, steps=int(config["teacher_steps"]),
                update_scale=float(config["update_scale"]), trust_radius_scale=float(config["trust_radius_scale"]),
                gate_threshold=float(config["gate_threshold"]), time_schedule_mode=config["time_schedule_mode"],
                fixed_t=config.get("fixed_t"), explicit_time_schedule=config.get("explicit_time_schedule"),
                strict_training_range=True, return_trajectory=True,
            )
            ptr = batch.ptr.detach().cpu().tolist(); refined = refined.detach().cpu()
            last = diagnostics[-1]
            uncertainty = last["graph_uncertainty"].detach().cpu().tolist()
            gate = last["graph_gate"].detach().cpu().tolist()
            for local, item in enumerate(selected):
                left, right = ptr[local], ptr[local + 1]
                raw = refined[left:right]
                raw_unchanged = bool(torch.equal(raw, item["input"]))
                raw_validity = (
                    item["input_validity"] if raw_unchanged
                    else validity.evaluate(raw, item["record"], baseline_coordinates=item["input"])
                )
                if config["acceptance_mode"] == "none":
                    accepted, accepted_coordinates, reasons, gain = True, raw, [], item["input_validity"]["total_thresholded_validity_score"] - raw_validity["total_thresholded_validity_score"]
                else:
                    decision = evaluate_candidate(
                        item["input"], raw, item["record"], validity, step=int(config["teacher_steps"]),
                        uncertainty=float(uncertainty[local]), input_validity_override=item["input_validity"],
                        candidate_validity_override=raw_validity,
                    )
                    accepted, reasons, gain = decision.accepted, decision.reject_reasons, decision.validity_gain
                    accepted_coordinates = raw if accepted else item["input"]
                accepted_validity = raw_validity if accepted else item["input_validity"]
                if not accepted:
                    accepted_target = item["input_target"]
                    accepted_rmsd = item["input_rmsd"]
                    disp = {"aligned_rms_displacement": 0.0, "mean_atom_displacement": 0.0, "max_atom_displacement": 0.0}
                else:
                    accepted_target_raw = internal_metrics(accepted_coordinates, item["target"], item["record"])
                    accepted_target = {
                        "bond_target_mae": accepted_target_raw["bond_violation"],
                        "angle_target_mae_rad": accepted_target_raw["angle_violation"],
                        "torsion_reference_error": accepted_target_raw["torsion_circular_error"],
                        "ring_bond_target_mae": accepted_target_raw["ring_invalidity"],
                        "clash_score": accepted_target_raw["clash_score"], "chirality_error": accepted_target_raw["chirality_error"],
                    }
                    accepted_rmsd = _nearest_rmsd_fast(accepted_coordinates, item["references"])
                    disp = displacement_metrics(item["input"], accepted_coordinates)
                raw_rmsd = item["input_rmsd"] if raw_unchanged else _nearest_rmsd_fast(raw, item["references"])
                result = {
                    **item["meta"], "molecule_id": item["meta"]["molecule_id"],
                    "accepted": float(accepted), "reject_reasons": ";".join(reasons),
                    "gate": float(gate[local]), "uncertainty": float(uncertainty[local]),
                    "validity_gain": float(gain), **disp,
                    "input_coordinates": item["input"], "raw_coordinates": raw,
                    "accepted_coordinates": accepted_coordinates, "references": item["references"],
                    "input_aligned_RMSD": item["input_rmsd"], "aligned_RMSD": accepted_rmsd,
                    "delta_aligned_RMSD": accepted_rmsd - item["input_rmsd"],
                    "raw_delta_aligned_RMSD": raw_rmsd - item["input_rmsd"],
                }
                for name in CHEMICAL:
                    result[f"input_{name}"] = item["input_validity"][name]
                    result[name] = accepted_validity[name]
                    result[f"delta_{name}"] = accepted_validity[name] - item["input_validity"][name]
                for name in TARGET_RELATIVE:
                    result[f"input_{name}"] = item["input_target"][name]
                    result[name] = accepted_target[name]
                    result[f"delta_{name}"] = accepted_target[name] - item["input_target"][name]
                records.append(result)
        mixed = [row for row in records if row["view_mixed"]]
        summary = {**_summarize(mixed), **config, "elapsed_seconds": time.perf_counter() - started}
        group_rows = []
        groups = {
            "mixed_all": mixed,
            "etflow_normal": [row for row in records if row["view_etflow_normal"]],
            "cartesian_all": [row for row in records if row["view_cartesian_severity"]],
        }
        for flex, predicate in {
            "rotatable_le_2": lambda value: value <= 2,
            "rotatable_3_5": lambda value: 3 <= value <= 5,
            "rotatable_ge_6": lambda value: value >= 6,
        }.items():
            groups[f"mixed_{flex}"] = [row for row in mixed if predicate(int(row["rotatable_bond_count"]))]
        groups["mixed_ring"] = [row for row in mixed if row["has_ring"]]
        groups["mixed_non_ring"] = [row for row in mixed if not row["has_ring"]]
        for name, selected_rows in groups.items():
            if selected_rows:
                group_rows.append({"config_id": config["config_id"], "group": name, **_summarize(selected_rows)})
        severity_rows = []
        for severity in ("normal", "mild", "medium", "severe", "extrapolated_extreme"):
            selected_rows = [row for row in records if row["source_severity"] == severity]
            severity_rows.append({"config_id": config["config_id"], "source_severity": severity, **_summarize(selected_rows)})
        return summary, group_rows, severity_rows, records

    configs = list(_coarse_configs())
    if args.max_coarse_configs is not None: configs = configs[: args.max_coarse_configs]
    configs.append(_historical_config())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summary, source_rows, severity_rows = [], [], []
    for index, config in enumerate(configs):
        summary, source, severity, records = evaluate(config)
        all_summary.append(summary); source_rows.extend(source); severity_rows.extend(severity)
        if (index + 1) % 25 == 0:
            pd.DataFrame(all_summary).to_csv(args.output_dir / "all_results.partial.csv", index=False)
            pd.DataFrame(source_rows).to_csv(args.output_dir / "source_summary.partial.csv", index=False)
            pd.DataFrame(severity_rows).to_csv(args.output_dir / "severity_summary.partial.csv", index=False)
            print(f"coarse {index + 1}/{len(configs)}", flush=True)
    frame = pd.DataFrame(all_summary)
    coarse_frame = frame[frame.phase == "coarse"].copy()
    pareto = _pareto(coarse_frame)
    if args.max_coarse_configs is None:
        fine = _fine_configs(pareto, set(frame.config_id))
        for index, config in enumerate(fine):
            summary, source, severity, records = evaluate(config)
            all_summary.append(summary); source_rows.extend(source); severity_rows.extend(severity)
            print(f"fine {index + 1}/{len(fine)}")
        frame = pd.DataFrame(all_summary); pareto = _pareto(frame[frame.phase != "historical_stage_a"])
    eligible = frame[
        (frame.delta_aligned_RMSD <= 0.015)
        & (frame.delta_MAT_P <= 0.015) & (frame.delta_MAT_R <= 0.015)
        & (frame.delta_COV_P >= -0.005) & (frame.delta_COV_R >= -0.005)
    ].copy()
    pool = eligible if not eligible.empty else pareto
    pool["selection_score"] = (
        pool.delta_total_thresholded_validity_score
        + 2.0 * pool.delta_aligned_RMSD.clip(lower=0)
        + pool.aligned_rms_displacement
        + pool.validity_worsened_fraction
    )
    best = pool.sort_values("selection_score").iloc[0]
    best_config = next(config for config in configs + (fine if args.max_coarse_configs is None else []) if config["config_id"] == best.config_id)
    _, _, _, best_records = evaluate(best_config)
    bootstrap_result = {metric: _bootstrap(best_records, metric) for metric in (
        "bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate",
        "severe_clash_rate", "total_thresholded_validity_score", "aligned_RMSD",
    )}
    source_frame = pd.DataFrame(source_rows); severity_frame = pd.DataFrame(severity_rows)
    best_source = source_frame[source_frame.config_id == best.config_id].set_index("group")
    best_severity = severity_frame[severity_frame.config_id == best.config_id].set_index("source_severity")
    true_ci = [name for name in ("bond_outlier_rate", "angle_outlier_rate", "ring_bond_outlier_rate", "severe_clash_rate") if bootstrap_result[name]["ci95_high"] < 0]
    train_range_nonworse, schedule_comparison = compare_train_range_to_legacy(frame)
    checks = {
        "two_true_validity_ci_improvements": len(true_ci) >= 2,
        "etflow_normal_noninferior": float(best_source.loc["etflow_normal", "delta_aligned_RMSD"]) <= 0.015,
        "mild_or_medium_effective": min(float(best_severity.loc["mild", "delta_total_thresholded_validity_score"]), float(best_severity.loc["medium", "delta_total_thresholded_validity_score"])) < 0,
        "not_extreme_only": min(float(best_severity.loc["normal", "delta_total_thresholded_validity_score"]), float(best_severity.loc["mild", "delta_total_thresholded_validity_score"]), float(best_severity.loc["medium", "delta_total_thresholded_validity_score"])) < 0,
        "accuracy_noninferior": all(float(best[name]) <= 0.015 for name in ("delta_aligned_RMSD", "delta_MAT_P", "delta_MAT_R")) and float(best.delta_COV_P) >= -0.005 and float(best.delta_COV_R) >= -0.005,
        "acceptance_reduces_rmsd_worsening": float(best.RMSD_worsened_fraction) < float(best.raw_RMSD_worsened_fraction),
        "high_flex_noninferior": float(best_source.loc["mixed_rotatable_ge_6", "delta_aligned_RMSD"]) <= 0.015,
        "train_range_nonworse_than_legacy_full": train_range_nonworse,
    }
    rescued = all(checks.values())
    any_signal = bool(true_ci) or float(best.delta_total_thresholded_validity_score) < 0
    status = "EXISTING_CKPT_RESCUED" if rescued else ("EXISTING_CKPT_DIAGNOSTIC_ONLY" if any_signal else "EXISTING_CKPT_NOT_RESCUED")
    frame.to_csv(args.output_dir / "all_results.csv", index=False)
    pareto.to_csv(args.output_dir / "pareto_front.csv", index=False)
    source_frame.to_csv(args.output_dir / "source_summary.csv", index=False)
    severity_frame.to_csv(args.output_dir / "severity_summary.csv", index=False)
    decision = {
        "stage": "STAGE_B", "decision": status, "best_config": best.to_dict(),
        "passed_checks": [name for name, value in checks.items() if value],
        "failed_checks": [name for name, value in checks.items() if not value],
        "bootstrap": bootstrap_result, "true_validity_metrics_with_ci_improvement": true_ci,
        "schedule_comparison": schedule_comparison,
        "checkpoint_sha256": _sha(args.checkpoint), "validity_stats_sha256": _sha(args.validity_stats),
        "views_manifest_sha256": _sha(args.views_manifest), "test_used": False,
        "training_started": False, "next_stage": "STAGE_C",
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
