#!/usr/bin/env python3
"""Evaluate frozen J0 against the repaired Cartesian D1 development baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_constraints import canonical_constraint_fields  # noqa: E402
from etflow.ecir.bac_evaluation import evaluate_bac_candidate  # noqa: E402
from etflow.ecir.bac_jacobian import (  # noqa: E402
    JacobianBACConfig,
    solve_bac_jacobian,
)
from etflow.ecir.bac_safety import BACSafetyConfig  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v2_bac import MCVRBACModel  # noqa: E402
from etflow.ecir.run_a_evaluation import (  # noqa: E402
    build_items,
    method_rows,
    paired_bootstrap,
    summarize_groups,
)


EXPECTED_MANIFEST_IDENTITY = (
    "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
)
EXPECTED_D1_SHA256 = (
    "9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _active_subset_metrics(
    records: pd.DataFrame, candidate_method: str, draws: int = 500
) -> dict[str, Any]:
    source = records[records.method == "upstream"].set_index("sample_id")
    candidate = records[records.method == candidate_method].set_index("sample_id")
    common = source.index.intersection(candidate.index)
    source = source.loc[common]
    candidate = candidate.loc[common]
    definitions = {
        "all": pd.Series(True, index=common),
        "bond_active": source.bond_outlier_rate > 0,
        "angle_active": source.angle_outlier_rate > 0,
        "clash_active": (source.clash_penetration > 0) | (source.severe_clash_rate > 0),
    }
    metric_columns = {
        "bond_delta": "bond_outlier_rate",
        "angle_delta": "angle_outlier_rate",
        "clash_delta": "clash_penetration",
        "weighted_bac_delta": "total_thresholded_validity_score",
        "ring_delta": "ring_bond_outlier_rate",
        "chirality_delta": "chirality_error",
        "rmsd_delta": "aligned_RMSD",
    }
    rng = np.random.default_rng(43017)
    result = {}
    for name, mask in definitions.items():
        selected = common[mask.to_numpy()]
        if not len(selected):
            result[name] = {"records": 0, "molecules": 0, "status": "NO_DATA_SUPPORT"}
            continue
        left = source.loc[selected]
        right = candidate.loc[selected]
        delta = pd.DataFrame(
            {
                output: right[column].to_numpy() - left[column].to_numpy()
                for output, column in metric_columns.items()
            },
            index=selected,
        )
        delta["molecule_id"] = left.molecule_id.to_numpy()
        molecule = delta.groupby("molecule_id").mean(numeric_only=True)
        metrics = {}
        for column in metric_columns:
            values = molecule[column].to_numpy(dtype=np.float64)
            sampled = np.asarray(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
            )
            metrics[column] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
            }
        result[name] = {
            "records": len(selected),
            "molecules": int(molecule.shape[0]),
            "metrics": metrics,
            "acceptance_fraction": float(right.accepted.mean()),
            "mean_displacement": float(right.molecule_rms_displacement.mean()),
        }
    return result


def _paired_j0_minus_d1(
    d1_records: pd.DataFrame,
    j0_records: pd.DataFrame,
    *,
    draws: int = 500,
) -> dict[str, Any]:
    d1 = d1_records[d1_records.method == "v2_bac_accepted"].set_index("sample_id")
    j0 = j0_records[j0_records.method == "j0_jacobian"].set_index("sample_id")
    source = j0_records[j0_records.method == "upstream"].set_index("sample_id")
    common = source.index.intersection(d1.index).intersection(j0.index)
    angle_active = source.loc[common].angle_outlier_rate > 0
    subsets = {"all": common, "angle_active": common[angle_active.to_numpy()]}
    columns = {
        "bond": "bond_outlier_rate",
        "angle": "angle_outlier_rate",
        "clash": "clash_penetration",
        "weighted_bac": "total_thresholded_validity_score",
        "ring": "ring_bond_outlier_rate",
        "chirality": "chirality_error",
        "rmsd": "aligned_RMSD",
        "displacement": "molecule_rms_displacement",
    }
    rng = np.random.default_rng(43018)
    result = {}
    for subset, selected in subsets.items():
        frame = pd.DataFrame(
            {
                name: j0.loc[selected, column].to_numpy()
                - d1.loc[selected, column].to_numpy()
                for name, column in columns.items()
            },
            index=selected,
        )
        frame["molecule_id"] = source.loc[selected].molecule_id.to_numpy()
        molecule = frame.groupby("molecule_id").mean(numeric_only=True)
        metrics = {}
        for column in columns:
            values = molecule[column].to_numpy(dtype=np.float64)
            sampled = np.asarray(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
            )
            metrics[column] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
            }
        result[subset] = {
            "records": len(selected),
            "molecules": int(molecule.shape[0]),
            "metrics": metrics,
        }
    return result


def _summary_metrics(summary: pd.DataFrame) -> dict[str, float]:
    rows = summary[summary.group == "all"].set_index("method")
    source = rows.loc["upstream"]
    candidate = rows.loc["j0_jacobian"]
    return {
        "bond_delta": float(candidate.bond_outlier_rate - source.bond_outlier_rate),
        "angle_delta": float(candidate.angle_outlier_rate - source.angle_outlier_rate),
        "clash_delta": float(candidate.clash_penetration - source.clash_penetration),
        "ring_delta": float(candidate.ring_bond_outlier_rate - source.ring_bond_outlier_rate),
        "chirality_delta": float(candidate.chirality_error - source.chirality_error),
        "rmsd_delta": float(candidate.aligned_RMSD - source.aligned_RMSD),
        "mat_p_delta": float(candidate.MAT_P - source.MAT_P),
        "mat_r_delta": float(candidate.MAT_R - source.MAT_R),
        "cov_p_delta": float(candidate.COV_P - source.COV_P),
        "cov_r_delta": float(candidate.COV_R - source.COV_R),
        "acceptance_fraction": float(candidate.accepted_fraction),
        "rollback_fraction": float(candidate.rejected_fraction),
        "mean_displacement": float(candidate.molecule_rms_displacement),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/manifests"),
    )
    parser.add_argument(
        "--d1-checkpoint",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v2_bac_recovery/runs/"
            "d1_pilot_1000step_seed43018/checkpoint_final.ckpt"
        ),
    )
    parser.add_argument(
        "--phase1-comparison",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/candidate_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/jacobian_bac"),
    )
    parser.add_argument("--d1-device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "formal_root",
        "source_cache_root",
        "manifest_dir",
        "d1_checkpoint",
        "phase1_comparison",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    manifest = json.loads(
        (args.manifest_dir / "recovery_development_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["identity_sha256"] != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError("Jacobian evaluation development manifest changed")
    if (
        manifest["test_records_read"] != 0
        or manifest["test_assets_opened"]
        or manifest["frozen_holdout_records_opened"] != 0
    ):
        raise RuntimeError("Jacobian evaluation manifest violates isolation")
    if _sha(args.d1_checkpoint) != EXPECTED_D1_SHA256:
        raise RuntimeError("fixed D1 comparator checkpoint SHA mismatch")
    if args.d1_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("fixed D1 comparison requested unavailable CUDA")

    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    source_metadata = json.loads(
        (args.formal_root / "real_sources" / "metadata.json").read_text(encoding="utf-8")
    )
    items = build_items(
        args.manifest_dir / "development_sources.parquet",
        args.manifest_dir / "development_targets.parquet",
        validity,
        source_cache_root=args.source_cache_root,
        target_cache_root=args.formal_root / "minimal_targets",
    )
    if len(items) != 1024:
        raise RuntimeError("Jacobian development record count changed")

    checkpoint = torch.load(args.d1_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    d1_device = torch.device(args.d1_device)
    d1_model = MCVRBACModel(**config["model"]).to(d1_device)
    incompatible = d1_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("fixed D1 comparator strict-load failed")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(d1_device)
    d1_started = time.perf_counter()
    d1_evaluation = evaluate_bac_candidate(
        d1_model,
        items,
        validity,
        device=d1_device,
        inference=config["inference"],
        source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        bootstrap_draws=500,
    )
    d1_runtime = time.perf_counter() - d1_started
    d1_peak_memory = (
        int(torch.cuda.max_memory_allocated(d1_device)) if torch.cuda.is_available() else 0
    )
    frozen_metrics = json.loads(
        (
            args.d1_checkpoint.parent / "run_metadata.json"
        ).read_text(encoding="utf-8")
    )["metrics"]
    for name in ("bond_delta", "angle_delta", "clash_delta", "accepted_fraction"):
        if not math.isclose(
            float(d1_evaluation["metrics"][name]),
            float(frozen_metrics[name]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"fixed D1 reproduction mismatch: {name}")

    j0_config = JacobianBACConfig()
    safety = BACSafetyConfig(
        max_atom_displacement=j0_config.max_atom_displacement,
        max_molecule_rms_displacement=j0_config.max_molecule_rms_displacement,
    )
    coordinates = []
    metadata = []
    diagnostics = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    j0_started = time.perf_counter()
    for index, item in enumerate(items):
        static = canonical_constraint_fields(
            validity,
            item["record"],
            source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        )
        result = solve_bac_jacobian(
            item["input"],
            item["record"],
            validity,
            bonds=static["active_bond_constraint_index"],
            bond_ranges=static["bond_allowed_range"],
            angles=static["active_angle_constraint_index"].t(),
            angle_ranges=static["angle_allowed_range"],
            atomic_numbers=item["record"].get("atomic_numbers"),
            config=j0_config,
            safety_config=safety,
        )
        coordinates.append(result["coordinates"])
        reasons = (
            []
            if result["accepted"]
            else result["final_safety"].get("reasons", [result["solver_status"]])
        )
        metadata.append(
            {
                "accepted": result["accepted"],
                "selected_step": result["iteration_count"],
                "reject_reasons": ";".join(reasons),
            }
        )
        iteration = result["iterations"][-1] if result["iterations"] else {}
        diagnostics.append(
            {
                "record_index": index,
                "sample_id": str(item["row"].sample_id),
                "molecule_id": str(item["row"].molecule_id),
                "accepted": result["accepted"],
                "solver_status": result["solver_status"],
                "iteration_count": result["iteration_count"],
                "initial_objective": result["initial_objective"],
                "final_objective": result["final_objective"],
                "objective_reduction": result["objective_reduction"],
                "runtime_seconds": result["runtime_seconds"],
                "constraint_counts": result["initial_constraint_counts"],
                "effective_rank": iteration.get("effective_rank", 0),
                "singular_value_max": iteration.get("singular_value_max", 0.0),
                "singular_value_min_retained": iteration.get(
                    "singular_value_min_retained", 0.0
                ),
                "condition_number": iteration.get("condition_number", 0.0),
                "truncated_direction_count": iteration.get(
                    "truncated_direction_count", 0
                ),
                "solver_backend": iteration.get("solver_backend", "none"),
                "raw_update_norm": iteration.get("raw_update_norm", 0.0),
                "trust_scaled_norm": iteration.get("trust_scaled_norm", 0.0),
                "accepted_step_scale": iteration.get("accepted_step_scale", 0.0),
                "predicted_reduction": iteration.get("predicted_reduction", 0.0),
                "actual_reduction": iteration.get("actual_reduction", 0.0),
                "reduction_ratio": iteration.get("reduction_ratio", 0.0),
                "attempted_step_scales": iteration.get("attempted_step_scales", []),
                "degenerate_bond_count": result["initial_constraint_diagnostics"][
                    "degenerate_bond_count"
                ],
                "degenerate_angle_count": result["initial_constraint_diagnostics"][
                    "degenerate_angle_count"
                ],
                "degenerate_clash_count": result["initial_constraint_diagnostics"][
                    "degenerate_clash_count"
                ],
                "near_linear_angle_count": result["initial_constraint_diagnostics"][
                    "near_linear_angle_count"
                ],
            }
        )
        peak_rss = max(peak_rss, process.memory_info().rss)
    j0_runtime = time.perf_counter() - j0_started
    j0_records = method_rows(
        items,
        {"upstream": [item["input"] for item in items], "j0_jacobian": coordinates},
        validity,
        method_metadata={"j0_jacobian": metadata},
    )
    summary, molecules = summarize_groups(
        j0_records,
        items,
        {"upstream": [item["input"] for item in items], "j0_jacobian": coordinates},
    )
    bootstrap = paired_bootstrap(molecules, candidate="j0_jacobian", draws=500)
    metrics = _summary_metrics(summary)
    active_subsets = _active_subset_metrics(j0_records, "j0_jacobian")
    paired = _paired_j0_minus_d1(d1_evaluation["records"], j0_records)
    failure_statuses = {
        "NONFINITE_SYSTEM",
        "NONFINITE_UPDATE",
        "FACTORIZATION_FAILED",
        "SVD_DIAGNOSTIC_FAILED",
        "EFFECTIVE_RANK_ZERO",
    }
    status_counts = Counter(row["solver_status"] for row in diagnostics)
    solver_failures = sum(status_counts[name] for name in failure_statuses)
    d1_displacement = float(d1_evaluation["metrics"]["mean_displacement"])
    angle_difference = paired["angle_active"]["metrics"]["angle"]
    movement_ratio = metrics["mean_displacement"] / max(d1_displacement, 1.0e-30)
    supported = bool(
        angle_difference["ci95_high"] < 0.0
        and metrics["bond_delta"] <= d1_evaluation["metrics"]["bond_delta"]
        and metrics["ring_delta"] <= 0.0
        and metrics["chirality_delta"] <= 0.0
        and solver_failures / len(items) <= 0.01
        and movement_ratio <= 1.25
    )
    decision = "JACOBIAN_SUPPORTED" if supported else "JACOBIAN_NOT_SUPPORTED"
    if active_subsets["angle_active"]["records"] < 30:
        decision = "JACOBIAN_INCONCLUSIVE"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    j0_records.to_csv(args.output_dir / "per_record.csv", index=False)
    molecules.to_csv(args.output_dir / "per_molecule.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "solver_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    phase1 = pd.read_csv(args.phase1_comparison)
    j0_row = {
        "candidate": "J0",
        "steps": 0,
        "records": 1024,
        "molecules": 512,
        **metrics,
        "weighted_bac_delta": active_subsets["all"]["metrics"][
            "weighted_bac_delta"
        ]["mean"],
        "angle_active_records": active_subsets["angle_active"]["records"],
        "angle_active_molecules": active_subsets["angle_active"]["molecules"],
        "angle_active_delta": active_subsets["angle_active"]["metrics"][
            "angle_delta"
        ]["mean"],
        "angle_active_ci95_low": active_subsets["angle_active"]["metrics"][
            "angle_delta"
        ]["ci95_low"],
        "angle_active_ci95_high": active_subsets["angle_active"]["metrics"][
            "angle_delta"
        ]["ci95_high"],
        "clash_active_records": active_subsets["clash_active"]["records"],
        "runtime_seconds": j0_runtime,
        "checkpoint_sha256": "not_applicable_non_learning_solver",
        "failure_rate": solver_failures / len(items),
    }
    pd.concat((phase1, pd.DataFrame([j0_row])), ignore_index=True).to_csv(
        args.output_dir / "candidate_comparison.csv", index=False
    )
    report = {
        "schema_version": "mcvr-v3-bac-jacobian-evaluation-v1",
        "decision": decision,
        "records": len(items),
        "molecules": 512,
        "j0_metrics": metrics,
        "j0_active_subsets": active_subsets,
        "j0_bootstrap": bootstrap,
        "j0_solver": {
            "status_counts": dict(status_counts),
            "solver_failure_count": solver_failures,
            "solver_failure_rate": solver_failures / len(items),
            "runtime_seconds": j0_runtime,
            "runtime_per_graph_seconds": j0_runtime / len(items),
            "peak_cpu_rss_bytes": peak_rss,
            "condition_number": {
                "mean": float(np.mean([row["condition_number"] for row in diagnostics])),
                "p95": float(np.quantile([row["condition_number"] for row in diagnostics], 0.95)),
                "max": float(max(row["condition_number"] for row in diagnostics)),
            },
        },
        "d1_reproduction": {
            "checkpoint_sha256": EXPECTED_D1_SHA256,
            "metrics": d1_evaluation["metrics"],
            "runtime_seconds": d1_runtime,
            "runtime_per_graph_seconds": d1_runtime / len(items),
            "peak_gpu_memory_bytes": d1_peak_memory,
        },
        "paired_j0_minus_d1": paired,
        "movement_ratio_j0_over_d1": movement_ratio,
        "configuration_selected_from_results": False,
        "j1_run": False,
        "start_learned_jacobian": decision == "JACOBIAN_SUPPORTED",
        "start_10k": False,
        "start_formal_large": False,
        "formal_test_records_read": 0,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    _write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
