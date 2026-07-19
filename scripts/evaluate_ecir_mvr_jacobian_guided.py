#!/usr/bin/env python3
"""Compare frozen Jacobian-guided candidates on the V2 development cohort."""

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
from etflow.ecir.bac_evaluation import attach_canonical_constraints, infer_bac  # noqa: E402
from etflow.ecir.bac_jacobian import JacobianBACConfig  # noqa: E402
from etflow.ecir.bac_jacobian_guided import (  # noqa: E402
    jacobian_projection,
    posthoc_jacobian_correction,
    trust_region_hybrid,
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


EXPECTED_MANIFEST_IDENTITY = "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
EXPECTED_D1_SHA256 = "9348744817ef7eec6d9d682dd95a35f0be86f0565b6dd060e8d5fe54e609e426"
METHODS = ("D1", "A025", "A050", "A100", "B", "C")
METRIC_COLUMNS = {
    "bond": "bond_outlier_rate",
    "angle": "angle_outlier_rate",
    "clash": "clash_penetration",
    "weighted_bac": "total_thresholded_validity_score",
    "ring": "ring_bond_outlier_rate",
    "chirality": "chirality_error",
    "rmsd": "aligned_RMSD",
    "displacement": "molecule_rms_displacement",
}


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


def _paired_candidate_minus_d1(
    records: pd.DataFrame,
    candidate: str,
    *,
    draws: int = 500,
    seed: int,
) -> dict[str, Any]:
    source = records[records.method == "upstream"].set_index("sample_id")
    d1 = records[records.method == "D1"].set_index("sample_id")
    guided = records[records.method == candidate].set_index("sample_id")
    common = source.index.intersection(d1.index).intersection(guided.index)
    angle_active = source.loc[common].angle_outlier_rate > 0
    subsets = {"all": common, "angle_active": common[angle_active.to_numpy()]}
    result = {}
    for subset_name, selected in subsets.items():
        frame = pd.DataFrame(
            {
                name: guided.loc[selected, column].to_numpy() - d1.loc[selected, column].to_numpy()
                for name, column in METRIC_COLUMNS.items()
            },
            index=selected,
        )
        frame["molecule_id"] = source.loc[selected].molecule_id.to_numpy()
        molecule = frame.groupby("molecule_id").mean(numeric_only=True)
        metrics = {}
        for offset, name in enumerate(METRIC_COLUMNS):
            values = molecule[name].to_numpy(dtype=np.float64)
            rng = np.random.default_rng(seed + offset)
            sampled = np.asarray(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
            )
            metrics[name] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
            }
        result[subset_name] = {
            "records": int(len(selected)),
            "molecules": int(molecule.shape[0]),
            "metrics": metrics,
        }
    return result


def _method_metrics(summary: pd.DataFrame, method: str) -> dict[str, float]:
    rows = summary[summary.group == "all"].set_index("method")
    source = rows.loc["upstream"]
    candidate = rows.loc[method]
    return {
        "bond_delta": float(candidate.bond_outlier_rate - source.bond_outlier_rate),
        "angle_delta": float(candidate.angle_outlier_rate - source.angle_outlier_rate),
        "clash_delta": float(candidate.clash_penetration - source.clash_penetration),
        "weighted_bac_delta": float(
            candidate.total_thresholded_validity_score - source.total_thresholded_validity_score
        ),
        "ring_delta": float(candidate.ring_bond_outlier_rate - source.ring_bond_outlier_rate),
        "chirality_delta": float(candidate.chirality_error - source.chirality_error),
        "rmsd_delta": float(candidate.aligned_RMSD - source.aligned_RMSD),
        "acceptance_fraction": float(candidate.accepted_fraction),
        "rejected_fraction": float(candidate.rejected_fraction),
        "mean_displacement": float(candidate.molecule_rms_displacement),
        "max_displacement": float(candidate.max_displacement),
    }


def _support(
    metrics: dict[str, float],
    d1: dict[str, float],
    paired: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "active_angle_ci95_high_lt_zero": paired["angle_active"]["metrics"]["angle"]["ci95_high"]
        < 0.0,
        "bond_degradation_vs_d1_le_0.005": metrics["bond_delta"] - d1["bond_delta"] <= 0.005,
        "weighted_bac_degradation_vs_d1_le_0.005": metrics["weighted_bac_delta"]
        - d1["weighted_bac_delta"]
        <= 0.005,
        "acceptance_drop_vs_d1_le_0.05": d1["acceptance_fraction"] - metrics["acceptance_fraction"]
        <= 0.05,
        "movement_ratio_vs_d1_le_1.1": metrics["mean_displacement"]
        <= 1.1 * d1["mean_displacement"],
        "ring_non_regressed_vs_d1": metrics["ring_delta"] <= d1["ring_delta"],
        "chirality_non_regressed_vs_d1": metrics["chirality_delta"] <= d1["chirality_delta"],
    }
    return {"supported": all(checks.values()), "checks": checks}


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
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/jacobian_guided"),
    )
    parser.add_argument("--d1-device", default="cuda:0")
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "formal_root",
        "source_cache_root",
        "manifest_dir",
        "d1_checkpoint",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    experiment = json.loads(
        (args.output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (args.manifest_dir / "recovery_development_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["identity_sha256"] != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError("V4 development manifest changed")
    if experiment["development_manifest_identity_sha256"] != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError("V4 experiment manifest development identity changed")
    for payload in (manifest, experiment):
        if (
            payload["test_records_read"] != 0
            or payload["test_assets_opened"]
            or payload["frozen_holdout_records_opened"] != 0
        ):
            raise RuntimeError("V4 isolation contract violated")
    if _sha(args.d1_checkpoint) != EXPECTED_D1_SHA256:
        raise RuntimeError("fixed D1 checkpoint SHA mismatch")
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
        raise RuntimeError("V4 development record count changed")

    checkpoint = torch.load(args.d1_checkpoint, map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    d1_device = torch.device(args.d1_device)
    model = MCVRBACModel(**train_config["model"]).to(d1_device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("fixed D1 strict-load failed")
    attach_canonical_constraints(
        items,
        validity,
        source_identity_sha256=source_metadata["formal_source_identity_sha256"],
    )
    if torch.cuda.is_available() and d1_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d1_device)
    d1_started = time.perf_counter()
    d1_coordinates, d1_metadata = infer_bac(
        model,
        items,
        validity,
        device=d1_device,
        steps=int(train_config["inference"].get("teacher_steps", 4)),
        step_size=float(train_config["inference"].get("step_size", 0.25)),
        batch_size=int(train_config["inference"].get("batch_size", 64)),
        safety_config=BACSafetyConfig(**dict(train_config["inference"].get("safety", {}))),
    )
    d1_runtime = time.perf_counter() - d1_started
    d1_peak_memory = (
        int(torch.cuda.max_memory_allocated(d1_device)) if d1_device.type == "cuda" else 0
    )

    jacobian_config = JacobianBACConfig()
    safety = BACSafetyConfig(
        max_atom_displacement=jacobian_config.max_atom_displacement,
        max_molecule_rms_displacement=jacobian_config.max_molecule_rms_displacement,
    )
    coordinates: dict[str, list[torch.Tensor]] = {
        "upstream": [item["input"] for item in items],
        "D1": d1_coordinates,
        **{method: [] for method in METHODS if method != "D1"},
    }
    metadata: dict[str, list[dict[str, Any]]] = {
        "D1": d1_metadata,
        **{method: [] for method in METHODS if method != "D1"},
    }
    diagnostics = []
    runtimes = Counter()
    statuses: dict[str, Counter[str]] = {method: Counter() for method in METHODS if method != "D1"}
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    guided_started = time.perf_counter()
    for index, (item, d1_coordinate, d1_meta) in enumerate(
        zip(items, d1_coordinates, d1_metadata, strict=True)
    ):
        static = canonical_constraint_fields(
            validity,
            item["record"],
            source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        )
        common = {
            "bonds": static["active_bond_constraint_index"],
            "bond_ranges": static["bond_allowed_range"],
            "angles": static["active_angle_constraint_index"].t(),
            "angle_ranges": static["angle_allowed_range"],
            "config": jacobian_config,
            "safety_config": safety,
        }
        results = {}
        for alpha, method in ((0.25, "A025"), (0.5, "A050"), (1.0, "A100")):
            results[method] = posthoc_jacobian_correction(
                item["input"],
                d1_coordinate,
                item["record"],
                validity,
                alpha=alpha,
                d1_accepted=bool(d1_meta["accepted"]),
                atomic_numbers=item["record"].get("atomic_numbers"),
                **common,
            )
        results["B"] = jacobian_projection(
            item["input"],
            d1_coordinate,
            item["record"],
            validity,
            d1_accepted=bool(d1_meta["accepted"]),
            **common,
        )
        results["C"] = trust_region_hybrid(
            item["input"],
            d1_coordinate,
            item["record"],
            validity,
            **common,
        )
        for method, result in results.items():
            coordinates[method].append(result["coordinates"])
            reasons = result["diagnostics"].get("hard_safety_reasons", [])
            metadata[method].append(
                {
                    "accepted": result["accepted"],
                    "selected_step": result["diagnostics"].get(
                        "selected_scale", d1_meta.get("selected_step", 0)
                    ),
                    "reject_reasons": ";".join(reasons),
                }
            )
            runtimes[method] += float(result["runtime_seconds"])
            statuses[method][result["status"]] += 1
            diagnostics.append(
                {
                    "record_index": index,
                    "sample_id": str(item["row"].sample_id),
                    "molecule_id": str(item["row"].molecule_id),
                    "method": method,
                    "accepted": result["accepted"],
                    "rolled_back": result["rolled_back"],
                    "status": result["status"],
                    "runtime_seconds": result["runtime_seconds"],
                    "diagnostics": result["diagnostics"],
                }
            )
        peak_rss = max(peak_rss, process.memory_info().rss)
    guided_wall_runtime = time.perf_counter() - guided_started

    records = method_rows(items, coordinates, validity, method_metadata=metadata)
    summary, molecules = summarize_groups(records, items, coordinates)
    metrics = {method: _method_metrics(summary, method) for method in METHODS}
    frozen_metrics = json.loads(
        (args.d1_checkpoint.parent / "run_metadata.json").read_text(encoding="utf-8")
    )["metrics"]
    frozen_names = {
        "bond_delta": "bond_delta",
        "angle_delta": "angle_delta",
        "clash_delta": "clash_delta",
        "acceptance_fraction": "accepted_fraction",
    }
    for name, frozen_name in frozen_names.items():
        if not math.isclose(
            metrics["D1"][name],
            float(frozen_metrics[frozen_name]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"fixed D1 reproduction mismatch: {name}")

    paired = {
        method: _paired_candidate_minus_d1(
            records,
            method,
            draws=args.bootstrap_draws,
            seed=44000 + index * 100,
        )
        for index, method in enumerate(METHODS[1:])
    }
    support = {
        method: _support(metrics[method], metrics["D1"], paired[method]) for method in METHODS[1:]
    }
    supported = [method for method in METHODS[1:] if support[method]["supported"]]
    source_bootstrap = {
        method: paired_bootstrap(
            molecules,
            candidate=method,
            draws=args.bootstrap_draws,
            seed=45000 + index,
        )
        for index, method in enumerate(METHODS)
    }
    fallback_fraction = {
        method: float(
            np.mean([row["rolled_back"] for row in diagnostics if row["method"] == method])
        )
        for method in METHODS[1:]
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "per_record.csv", index=False)
    molecules.to_csv(args.output_dir / "per_molecule.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "candidate_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    report = {
        "schema_version": "mcvr-v4-jacobian-guided-evaluation-v1",
        "decision": (
            "JACOBIAN_GUIDANCE_SUPPORTED" if supported else "JACOBIAN_GUIDANCE_NOT_SUPPORTED"
        ),
        "supported_candidates": supported,
        "records": len(items),
        "molecules": int(records.molecule_id.nunique()),
        "metrics": metrics,
        "paired_candidate_minus_d1": paired,
        "source_bootstrap": source_bootstrap,
        "support_rule": support,
        "guided_diagnostics": {
            "status_counts": {method: dict(statuses[method]) for method in METHODS[1:]},
            "fallback_fraction": fallback_fraction,
            "candidate_runtime_seconds": dict(runtimes),
            "wall_runtime_seconds": guided_wall_runtime,
            "peak_cpu_rss_bytes": peak_rss,
        },
        "d1_reproduction": {
            "checkpoint_sha256": EXPECTED_D1_SHA256,
            "runtime_seconds": d1_runtime,
            "peak_gpu_memory_bytes": d1_peak_memory,
            "metrics": metrics["D1"],
        },
        "development_manifest_identity_sha256": EXPECTED_MANIFEST_IDENTITY,
        "formal_source_identity_sha256": source_metadata["formal_source_identity_sha256"],
        "configuration_selected_from_results": False,
        "additional_candidates_run": False,
        "training_runs": 0,
        "target_rematerialization": False,
        "model_state_dict_changed": False,
        "start_10k": bool(supported),
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
