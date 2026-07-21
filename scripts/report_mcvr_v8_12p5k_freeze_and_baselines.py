#!/usr/bin/env python
"""Freeze V8 12.5K and build the identity-matched formal-large comparison."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import numpy as np
import psutil
import torch

from etflow.ecir.run_a_evaluation import rmsd_matrix
from etflow.ecir.v8_validation_cache import ISOLATION, atomic_json, iter_prediction_records
from scripts.evaluate_ecir_mvr_v8_prediction_cache import _memberships


REPORT_DIR = ROOT / "reports/ecir_mvr/v8_full_v1/formal_large_12p5k"
V8_RUN = ROOT / "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k/full_seed43"
CACHE = ROOT / "diagnostics/ecir_mvr/validation_cache/formal_large_seed43"
REPORT_PATHS = {
    "D1": CACHE / "d1/evaluation.json",
    "V5-B": CACHE / "v5_b/evaluation.json",
    "V7": CACHE / "v7/evaluation.json",
    "V8 Full 5K": CACHE / "v8_step005000_full/evaluation.json",
    "V8 Full 12.5K": V8_RUN / "validation_cache/step012500/full/evaluation.json",
}
EXPECTED_CACHE_IDENTITIES = {
    "Source": "b0bd4bf83eff7a81782bc801d8fcc742375c4edb69f3d3829f1b0ba3b4c77f51",
    "D1": "5c1c1497953a946df5e020f555c6dc6d32d177e677215ddfe7a66d61183d5553",
    "V5-B": "8184dad3962dc4e0d76ef921ba9bd47e76a21b1d2331a503ff713b2223d22151",
    "V7": "ddf963dbb81edb4763a932056d9fe18a17faf93a491de8feb490d5d88af0ee2b",
}
METRICS = (
    "accepted",
    "weighted_bac_delta",
    "bond_delta",
    "angle_delta",
    "active_angle_delta",
    "clash_delta",
    "ring_delta",
    "chirality_preserved",
    "mean_displacement",
    "max_atom_displacement",
    "solver_failure_count",
    "rmsd",
    "target_loss",
)
SET_METRICS = ("MAT_P", "MAT_R", "COV_P", "COV_R")
COHORTS = (
    "natural",
    "active_angle",
    "active_clash",
    "ring_risk",
    "high_flexibility",
    "low_error_minimal_movement",
)
HIGHER_IS_BETTER = {"accepted", "chirality_preserved", "COV_P", "COV_R"}
TOLERANCE = {"accepted": 0.0, "chirality_preserved": 0.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def assert_no_v8_training_process() -> None:
    offenders = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except psutil.Error:
            continue
        if "train_ecir_mvr_v8.py" in command and str(V8_RUN) in command:
            offenders.append((process.pid, command))
    if offenders:
        raise RuntimeError(f"residual V8 Full processes exist: {offenders}")


def freeze_v8() -> dict[str, Any]:
    assert_no_v8_training_process()
    status = read_json(V8_RUN / "status.json")
    orchestration = read_json(V8_RUN / "graceful_stop_orchestration.json")
    assets = read_json(V8_RUN / "asset_hashes.json")
    if status.get("status") != "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED":
        raise RuntimeError("V8 12.5K result is not completed")
    if int(status.get("actual_completed_step", -1)) != 12500:
        raise RuntimeError("V8 completed step changed")
    artifacts = [
        V8_RUN / "checkpoints/step012500.ckpt",
        V8_RUN / "checkpoints/last.ckpt",
        V8_RUN / "checkpoints/step010000.ckpt",
        V8_RUN / "status.json",
        V8_RUN / "validation_cache/step012500/full/evaluation.json",
        V8_RUN / "validation_cache/step012500/full/paired_baseline_comparison.json",
        V8_RUN / "validation_protocol.jsonl",
        V8_RUN / "resume_to_12p5k_stdout.log",
        V8_RUN / "resume_to_12p5k_stderr.log",
        V8_RUN / "config.resolved.json",
        V8_RUN / "train.jsonl",
        V8_RUN / "graceful_stop_orchestration.json",
        V8_RUN / "control/normal_interruption_evidence.json",
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(f"V8 freeze artifacts are missing: {missing}")
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in artifacts}
    step_sha = hashes[str((V8_RUN / "checkpoints/step012500.ckpt").relative_to(ROOT)).replace("\\", "/")]
    last_sha = hashes[str((V8_RUN / "checkpoints/last.ckpt").relative_to(ROOT)).replace("\\", "/")]
    if step_sha != last_sha or step_sha != orchestration["step012500_checkpoint_sha256"]:
        raise RuntimeError("V8 frozen checkpoint identity changed")
    source_manifest = read_json(CACHE / "source/prediction_manifest.json")
    identities = {
        "parent_step5000_checkpoint_sha256": sha256(
            ROOT / "diagnostics/ecir_mvr/v8_full_v1/formal_large_5k/full_seed43/checkpoints/last.ckpt"
        ),
        "step10000_checkpoint_sha256": orchestration["step10000_checkpoint_sha256"],
        "step12500_checkpoint_sha256": step_sha,
        "last_checkpoint_sha256": last_sha,
        "resolved_config_file_sha256": sha256(V8_RUN / "config.resolved.json"),
        "resolved_config_identity_sha256": assets["resolved_config_sha256"],
        "dataset_identity": {
            "train_sources_sha256": assets["train_sources_sha256"],
            "train_targets_sha256": assets["train_targets_sha256"],
            "validation_sources_sha256": assets["val_sources_sha256"],
            "validation_targets_sha256": assets["val_targets_sha256"],
        },
        "train_scale_file_sha256": assets["residual_scales_file_sha256"],
        "train_scale_identity_sha256": assets["residual_scales_identity_sha256"],
        "stratified_manifest_file_sha256": assets["stratified_manifest_file_sha256"],
        "stratified_manifest_identity_sha256": assets[
            "stratified_manifest_identity_sha256"
        ],
        "validation_manifest_sha256": sha256(
            ROOT / "reports/ecir_mvr/validation_protocol/formal_large_fast1000_manifest.json"
        ),
        "validation_identity_sha256": read_json(
            ROOT / "reports/ecir_mvr/validation_protocol/formal_large_fast1000_manifest.json"
        )["validation_identity_sha256"],
        "baseline_cache_identities": EXPECTED_CACHE_IDENTITIES,
        "evaluator_semantics_sha256": source_manifest["identity"][
            "evaluator_semantics_sha256"
        ],
        "safety_semantics_sha256": source_manifest["identity"]["safety_semantics_sha256"],
    }
    payload = {
        "schema_version": "mcvr-v8-full-v1-formal-large-12p5k-freeze-v1",
        "status": "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_FROZEN",
        "method_branch": assets["git_branch"],
        "asset_recorded_head_at_resume": assets["git_head"],
        "training_launch_commit": "e28e5a9",
        "graceful_stop_implementation_commit": "bc39716e13472a830daf1ceaaa5c79401cd989b5",
        "stop_orchestration_commit": "054413c6c003feb4d77c27f6a910f527cea1689c",
        "resume_implementation_commit": "f5f5f4f0ad88ee3a64f89f9a1829ee1decfc4b49",
        "freeze_branch": git("branch", "--show-current"),
        "freeze_head_before_report_commit": git("rev-parse", "HEAD"),
        "run_directory": str(V8_RUN.relative_to(ROOT)).replace("\\", "/"),
        "identities": identities,
        "accounting": {
            "planned_original_total_steps": 200000,
            "actual_completed_step": 12500,
            "effective_batch": 64,
            "total_record_exposure": 800000,
            "equivalent_old_batch8_steps": 100000,
            "scheduler_provenance": "step12500 checkpoint from the original 200K schedule",
            "is_independent_12p5k_scheduler_experiment": False,
        },
        "artifact_sha256": hashes,
        "freeze_contract": {
            "original_results_are_immutable": True,
            "must_not_overwrite_or_modify": True,
            "large_diagnostics_and_checkpoints_not_committed_to_git": True,
        },
        "isolation": {key: status[key] for key in ISOLATION},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT_DIR / "V8_FULL_12P5K_FROZEN.json", payload)
    sums = "".join(f"{value}  {path}\n" for path, value in sorted(hashes.items()))
    (REPORT_DIR / "V8_FULL_12P5K_SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    markdown = f"""# MCVR V8 Full 12.5K frozen result

Status: `MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_FROZEN`

- Branch: `{assets['git_branch']}`
- Resume asset HEAD: `{assets['git_head']}`
- Step 12.5K SHA256: `{step_sha}`
- Effective batch: `64`
- Total record exposure: `800000`
- Equivalent old batch-8 steps: `100000`
- Formal-test reads: `0`; frozen-holdout reads: `0`

This checkpoint is the step 12500 point of the original 200K schedule. It is not an
independently designed 12.5K scheduler experiment. The run directory, checkpoints,
validation caches, logs, resolved configuration, and orchestration result are frozen and
must not be overwritten or modified.
"""
    (REPORT_DIR / "V8_FULL_12P5K_FROZEN.md").write_text(markdown, encoding="utf-8")
    return payload


def source_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, bool]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    memberships: dict[str, dict[str, bool]] = {}
    molecule_coordinates: dict[str, list[torch.Tensor]] = defaultdict(list)
    molecule_references: dict[str, torch.Tensor] = {}
    for source in iter_prediction_records(CACHE / "source/prediction_manifest.json"):
        item = source["item"]
        record = source["record"]
        coordinates = torch.as_tensor(item.x_input).detach().cpu()
        references = torch.as_tensor(
            record.get("x_ref_candidates", record.get("x_ref_aligned")), dtype=torch.float32
        )
        if references.ndim == 2:
            references = references.unsqueeze(0)
        identity = str(source["sample_id"])
        rows.append(
            {
                "record_index": int(source["record_index"]),
                "sample_id": identity,
                "molecule_id": str(source["molecule_id"]),
                "accepted": 1.0,
                "weighted_bac_delta": 0.0,
                "bond_delta": 0.0,
                "angle_delta": 0.0,
                "active_angle_delta": 0.0,
                "clash_delta": 0.0,
                "ring_delta": 0.0,
                "chirality_preserved": 1.0,
                "mean_displacement": 0.0,
                "max_atom_displacement": 0.0,
                "solver_failure_count": 0.0,
                "rmsd": float(rmsd_matrix([coordinates], references).min()),
                "target_loss": float(torch.nn.functional.smooth_l1_loss(coordinates, item.x_target)),
            }
        )
        memberships[identity] = {
            **_memberships(item),
            "chirality_applicable": bool(item.protected_chirality_constraint_index.numel()),
        }
        molecule = str(source["molecule_id"])
        molecule_coordinates[molecule].append(coordinates)
        molecule_references.setdefault(molecule, references)
    set_rows = []
    for molecule, coordinates in molecule_coordinates.items():
        matrix = rmsd_matrix(coordinates, molecule_references[molecule])
        set_rows.append(
            {
                "MAT_P": float(matrix.min(1).values.mean()),
                "MAT_R": float(matrix.min(0).values.mean()),
                "COV_P": float((matrix.min(1).values < 1.25).float().mean()),
                "COV_R": float((matrix.min(0).values < 1.25).float().mean()),
            }
        )
    set_metrics = {
        key: float(np.mean([row[key] for row in set_rows])) for key in SET_METRICS
    }
    return rows, memberships, set_metrics


def applicable(metric: str, membership: dict[str, bool]) -> bool:
    if metric in {"angle_delta", "active_angle_delta"}:
        return membership["active_angle"]
    if metric == "clash_delta":
        return membership["active_clash"]
    if metric == "ring_delta":
        return membership["ring_risk"]
    if metric == "chirality_preserved":
        return membership["chirality_applicable"]
    return True


def paired_statistics(values: np.ndarray, *, draws: int = 10000, seed: int = 43) -> dict[str, Any]:
    if not values.size:
        return {
            "paired_mean_difference": None,
            "median_difference": None,
            "bootstrap_ci95_low": None,
            "bootstrap_ci95_high": None,
            "bootstrap_draws": draws,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "paired_mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "bootstrap_ci95_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_draws": draws,
    }


def direction(metric: str, value: float | None) -> str:
    if value is None or abs(value) <= TOLERANCE.get(metric, 1.0e-12):
        return "tie"
    better = value > 0 if metric in HIGHER_IS_BETTER else value < 0
    return "V8_12P5K_better" if better else "V8_12P5K_worse"


def build_comparison() -> dict[str, Any]:
    reports = {name: read_json(path) for name, path in REPORT_PATHS.items()}
    rows = {name: report["per_record_metrics"] for name, report in reports.items()}
    source, memberships, source_set = source_rows()
    rows = {"Source": source, **rows}
    identity = [(row["record_index"], row["sample_id"]) for row in source]
    for method, values in rows.items():
        if [(row["record_index"], row["sample_id"]) for row in values] != identity:
            raise RuntimeError(f"formal-large record identity/order changed: {method}")
    for method, directory in (("Source", "source"), ("D1", "d1"), ("V5-B", "v5_b"), ("V7", "v7")):
        manifest = read_json(CACHE / directory / "prediction_manifest.json")
        if manifest["identity"]["identity_sha256"] != EXPECTED_CACHE_IDENTITIES[method]:
            raise RuntimeError(f"frozen cache identity changed: {method}")
    summaries: dict[str, Any] = {}
    for method, values in rows.items():
        metric_summary = {
            metric: float(np.mean([float(row[metric]) for row in values])) for metric in METRICS
        }
        set_metrics = source_set if method == "Source" else reports[method]["set_metrics"]
        summaries[method] = {**metric_summary, **set_metrics}
    candidate = rows["V8 Full 12.5K"]
    comparisons: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for baseline in ("Source", "D1", "V5-B", "V7", "V8 Full 5K"):
        comparison_name = f"V8 Full 12.5K minus {baseline}"
        comparisons[comparison_name] = {}
        for cohort in COHORTS:
            cohort_result: dict[str, Any] = {}
            for metric in METRICS:
                selected = [
                    index
                    for index, row in enumerate(candidate)
                    if memberships[row["sample_id"]][cohort]
                    and applicable(metric, memberships[row["sample_id"]])
                ]
                values = np.asarray(
                    [
                        float(candidate[index][metric]) - float(rows[baseline][index][metric])
                        for index in selected
                    ],
                    dtype=np.float64,
                )
                stats = paired_statistics(values)
                tolerance = TOLERANCE.get(metric, 1.0e-12)
                if metric in HIGHER_IS_BETTER:
                    wins = int((values > tolerance).sum())
                    losses = int((values < -tolerance).sum())
                else:
                    wins = int((values < -tolerance).sum())
                    losses = int((values > tolerance).sum())
                ties = int(len(values) - wins - losses)
                low, high = stats["bootstrap_ci95_low"], stats["bootstrap_ci95_high"]
                significant = low is not None and (low > 0 or high < 0)
                effect = direction(metric, stats["paired_mean_difference"])
                result = {
                    **stats,
                    "win_count": wins,
                    "tie_count": ties,
                    "loss_count": losses,
                    "applicable_record_count": len(selected),
                    "significance_status": (
                        f"SIGNIFICANT_{effect}" if significant else "NOT_SIGNIFICANT"
                    ),
                    "effect_direction": effect,
                }
                cohort_result[metric] = result
                csv_rows.append(
                    {
                        "row_type": "paired_comparison",
                        "method": comparison_name,
                        "cohort": cohort,
                        "metric": metric,
                        **result,
                    }
                )
            comparisons[comparison_name][cohort] = cohort_result
    output = {
        "schema_version": "mcvr-v8-full-12p5k-unified-baseline-comparison-v1",
        "status": "COMPLETED",
        "records": 10000,
        "record_identity_and_order_equal": True,
        "methods": list(rows),
        "method_summaries": summaries,
        "paired_comparisons": comparisons,
        "cache_identities": EXPECTED_CACHE_IDENTITIES,
        "evaluator": "frozen_v7_bac_safety_weighted_thresholded_validity",
        "bootstrap_draws": 10000,
        "conclusions": {
            "A_proven": (
                "V8 Full 12.5K significantly improves weighted BAC, bond, active-angle, "
                "and ring deltas relative to Source on the same 10K records; acceptance "
                "and movement trade-offs are reported separately."
            ),
            "B_not_yet_proven": (
                "The frozen D1 checkpoint is not a strict matched control. Attribution to "
                "V8 constraints requires the matched D1-only 12.5K exposure-control run."
            ),
        },
        **ISOLATION,
    }
    atomic_json(REPORT_DIR / "V8_FULL_12P5K_BASELINE_COMPARISON.json", output)
    for method, summary in summaries.items():
        for metric, value in summary.items():
            csv_rows.append(
                {
                    "row_type": "method_summary",
                    "method": method,
                    "cohort": "natural",
                    "metric": metric,
                    "paired_mean_difference": value,
                }
            )
    columns = sorted({key for row in csv_rows for key in row})
    with (REPORT_DIR / "V8_FULL_12P5K_BASELINE_COMPARISON.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    method_header = "| Method | Acceptance | Weighted BAC | Bond | Angle | Active angle | Clash | Ring | Chirality | Mean disp. | Max disp. | Solver fail | RMSD | MAT-P | MAT-R | COV-P | COV-R |"
    method_rows = []
    for method, summary in summaries.items():
        values = [
            method,
            *(f"{summary[key]:.8g}" for key in (
                "accepted", "weighted_bac_delta", "bond_delta", "angle_delta",
                "active_angle_delta", "clash_delta", "ring_delta", "chirality_preserved",
                "mean_displacement", "max_atom_displacement", "solver_failure_count", "rmsd",
                "MAT_P", "MAT_R", "COV_P", "COV_R",
            )),
        ]
        method_rows.append("| " + " | ".join(values) + " |")
    primary_rows = []
    for name, cohorts in comparisons.items():
        for metric in ("accepted", "weighted_bac_delta", "bond_delta", "active_angle_delta", "ring_delta", "rmsd"):
            value = cohorts["natural"][metric]
            primary_rows.append(
                f"| {name} | {metric} | {value['paired_mean_difference']:.8g} | "
                f"{value['median_difference']:.8g} | [{value['bootstrap_ci95_low']:.8g}, "
                f"{value['bootstrap_ci95_high']:.8g}] | {value['win_count']}/"
                f"{value['tie_count']}/{value['loss_count']} | {value['applicable_record_count']} | "
                f"{value['significance_status']} |"
            )
    markdown = "\n".join(
        [
            "# V8 Full 12.5K unified formal-large baseline comparison",
            "",
            "All methods use the same ordered 10K validation records and frozen evaluator. "
            "Non-applicable records are excluded from metric/cohort denominators.",
            "",
            method_header,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *method_rows,
            "",
            "## Primary paired results (V8 12.5K minus baseline)",
            "",
            "| Comparison | Metric | Mean | Median | Bootstrap 95% CI | W/T/L | Applicable | Status |",
            "|---|---|---:|---:|---|---:|---:|---|",
            *primary_rows,
            "",
            "## Conclusions",
            "",
            "**A — proven:** V8 Full 12.5K improves weighted BAC, bond, active-angle, and ring "
            "metrics relative to Source on the same records. Acceptance and movement trade-offs "
            "remain explicit in the table.",
            "",
            "**B — not yet proven:** frozen D1 is older and is not a strict matched control. No "
            "causal attribution to the V8 constraint module is made before the matched D1-only "
            "12.5K exposure-control run.",
        ]
    )
    (REPORT_DIR / "V8_FULL_12P5K_BASELINE_COMPARISON.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    freeze = freeze_v8()
    comparison = build_comparison()
    print(
        json.dumps(
            {
                "freeze_status": freeze["status"],
                "comparison_status": comparison["status"],
                "report_dir": str(REPORT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
