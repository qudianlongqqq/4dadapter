#!/usr/bin/env python
"""Strict paired reports for external refinement and frozen neural caches."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
from rdkit.Chem import AllChem

from etflow.ecir.external_refinement_baselines import ISOLATION, derive_total_charge, derive_unpaired_electrons
from etflow.ecir.v8_validation_cache import atomic_json, file_sha256, iter_prediction_records
from scripts.evaluate_ecir_mvr_v8_prediction_cache import _memberships


METRICS = (
    "accepted", "weighted_bac_delta", "bond_delta", "angle_delta",
    "active_angle_delta", "ring_delta", "clash_delta", "chirality_preserved",
    "mean_displacement", "max_atom_displacement", "rmsd", "target_loss",
)
HIGHER_IS_BETTER = {"accepted", "chirality_preserved"}
PAIRINGS = (
    ("V8_FULL_12P5K", "RAW"),
    ("V8_FULL_12P5K", "MMFF94S"),
    ("V8_FULL_12P5K", "GFN2_XTB"),
    ("V8_FULL_12P5K", "MATCHED_D1_12P5K"),
    ("MMFF94S", "RAW"),
    ("GFN2_XTB", "RAW"),
    ("GFN2_XTB", "MMFF94S"),
)


def stats(values: np.ndarray, *, draws: int) -> dict[str, Any]:
    if not len(values):
        return {"paired_mean_difference": None, "median_difference": None, "bootstrap_ci95_low": None, "bootstrap_ci95_high": None, "draws": draws}
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "paired_mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "bootstrap_ci95_low": None,
        "bootstrap_ci95_high": None,
        "draws": draws,
        "standardized_mean_effect": float(values.mean() / std) if std > 0 else 0.0,
    }


def bootstrap_many(values: np.ndarray, *, draws: int, seed: int = 43) -> tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap many metric rows with one index stream per cohort mask."""
    rows, records = values.shape
    if not records:
        return np.full(rows, np.nan), np.full(rows, np.nan)
    rng = np.random.default_rng(seed)
    sampled = np.empty((rows, draws), dtype=np.float64)
    # Count vectors are exactly equivalent to sampling record indices with
    # replacement. ``einsum(optimize=False)`` intentionally avoids a native
    # BLAS crash observed in this frozen Windows NumPy environment.
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, records, size=(count, records))
        weights = np.stack(
            [np.bincount(row, minlength=records) for row in indices]
        ).astype(np.float64)
        sampled[:, start : start + count] = (
            np.einsum("rn,dn->rd", values, weights, optimize=False) / records
        )
    return np.quantile(sampled, 0.025, axis=1), np.quantile(sampled, 0.975, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("SMOKE100", "FAST1000", "FULL10K"), required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", required=True, help="METHOD=path")
    parser.add_argument("--prediction", action="append", default=[], help="METHOD=manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    eval_paths = {value.split("=", 1)[0]: Path(value.split("=", 1)[1]).resolve() for value in args.evaluation}
    pred_paths = {value.split("=", 1)[0]: Path(value.split("=", 1)[1]).resolve() for value in args.prediction}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in eval_paths.items()}
    rows = {name: report["per_record_metrics"] for name, report in reports.items()}
    required_methods = {name for pair in PAIRINGS for name in pair}
    if not required_methods.issubset(rows):
        raise RuntimeError(f"missing comparison methods: {sorted(required_methods - set(rows))}")
    identity = [(int(row["record_index"]), str(row["sample_id"])) for row in rows["RAW"]]
    for method, values in rows.items():
        if [(int(row["record_index"]), str(row["sample_id"])) for row in values] != identity:
            raise RuntimeError(f"paired record identity/order changed: {method}")
    external = {name: {int(row["record_index"]): row for row in iter_prediction_records(path)} for name, path in pred_paths.items()}
    source_by_id = {str(row["sample_id"]): row for row in iter_prediction_records(args.source_cache_manifest.resolve()) if str(row["sample_id"]) in {sample for _, sample in identity}}
    memberships = {}
    for index, sample in identity:
        source = source_by_id[sample]
        mol = source["record"]["_formal_rdkit_mol"]
        base = _memberships(source["item"])
        memberships[sample] = {
            **base,
            "chirality_applicable": bool(source["item"].protected_chirality_constraint_index.numel()),
            "mmff_supported": bool(AllChem.MMFFHasAllMoleculeParams(mol)),
            "xtb_converged": bool(external.get("GFN2_XTB", {}).get(index, {}).get("converged", False)),
            "charged": derive_total_charge(mol) != 0,
            "radical_open_shell": derive_unpaired_electrons(mol) != 0,
        }
    cohorts = ("natural", "active_angle", "active_clash", "ring_risk", "high_flexibility", "low_error_minimal_movement", "mmff_supported", "xtb_converged", "charged", "radical_open_shell")
    comparisons = {}
    pending: dict[tuple[int, ...], list[tuple[dict[str, Any], np.ndarray, str, str]]] = {}
    for left, right in PAIRINGS:
        name = f"{left}-minus-{right}"
        comparisons[name] = {}
        for cohort in cohorts:
            comparisons[name][cohort] = {}
            for metric in METRICS:
                selected = []
                selected_offsets = []
                for offset, (_, sample) in enumerate(identity):
                    member = memberships[sample]
                    if cohort != "natural" and not member[cohort]:
                        continue
                    if metric == "active_angle_delta" and not member["active_angle"]:
                        continue
                    if metric == "clash_delta" and not member["active_clash"]:
                        continue
                    if metric == "ring_delta" and not member["ring_risk"]:
                        continue
                    if metric == "chirality_preserved" and not member["chirality_applicable"]:
                        continue
                    selected.append(float(rows[left][offset][metric]) - float(rows[right][offset][metric]))
                    selected_offsets.append(offset)
                values = np.asarray(selected, dtype=np.float64)
                result = stats(values, draws=args.bootstrap_draws)
                tolerance = 1.0e-12
                if metric in HIGHER_IS_BETTER:
                    wins, losses = int((values > tolerance).sum()), int((values < -tolerance).sum())
                else:
                    wins, losses = int((values < -tolerance).sum()), int((values > tolerance).sum())
                result.update({"win_count": wins, "tie_count": int(len(values)-wins-losses), "loss_count": losses, "applicable_record_count": len(values)})
                comparisons[name][cohort][metric] = result
                pending.setdefault(tuple(selected_offsets), []).append((result, values, left, metric))
    for selected_offsets, entries in pending.items():
        if not selected_offsets:
            for result, _, _, _ in entries:
                result["statistical_status"] = "NOT_APPLICABLE"
            continue
        matrix = np.stack([values for _, values, _, _ in entries])
        lows, highs = bootstrap_many(matrix, draws=args.bootstrap_draws)
        for index, (result, _, left, metric) in enumerate(entries):
            low, high = float(lows[index]), float(highs[index])
            result["bootstrap_ci95_low"] = low
            result["bootstrap_ci95_high"] = high
            significant = low > 0 or high < 0
            mean = result["paired_mean_difference"]
            favorable = mean is not None and ((mean > 0) if metric in HIGHER_IS_BETTER else (mean < 0))
            result["statistical_status"] = "NOT_SIGNIFICANT" if not significant else (f"SIGNIFICANT_{left}_BETTER" if favorable else f"SIGNIFICANT_{left}_WORSE")
    csv_rows = [
        {"comparison": comparison, "cohort": cohort, "metric": metric, **result}
        for comparison, cohort_values in comparisons.items()
        for cohort, metric_values in cohort_values.items()
        for metric, result in metric_values.items()
    ]
    summaries = {name: {**report["metrics"], **(report.get("set_metrics") or {}), "external_refinement": report.get("external_refinement")} for name, report in reports.items()}
    output = {
        "schema_version": "mcvr-external-refinement-paired-comparison-v1",
        "status": f"MCVR_EXTERNAL_REFINEMENT_{args.phase}_COMPLETED",
        "phase": args.phase,
        "records": len(identity),
        "record_identity_and_order_equal": True,
        "method_summaries": summaries,
        "paired_comparisons": comparisons,
        "bootstrap_draws": args.bootstrap_draws,
        "evaluation_sha256": {name: file_sha256(path) for name, path in eval_paths.items()},
        "native_energies_not_cross_compared": True,
        **ISOLATION,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.phase}_RESULTS"
    atomic_json(args.output_dir / f"{stem}.json", output)
    atomic_json(args.output_dir / "PAIRED_COMPARISONS.json", output)
    columns = sorted({key for row in csv_rows for key in row})
    for path in (args.output_dir / f"{stem}.csv", args.output_dir / "PAIRED_COMPARISONS.csv"):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(csv_rows)
    header = "| Method | Success | Fallback | Accept | Weighted BAC | Bond | Angle | Active angle | Ring | Clash | Chirality | Mean disp. | RMSD | MAT-P | MAT-R | COV-P | COV-R |"
    lines = [f"# MCVR external refinement {args.phase}", "", header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, summary in summaries.items():
        ext = summary.get("external_refinement") or {}
        def f(key: str) -> str:
            value = summary.get(key); return "n/a" if value is None else f"{float(value):.8g}"
        lines.append("| " + " | ".join([method, f"{ext.get('success_rate', 1.0):.6g}", f"{ext.get('fallback_rate', 0.0):.6g}", f("accepted"), f("weighted_bac_delta"), f("bond_delta"), f("angle_delta"), f("active_angle_delta"), f("ring_delta"), f("clash_delta"), f("chirality_preserved"), f("mean_displacement"), f("rmsd"), f("MAT_P"), f("MAT_R"), f("COV_P"), f("COV_R")]) + " |")
    lines.extend(["", "All methods use the same ordered frozen Source records and evaluator. External failures remain in the all-record result via bitwise Source fallback.", "", "Native MMFF94s and GFN2-xTB energies are retained only within their own method and are not compared across methods."])
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "PAIRED_COMPARISONS.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": output["status"], "records": len(identity)}, indent=2))


if __name__ == "__main__":
    main()
