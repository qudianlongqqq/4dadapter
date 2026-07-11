#!/usr/bin/env python
"""Generate fail-closed training-budget and historical-method comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _write_csv(path, rows, default_fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else default_fields
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _budget_comparison():
    reference_path = REPORTS / "reference_4d_training_budget.json"
    config_path = ROOT / "configs/global_coupled_4d_local025_matched.yaml"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.is_file() else {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    global_budget = {
        "max_steps": config["trainer"]["max_steps"],
        "batch_size": config["data"]["batch_size"],
        "accumulate_grad_batches": config["trainer"]["accumulate_grad_batches"],
        "effective_batch_size": config["data"]["batch_size"] * config["trainer"]["accumulate_grad_batches"],
        "learning_rate": config["optimizer"]["lr"],
        "scheduler": config["optimizer"].get("scheduler", "none"),
        "t_min": config["time_sampling"]["t_min"], "t_max": config["time_sampling"]["t_max"],
        "seed": config["seed"], "precision": str(config["trainer"].get("precision", "unknown")),
    }
    rows, all_match = [], reference.get("confidence") == "high" and not reference.get("ambiguous")
    for field, value in global_budget.items():
        old = reference.get(field)
        match = str(old).lower() == str(value).lower()
        try:
            match = match or abs(float(old) - float(value)) < 1e-12
        except (TypeError, ValueError):
            pass
        rows.append({"field": field, "reference_4d": old, "global_coupled_4d": value, "match": match})
        all_match = all_match and match
    _write_csv(REPORTS / "global_coupled_4d_training_budget_comparison.csv", rows,
               ["field", "reference_4d", "global_coupled_4d", "match"])
    label = "FAIR_DIRECT_COMPARISON" if all_match else "NOT_DIRECTLY_COMPARABLE"
    lines = ["# Global Coupled 4D training budget comparison", "", f"Status: **{label}**", "",
             f"Reference confidence: `{reference.get('confidence', 'none')}`", "",
             "| field | reference 4D | global coupled 4D | match |", "|---|---|---|---|"]
    lines.extend(f"| {row['field']} | {row['reference_4d']} | {row['global_coupled_4d']} | {row['match']} |" for row in rows)
    (REPORTS / "global_coupled_4d_training_budget_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_match


def _method_name(value):
    mapping = {
        "cartesian_adapter": "Cartesian local025",
        "flexbond4d_adapter": "Legacy FlexBond-4D",
        "gated_kinematic_adapter": "Gated Kinematic 1D",
        "global_coupled_4d_adapter": "Global Coupled 4D",
    }
    return mapping.get(value)


def _final_comparison(budget_match):
    evaluations = []
    for path in list(ROOT.glob("diagnostics/**/summary.csv")) + list(ROOT.glob("logs*/**/summary.csv")):
        try:
            rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        except Exception:
            continue
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        signature = {
            "manifest": metadata.get("manifest"),
            "threshold": metadata.get("threshold"),
            "reference_cache": (metadata.get("provenance") or {}).get("cache_path"),
        }
        for row in rows:
            display = _method_name(row.get("method"))
            if display and row.get("subset") == "all":
                evaluations.append({"method_display": display, "summary_path": str(path),
                                    "signature": json.dumps(signature, sort_keys=True), **row})
    global_rows = [row for row in evaluations if row["method"] == "global_coupled_4d_adapter"]
    reference_signature = global_rows[-1]["signature"] if global_rows else None
    output = []
    for row in evaluations:
        fair = bool(reference_signature and row["signature"] == reference_signature and budget_match)
        output.append({
            "method": row["method_display"], "subset": row.get("subset"),
            "rmsd_mean": row.get("rmsd_mean"), "COV-R": row.get("COV-R"),
            "COV-P": row.get("COV-P"), "MAT-R": row.get("MAT-R"),
            "MAT-P": row.get("MAT-P"), "failure_rate": row.get("failure_rate"),
            "comparison_status": "FAIR_DIRECT_COMPARISON" if fair else "NOT_DIRECTLY_COMPARABLE",
            "summary_path": row["summary_path"],
        })
    _write_csv(REPORTS / "global_coupled_4d_final_comparison.csv", output,
               ["method", "subset", "rmsd_mean", "COV-R", "COV-P", "MAT-R", "MAT-P", "failure_rate", "comparison_status", "summary_path"])
    lines = ["# Global Coupled 4D final comparison", ""]
    if not output:
        lines.extend(["Status: **NOT_DIRECTLY_COMPARABLE**", "", "No completed compatible rollout evaluations were found."])
    else:
        lines.extend(["Only rows explicitly marked `FAIR_DIRECT_COMPARISON` may be ranked.", "",
                      "| method | RMSD | COV-R | COV-P | MAT-R | MAT-P | failure | status |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"])
        lines.extend(f"| {r['method']} | {r['rmsd_mean']} | {r['COV-R']} | {r['COV-P']} | {r['MAT-R']} | {r['MAT-P']} | {r['failure_rate']} | {r['comparison_status']} |" for r in output)
    (REPORTS / "global_coupled_4d_final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    budget_match = _budget_comparison()
    _final_comparison(budget_match)
    print("Wrote fail-closed Global Coupled 4D comparison reports.")


if __name__ == "__main__":
    main()

