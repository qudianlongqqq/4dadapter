#!/usr/bin/env python
"""Compare new 5k rollout results with the existing old 5k run, read-only."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = ROOT / "logs_flexbond_formal_small/flexbond4d_hybrid_5k"
NEW_BEST = ROOT / "diagnostics/global_coupled_4d/checkpoint_sweep_5k/best_checkpoint.json"
OUTPUT_CSV = ROOT / "reports/global_coupled_4d_5k_comparison.csv"
OUTPUT_MD = ROOT / "reports/global_coupled_4d_5k_comparison.md"


def _metadata(summary_path: Path) -> dict:
    path = summary_path.with_suffix(".json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "manifest": str(payload.get("manifest", "")),
            "threshold": str(payload.get("threshold", "")),
        }
    except Exception:
        return {"manifest": "", "threshold": ""}


def _rows(path: Path, method: str):
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return [row for row in csv.DictReader(handle) if row.get("method") == method and row.get("subset") == "all"]
    except Exception:
        return []


def _key(row):
    return (
        float(row["failure_rate"]), float(row["rmsd_mean"]),
        float(row["MAT-R"]), float(row["MAT-P"]),
        -float(row["COV-R"]), -float(row["COV-P"]),
    )


def main():
    old_checkpoint_count = len(list(OLD_RUN.rglob("*.ckpt"))) if OLD_RUN.is_dir() else 0
    old_sample_count = len(list(OLD_RUN.rglob("*.pt"))) if OLD_RUN.is_dir() else 0
    old_summary_count = len(list(OLD_RUN.rglob("summary.csv"))) if OLD_RUN.is_dir() else 0
    old_candidates = []
    if OLD_RUN.is_dir():
        for path in OLD_RUN.rglob("summary.csv"):
            for row in _rows(path, "flexbond4d_adapter"):
                old_candidates.append((row, path))
    old = min(old_candidates, key=lambda item: _key(item[0])) if old_candidates else None
    new = None
    if NEW_BEST.is_file():
        best = json.loads(NEW_BEST.read_text(encoding="utf-8"))
        path = Path(best["summary_path"])
        rows = _rows(path, "global_coupled_4d_adapter")
        if rows:
            new = (rows[0], path)
    comparable = bool(old and new and _metadata(old[1]) == _metadata(new[1]) and all(_metadata(old[1]).values()))
    output = []
    for method, result in (("Legacy FlexBond-4D 5k", old), ("Global Coupled 4D 5k", new)):
        if result is None:
            output.append({
                "method": method, "status": "OLD_RESULT_MISSING" if method.startswith("Legacy") else "NEW_RESULT_MISSING",
                "rmsd_mean": "", "COV-R": "", "COV-P": "", "MAT-R": "", "MAT-P": "",
                "failure_rate": "", "comparison_status": "NOT_DIRECTLY_COMPARABLE", "summary_path": "",
            })
        else:
            row, path = result
            output.append({
                "method": method, "status": "FOUND", "rmsd_mean": row["rmsd_mean"],
                "COV-R": row["COV-R"], "COV-P": row["COV-P"], "MAT-R": row["MAT-R"],
                "MAT-P": row["MAT-P"], "failure_rate": row["failure_rate"],
                "comparison_status": "FAIR_DIRECT_COMPARISON" if comparable else "NOT_DIRECTLY_COMPARABLE",
                "summary_path": str(path),
            })
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    lines = ["# Global Coupled 4D 5k comparison", "",
             f"Status: **{'FAIR_DIRECT_COMPARISON' if comparable else 'NOT_DIRECTLY_COMPARABLE'}**", "",
             "> The pipeline never trains or resamples the old FlexBond-4D model.", "",
             f"Old read-only inventory: `{old_checkpoint_count}` checkpoints, `{old_sample_count}` sample files, `{old_summary_count}` evaluation summaries.", "",
             "| method | result | RMSD | COV-R | COV-P | MAT-R | MAT-P | failure |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {row['method']} | {row['status']} | {row['rmsd_mean']} | {row['COV-R']} | {row['COV-P']} | {row['MAT-R']} | {row['MAT-P']} | {row['failure_rate']} |" for row in output)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote read-only 5k comparison reports.")


if __name__ == "__main__":
    main()
