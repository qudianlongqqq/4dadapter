#!/usr/bin/env python
"""Report the one retained 5k engineering result after it is fully valid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


DEFAULT_GROUP = Path(
    "diagnostics/global_coupled_4d/checkpoint_sweep_5k/step1000_alpha02"
)


def _summary_path(group: Path) -> Path:
    candidates = (group / "summary.csv", group / "eval/summary.csv")
    return next(
        (path for path in candidates if path.is_file() and path.stat().st_size),
        candidates[-1],
    )


def load_valid_result(group: Path) -> dict:
    samples_path = group / "samples.pt"
    summary_path = _summary_path(group)
    if not samples_path.is_file() or not samples_path.stat().st_size:
        raise ValueError("First-group samples.pt is absent or empty")
    if not summary_path.is_file() or not summary_path.stat().st_size:
        raise ValueError("First-group summary.csv is absent or empty")
    payload = torch.load(samples_path, map_location="cpu", weights_only=False)
    if payload.get("partial") is True or not payload.get("records"):
        raise ValueError("First-group sample payload is partial or empty")
    provenance = payload.get("manifest_provenance", {})
    manifest_hash = provenance.get("manifest", {}).get("sha256")
    if not manifest_hash or int(provenance.get("sample_count", -1)) != len(payload["records"]):
        raise ValueError("First-group sample provenance is invalid")
    with summary_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    global_row = next(
        row for row in rows
        if row.get("method") == "global_coupled_4d_adapter" and row.get("subset") == "all"
    )
    raw_row = next(
        (
            row
            for row in rows
            if row.get("method") == "upstream_only" and row.get("subset") == "all"
        ),
        None,
    )
    state_path = group / "sampling_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    output = {
        "checkpoint": str(payload["records"][0].get("checkpoint_path")),
        "alpha": float(payload["records"][0].get("alpha", 0.2)),
        "manifest_sha256": manifest_hash,
        "sample_record_count": len(payload["records"]),
        "molecule_count": len(
            {
                str(row.get("source_mol_id", row.get("mol_id")))
                for row in payload["records"]
            }
        ),
        "RMSD": float(global_row["rmsd_mean"]),
        "MAT-P": float(global_row["MAT-P"]),
        "MAT-R": float(global_row["MAT-R"]),
        "COV-P": float(global_row["COV-P"]),
        "COV-R": float(global_row["COV-R"]),
        "failure_rate": float(global_row["failure_rate"]),
        "sampling_seconds": state.get("total_seconds"),
        "solver_backend": payload.get("solver_backend_counts", state.get("solver_backend_counts")),
        "cache_hit_rate": state.get("topology_cache_hit_rate"),
        "summary_path": str(summary_path.resolve()),
        "samples_path": str(samples_path.resolve()),
        "engineering_only_not_formal": True,
    }
    if raw_row is not None:
        output["etflow_raw"] = {
            "RMSD": float(raw_row["rmsd_mean"]),
            "MAT-P": float(raw_row["MAT-P"]),
            "MAT-R": float(raw_row["MAT-R"]),
            "COV-P": float(raw_row["COV-P"]),
            "COV-R": float(raw_row["COV-R"]),
            "failure_rate": float(raw_row["failure_rate"]),
            "same_manifest_sha256": manifest_hash,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = load_valid_result(args.group)
    if args.check_only:
        print("VALID")
        return
    Path("reports").mkdir(exist_ok=True)
    Path("reports/global_coupled_4d_first_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    Path("reports/global_coupled_4d_first_result.md").write_text(
        "# Global Coupled 4D first 5k result\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
