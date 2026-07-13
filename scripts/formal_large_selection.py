#!/usr/bin/env python
"""Create validation cohorts, rank candidates, and freeze formal-large configs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    checkpoint_inference_identity,
)
from etflow.formal_large import (
    CONFIRM_MAX_RECORDS,
    REFINEMENT_STEPS,
    SCREEN_MAX_RECORDS,
    SEED,
    canonical_sha256,
    file_sha256,
    select_stratified_manifest,
    top_candidates,
)


COUNTS = {"screen10": {"low": 2, "medium": 3, "high": 5},
          "confirm30": {"low": 5, "medium": 10, "high": 15}}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("**/eval/summary.csv")):
        group = path.parent.parent.name
        checkpoint_match = re.search(r"step(\d+)", group)
        alpha_match = re.search(r"alpha(\d+)", group)
        method = path.parent.parent.parent.name
        with path.open(encoding="utf-8-sig") as handle:
            values = list(csv.DictReader(handle))
        all_row = next(
            row for row in values
            if row.get("subset") == "all" and row.get("method") != "upstream_only"
        )
        high = next(
            (row for row in values if row.get("subset") == "rotatable_ge_6"
             and row.get("method") == all_row.get("method")),
            {},
        )
        rows.append({
            "method": method,
            "checkpoint_step": int(checkpoint_match.group(1)),
            "alpha": float(f"0.{alpha_match.group(1).lstrip('0')}"),
            "failure_rate": float(all_row["failure_rate"]),
            "rmsd_mean": float(all_row["rmsd_mean"]),
            "MAT-P": float(all_row["MAT-P"]),
            "MAT-R": float(all_row["MAT-R"]),
            "COV-P": float(all_row["COV-P"]),
            "COV-R": float(all_row["COV-R"]),
            "high_flex_rmsd": float(high.get("rmsd_mean", "inf")),
            "summary_path": str(path.resolve()),
        })
    return rows


def _write_table(rows: list[dict], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(row[key]) for key in columns) + " |" for row in rows]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-manifest")
    create.add_argument("--kind", choices=tuple(COUNTS), required=True)
    create.add_argument("--source", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--max_records", type=int)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--kind", choices=tuple(COUNTS), required=True)
    summarize.add_argument("--root", required=True, type=Path)
    summarize.add_argument("--output", required=True, type=Path)
    summarize.add_argument("--best_configs", type=Path)
    args = parser.parse_args()

    if args.command == "create-manifest":
        maximum = args.max_records
        if maximum is None:
            maximum = (
                SCREEN_MAX_RECORDS
                if args.kind == "screen10"
                else CONFIRM_MAX_RECORDS
            )
        manifest = select_stratified_manifest(
            _load_json(args.source),
            COUNTS[args.kind],
            seed=SEED,
            max_records=maximum,
        )
        atomic_json_save(manifest, args.output)
        atomic_json_save(
            manifest["selection_report"],
            args.output.with_suffix(".selection.json"),
        )
        print(
            f"Wrote {args.kind} manifest with "
            f"{manifest['selection_report']['selected_molecule_count']} molecules and "
            f"{len(manifest['records'])} records"
        )
        return

    rows = _evaluation_rows(args.root)
    methods = {row["method"] for row in rows}
    expected = 8 if args.kind == "screen10" else 2
    for method in methods:
        count = sum(row["method"] == method for row in rows)
        if count != expected:
            raise ValueError(f"Expected {expected} {method} evaluations, found {count}")
    ranked = []
    for method in sorted(methods):
        for rank, row in enumerate(top_candidates(
            [value for value in rows if value["method"] == method], len(rows)
        ), 1):
            ranked.append({"rank": rank, **row})
    _write_table(ranked, args.output, f"Formal-large {args.kind}")
    atomic_json_save({
        "selection_split": "validation",
        "test_used_for_selection": False,
        "rows": ranked,
        "top2": {
            method: top_candidates(
                [row for row in rows if row["method"] == method], 2
            )
            for method in sorted(methods)
        },
    }, args.output.with_suffix(".json"))

    if args.kind == "confirm30":
        if args.best_configs is None:
            raise ValueError("--best_configs is required for confirm30")
        validation_manifest = _load_json(Path("manifests/formal_large_val_confirm30.json"))
        frozen = {}
        for method in sorted(methods):
            best = top_candidates(
                [row for row in rows if row["method"] == method], 1
            )[0]
            run = Path(
                "logs_formal_large/cartesian_seed42_200k"
                if method == "cartesian"
                else "logs_formal_large/global4d_seed42_200k"
            )
            checkpoint = run / "checkpoints" / f"step{best['checkpoint_step']}.ckpt"
            config = run / "config.resolved.yaml"
            identity = checkpoint_inference_identity(checkpoint)
            frozen[method] = {
                "method": method,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_file_sha256": identity["file_sha256"],
                "checkpoint_inference_sha256": identity["inference_sha256"],
                "alpha": best["alpha"],
                "refinement_steps": REFINEMENT_STEPS,
                "config_path": str(config.resolve()),
                "config_file_sha256": file_sha256(config),
                "validation_manifest_sha256": canonical_sha256(validation_manifest),
                "seed": SEED,
                "selection_metrics": best,
                "selected_at": datetime.now(timezone.utc).isoformat(),
                "selection_split": "validation",
                "test_used_for_selection": False,
            }
        atomic_json_save({"configs": frozen}, args.best_configs)


if __name__ == "__main__":
    main()
