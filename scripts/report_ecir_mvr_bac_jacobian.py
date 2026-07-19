#!/usr/bin/env python3
"""Enrich the frozen J0 result without rerunning either coordinate method."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_constraints import canonical_constraint_fields  # noqa: E402
from etflow.ecir.bac_jacobian import (  # noqa: E402
    JacobianBACConfig,
    build_constraint_system,
    constraint_type_statistics,
)
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.run_a_evaluation import build_items  # noqa: E402


EXPECTED_MANIFEST_IDENTITY = (
    "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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
        "--output-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/jacobian_bac"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("formal_root", "source_cache_root", "manifest_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    manifest = json.loads(
        (args.manifest_dir / "recovery_development_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["identity_sha256"] != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError("Jacobian report development manifest changed")
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["decision"] != "JACOBIAN_NOT_SUPPORTED":
        raise RuntimeError("Jacobian result is not the frozen J0 decision")
    raw = [
        json.loads(line)
        for line in (args.output_dir / "solver_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records = pd.read_csv(args.output_dir / "per_record.csv")
    source_rows = records[records.method == "upstream"].set_index("sample_id")
    candidate_rows = records[records.method == "j0_jacobian"].set_index("sample_id")
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
    if len(items) != len(raw) or len(items) != 1024:
        raise RuntimeError("Jacobian reporting record identity/count mismatch")
    config = JacobianBACConfig()
    enriched = []
    metric_names = {
        "bond": "bond_outlier_rate",
        "angle": "angle_outlier_rate",
        "clash": "clash_penetration",
        "ring_bond": "ring_bond_outlier_rate",
        "ring_planarity": "ring_planarity_outlier_rate",
        "chirality": "chirality_error",
    }
    for item, row in zip(items, raw, strict=True):
        sample_id = str(item["row"].sample_id)
        if row["sample_id"] != sample_id:
            raise RuntimeError("Jacobian diagnostic row order changed")
        static = canonical_constraint_fields(
            validity,
            item["record"],
            source_identity_sha256=source_metadata["formal_source_identity_sha256"],
        )
        system = build_constraint_system(
            item["input"],
            static["active_bond_constraint_index"],
            static["bond_allowed_range"],
            static["active_angle_constraint_index"].t(),
            static["angle_allowed_range"],
            config,
        )
        before = source_rows.loc[sample_id]
        after = candidate_rows.loc[sample_id]
        metric_before_after = {
            name: {
                "before": float(before[column]),
                "after": float(after[column]),
                "delta": float(after[column] - before[column]),
            }
            for name, column in metric_names.items()
        }
        enriched.append(
            {
                **row,
                "jacobian_shape": system.diagnostics["jacobian_shape"],
                "active_residuals_by_type": constraint_type_statistics(system),
                "metric_before_after": metric_before_after,
                "ring_hard_safety_preserved": bool(
                    metric_before_after["ring_bond"]["delta"] <= 0.0
                    and metric_before_after["ring_planarity"]["delta"] <= 0.0
                ),
                "chirality_hard_safety_preserved": bool(
                    metric_before_after["chirality"]["delta"] <= 0.0
                ),
                "damping": config.damping_lambda,
                "test_records_read": 0,
                "test_assets_opened": False,
                "frozen_holdout_records_opened": 0,
            }
        )
    with (args.output_dir / "solver_diagnostics_enriched.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in enriched:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    reject_reasons = Counter()
    for value in candidate_rows.reject_reasons.dropna():
        reject_reasons.update(str(value).split(";"))
    failure = {
        "schema_version": "mcvr-v3-bac-jacobian-failure-analysis-v1",
        "solver_status_counts": dict(Counter(row["solver_status"] for row in raw)),
        "final_reject_reason_counts": dict(reject_reasons),
        "solver_failure_count": summary["j0_solver"]["solver_failure_count"],
        "solver_failure_rate": summary["j0_solver"]["solver_failure_rate"],
        "interpretation": (
            "J0 rejection is dominated by no BAC gain and hard non-regression, "
            "not nonfinite or factorization failure."
        ),
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    _write_json(args.output_dir / "failure_analysis.json", failure)
    recommendation = {
        "schema_version": "mcvr-v3-bac-jacobian-decision-v1",
        "decision": "JACOBIAN_NOT_SUPPORTED",
        "start_learned_jacobian": False,
        "start_j1": False,
        "start_10k": False,
        "start_formal_large": False,
        "retain_best_cartesian_candidate": "D1",
        "reason": (
            "J0 has additional active-Angle gain at lower movement, but sacrifices "
            "Bond/weighted BAC and acceptance relative to D1."
        ),
        "clash_conclusion": "INCONCLUSIVE_DATA_SUPPORT",
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
    }
    _write_json(args.output_dir / "recommended_next_step.json", recommendation)
    markdown = f"""# MCVR V3-BAC Jacobian Summary

Decision: `JACOBIAN_NOT_SUPPORTED`

- Records/molecules: {summary['records']}/{summary['molecules']}
- J0 acceptance: {summary['j0_metrics']['acceptance_fraction']:.4%}
- J0 Bond/Angle delta: {summary['j0_metrics']['bond_delta']:.8f} / {summary['j0_metrics']['angle_delta']:.8f}
- J0 weighted BAC delta: {summary['j0_active_subsets']['all']['metrics']['weighted_bac_delta']['mean']:.8f}
- J0 mean displacement: {summary['j0_metrics']['mean_displacement']:.8f} Angstrom
- Solver failure: {summary['j0_solver']['solver_failure_count']}/{summary['records']}
- D1 acceptance: {summary['d1_reproduction']['metrics']['accepted_fraction']:.4%}
- D1 Bond/Angle delta: {summary['d1_reproduction']['metrics']['bond_delta']:.8f} / {summary['d1_reproduction']['metrics']['angle_delta']:.8f}
- test_records_read: 0
- test_assets_opened: false
- frozen_holdout_records_opened: 0

J0 improves active Angle beyond D1 with less movement, but loses most Bond and
weighted BAC gain and has much lower acceptance. J1, learned Jacobian, 10k, and
formal-large are not authorized.
"""
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(recommendation, sort_keys=True))


if __name__ == "__main__":
    main()
