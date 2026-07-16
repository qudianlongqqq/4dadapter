#!/usr/bin/env python
"""Audit ECIR performance on a source condition excluded from training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--held_out_checkpoint")
    parser.add_argument("--held_out_nfe", type=int)
    parser.add_argument("--held_out_seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_internal_improvements", type=int, default=2)
    parser.add_argument("--rmsd_margin", type=float, default=0.02)
    args = parser.parse_args()
    atlas = pd.read_parquet(args.atlas)
    held_out = pd.Series(True, index=atlas.index)
    conditions = {}
    for column, value in (
        ("checkpoint", args.held_out_checkpoint),
        ("NFE", args.held_out_nfe),
        ("seed", args.held_out_seed),
    ):
        if value is not None:
            held_out &= atlas[column] == value
            conditions[column] = value
    if not conditions or not held_out.any() or not (~held_out).any():
        raise ValueError("Held-out conditions must select a nonempty strict subset")
    train_ids = set(atlas.loc[~held_out, "sample_id"].astype(str))
    test_ids = set(atlas.loc[held_out, "sample_id"].astype(str))
    if train_ids & test_ids:
        raise ValueError("leave-one-source-out sample leakage")
    evaluation = pd.read_csv(args.evaluation)
    evaluation = evaluation[evaluation["sample_id"].astype(str).isin(test_ids)]
    pivot = evaluation.pivot_table(index="molecule_id", columns="method", values=["bond_violation", "angle_violation", "torsion_circular_error", "clash_score", "aligned_RMSD"])
    deltas = {}
    for metric in pivot.columns.levels[0]:
        if (metric, "upstream") not in pivot or (metric, "ECIR_4step_teacher") not in pivot:
            continue
        delta = (pivot[(metric, "ECIR_4step_teacher")] - pivot[(metric, "upstream")]).dropna().to_numpy()
        deltas[metric] = {"mean": float(delta.mean()), "molecules": int(delta.size)}
    internal = [
        metric
        for metric in ("bond_violation", "angle_violation", "torsion_circular_error", "clash_score")
        if metric in deltas and deltas[metric]["mean"] < 0.0
    ]
    status = (
        "PASS"
        if len(internal) >= args.min_internal_improvements
        and deltas.get("aligned_RMSD", {}).get("mean", float("inf")) <= args.rmsd_margin
        else "FAIL"
    )
    result = {
        "status": status,
        "held_out": conditions,
        "train_records": len(train_ids),
        "held_out_records": len(test_ids),
        "sample_leakage": False,
        "internal_metrics_improved": internal,
        "candidate_minus_upstream": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
