#!/usr/bin/env python3
"""Measure Prototype A component use without evaluating reference metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etflow.ecir.bac_evaluation import attach_canonical_constraints  # noqa: E402
from etflow.ecir.chemical_validity import ChemicalValidity  # noqa: E402
from etflow.ecir.mvr_v5_constraint_hybrid import (  # noqa: E402
    MCVRConstraintMultiHeadModel,
)
from etflow.ecir.run_a_evaluation import build_items  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "diagnostics/ecir_mvr/v5_constraint_hybrid/runs/"
            "v5_a_pilot_1000step_seed43018/checkpoint_final.ckpt"
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_recovery/manifests"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v5_constraint_hybrid/component_diagnostics.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    result = {}
    for group in ("all", "bond_active", "angle_active", "clash_active"):
        selected = rows if group == "all" else [row for row in rows if row[group] > 0]
        if not selected:
            result[group] = {"graph_steps": 0}
            continue
        names = [
            "bond_allocation",
            "angle_allocation",
            "clash_allocation",
            "bond_component_rms",
            "angle_component_rms",
            "clash_component_rms",
            "constraint_fused_rms",
            "prior_rms",
            "activity_gate",
            "trust_gate",
        ]
        result[group] = {
            "graph_steps": len(selected),
            **{name: float(np.mean([row[name] for row in selected])) for name in names},
        }
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    for name in (
        "formal_root",
        "source_cache_root",
        "checkpoint",
        "manifest_dir",
        "output",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MCVRConstraintMultiHeadModel(**config["model"], **config["prototype_a"]).to(args.device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Prototype A diagnostic strict-load failed")
    model.eval()
    validity = ChemicalValidity("data/ecir_mvr/validity_reference_stats.json")
    metadata = json.loads(
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
        raise RuntimeError("Prototype A diagnostic development count changed")
    attach_canonical_constraints(
        items,
        validity,
        source_identity_sha256=metadata["formal_source_identity_sha256"],
    )
    rows: list[dict[str, float]] = []
    for start in range(0, len(items), args.batch_size):
        selected = items[start : start + args.batch_size]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(args.device)
        current = batch.x_init.clone()
        ptr = batch.ptr.tolist()
        for step in range(4):
            output = model(
                batch,
                current,
                current.new_full((batch.num_graphs,), 1.0 - step / 4.0),
            )
            for graph_index, item in enumerate(selected):
                left, right = ptr[graph_index], ptr[graph_index + 1]
                active = item["data"].active_mode_mask.reshape(-1)
                allocation = output["constraint_allocation"][left:right].mean(0)
                component_rms = []
                for name in ("bond", "angle", "clash"):
                    value = output[f"v_{name}_component"][left:right]
                    component_rms.append(float(torch.sqrt(value.square().sum(-1).mean())))
                fused = output["v_constraint_fused"][left:right]
                prior = output["v_cartesian_raw"][left:right]
                rows.append(
                    {
                        "step": float(step + 1),
                        "bond_active": float(active[0]),
                        "angle_active": float(active[1]),
                        "clash_active": float(active[3]),
                        "bond_allocation": float(allocation[0]),
                        "angle_allocation": float(allocation[1]),
                        "clash_allocation": float(allocation[2]),
                        "bond_component_rms": component_rms[0],
                        "angle_component_rms": component_rms[1],
                        "clash_component_rms": component_rms[2],
                        "constraint_fused_rms": float(torch.sqrt(fused.square().sum(-1).mean())),
                        "prior_rms": float(torch.sqrt(prior.square().sum(-1).mean())),
                        "activity_gate": float(
                            output["constraint_activity_gate"][left:right].mean()
                        ),
                        "trust_gate": float(output["constraint_trust_gate"][left:right].mean()),
                    }
                )
            current = current + 0.25 * output["v_final"]
    report = {
        "schema_version": "mcvr-v5-component-diagnostics-v1",
        "records": 1024,
        "graph_steps": len(rows),
        "summary": _summary(rows),
        "model_inputs_label_free": True,
        "reference_metrics_computed": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
