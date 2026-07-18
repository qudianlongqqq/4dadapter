"""Validation-only inference and metrics for unified BAC candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Batch

from .bac_constraints import canonical_constraint_fields
from .bac_safety import (
    BACSafetyConfig,
    evaluate_bac_proposal,
    select_safe_bac_proposal,
)
from .run_a_evaluation import method_rows, paired_bootstrap, summarize_groups


def attach_canonical_constraints(
    items: Sequence[dict[str, Any]],
    validity: Any,
    *,
    source_identity_sha256: str,
) -> None:
    for item in items:
        fields = canonical_constraint_fields(
            validity,
            item["record"],
            source_identity_sha256=source_identity_sha256,
        )
        for name, value in fields.items():
            setattr(item["data"], name, value)


@torch.inference_mode()
def infer_bac(
    model: torch.nn.Module,
    items: Sequence[dict[str, Any]],
    validity: Any,
    *,
    device: torch.device,
    steps: int = 4,
    step_size: float = 0.25,
    batch_size: int = 64,
    safety_config: BACSafetyConfig | None = None,
) -> tuple[list[Tensor], list[dict[str, Any]]]:
    config = safety_config or BACSafetyConfig()
    coordinates: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(items), int(batch_size)):
        selected = items[start : start + int(batch_size)]
        batch = Batch.from_data_list([item["data"] for item in selected]).to(device)
        current = batch.x_init.clone()
        ptr = batch.ptr.detach().cpu().tolist()
        trajectories: list[list[Tensor]] = [[] for _ in selected]
        step_diagnostics: list[list[dict[str, float]]] = [[] for _ in selected]
        for step in range(int(steps)):
            t = current.new_full((batch.num_graphs,), 1.0 - step / max(int(steps), 1))
            output = model(batch, current, t)
            current = current + float(step_size) * output["v_final"]
            for local in range(len(selected)):
                left, right = ptr[local], ptr[local + 1]
                trajectories[local].append(current[left:right].detach().cpu().clone())
                step_diagnostics[local].append(
                    {
                        "step": step + 1,
                        "global_safety_gate_mean": float(
                            output["global_safety_gate"][local].mean()
                        ),
                        "velocity_norm_mean": float(
                            torch.linalg.vector_norm(
                                output["v_final"][left:right], dim=-1
                            ).mean()
                        ),
                        "angle_gate_mean": float(
                            output.get("angle_gate", current.new_zeros(1)).mean()
                        ),
                        "clash_gate_mean": float(
                            output.get("clash_gate", current.new_zeros(1)).mean()
                        ),
                    }
                )
        for local, item in enumerate(selected):
            source = item["input"]
            candidates = []
            for step, proposal in enumerate(trajectories[local], start=1):
                if config.enable_backtracking:
                    checked, decision = select_safe_bac_proposal(
                        source,
                        proposal - source,
                        item["record"],
                        validity,
                        config,
                    )
                else:
                    checked = proposal
                    decision = evaluate_bac_proposal(
                        source, checked, item["record"], validity, config
                    )
                candidates.append((step, checked, decision))
            safe = [value for value in candidates if value[2]["accepted"]]
            if safe:
                selected_step, accepted, decision = max(
                    safe,
                    key=lambda value: (
                        value[2]["bac_gain"],
                        -value[2]["displacement"]["aligned_rms_displacement"],
                        -value[0],
                    ),
                )
                rolled_back = False
            else:
                selected_step, accepted = 0, source.clone()
                decision = candidates[-1][2] if candidates else {
                    "accepted": False,
                    "reasons": ["no_candidate"],
                }
                rolled_back = True
            coordinates.append(torch.as_tensor(accepted).clone())
            diag = step_diagnostics[local][max(selected_step - 1, 0)]
            metadata.append(
                {
                    "accepted": not rolled_back,
                    "selected_step": selected_step,
                    "reject_reasons": ";".join(decision.get("reasons", [])),
                    "rolled_back": rolled_back,
                    "bac_gain": float(decision.get("bac_gain", 0.0)),
                    "backtracking_enabled": bool(config.enable_backtracking),
                    "selected_scale": float(decision.get("selected_scale", 1.0)),
                    "backtracking_attempts": len(decision.get("attempts", [])),
                    **diag,
                }
            )
    return coordinates, metadata


def evaluate_bac_candidate(
    model: torch.nn.Module,
    items: Sequence[dict[str, Any]],
    validity: Any,
    *,
    device: torch.device,
    inference: Mapping[str, Any],
    source_identity_sha256: str,
    bootstrap_draws: int = 500,
) -> dict[str, Any]:
    attach_canonical_constraints(
        items, validity, source_identity_sha256=source_identity_sha256
    )
    accepted, metadata = infer_bac(
        model,
        items,
        validity,
        device=device,
        steps=int(inference.get("teacher_steps", 4)),
        step_size=float(inference.get("step_size", 0.25)),
        batch_size=int(inference.get("batch_size", 64)),
        safety_config=BACSafetyConfig(**dict(inference.get("safety", {}))),
    )
    methods = {
        "upstream": [item["input"] for item in items],
        "v2_bac_accepted": accepted,
    }
    records = method_rows(
        items,
        methods,
        validity,
        method_metadata={"v2_bac_accepted": metadata},
    )
    summary, molecules = summarize_groups(records, items, methods)
    bootstrap = paired_bootstrap(
        molecules,
        candidate="v2_bac_accepted",
        draws=int(bootstrap_draws),
    )
    all_rows = summary[summary.group == "all"].set_index("method")
    candidate = all_rows.loc["v2_bac_accepted"]
    baseline = all_rows.loc["upstream"]
    return {
        "records": records,
        "molecules": molecules,
        "summary": summary,
        "bootstrap": bootstrap,
        "metrics": {
            "bond_delta": float(
                candidate.bond_outlier_rate - baseline.bond_outlier_rate
            ),
            "angle_delta": float(
                candidate.angle_outlier_rate - baseline.angle_outlier_rate
            ),
            "clash_delta": float(
                candidate.clash_penetration - baseline.clash_penetration
            ),
            "ring_delta": float(
                candidate.ring_bond_outlier_rate
                - baseline.ring_bond_outlier_rate
            ),
            "rmsd_delta": float(candidate.aligned_RMSD - baseline.aligned_RMSD),
            "mat_p_delta": float(candidate.MAT_P - baseline.MAT_P),
            "mat_r_delta": float(candidate.MAT_R - baseline.MAT_R),
            "cov_p_delta": float(candidate.COV_P - baseline.COV_P),
            "cov_r_delta": float(candidate.COV_R - baseline.COV_R),
            "accepted_fraction": float(candidate.accepted_fraction),
            "rollback_fraction": float(candidate.rejected_fraction),
            "mean_displacement": float(candidate.molecule_rms_displacement),
            "failure_rate": 0.0,
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }


def summary_json(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": dict(result["metrics"]),
        "bootstrap": dict(result["bootstrap"]),
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }


def write_evaluation(result: Mapping[str, Any], output: str) -> None:
    path = pd.io.common.stringify_path(output)
    result["records"].to_csv(f"{path}_per_record.csv", index=False)
    result["molecules"].to_csv(f"{path}_per_molecule.csv", index=False)
    result["summary"].to_csv(f"{path}_summary.csv", index=False)
